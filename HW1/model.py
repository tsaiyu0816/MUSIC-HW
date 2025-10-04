import torch
import torch.nn as nn
import torch.nn.functional as F

# ---- 小積木：兩層 3x3 conv + BN + ReLU，再接 2x2 pool（可參考 PANNs 的 ConvBlock）----
class Conv_2d(nn.Module):
    def __init__(self, in_ch, out_ch, pooling=2, p_drop=0.0):
        super().__init__()
        self.net = nn.Sequential(
            nn.Conv2d(in_ch, out_ch, kernel_size=3, padding=1, bias=False),
            nn.BatchNorm2d(out_ch),
            nn.ReLU(inplace=True),
            nn.Conv2d(out_ch, out_ch, kernel_size=3, padding=1, bias=False),
            nn.BatchNorm2d(out_ch),
            nn.ReLU(inplace=True),
        )
        if pooling and pooling > 1:
            self.pool = nn.MaxPool2d(kernel_size=pooling, stride=pooling)
        else:
            self.pool = nn.Identity()
        self.drop = nn.Dropout2d(p_drop) if p_drop > 0 else nn.Identity()

    def forward(self, x):
        x = self.net(x)
        x = self.pool(x)
        x = self.drop(x)
        return x

# ---- 版本：吃「已算好的特徵」(B,1,H,T)，取前 n_mels 列當成 Mel 圖餵下去 ----
class SCNN(nn.Module):
    """
    Short-Chunk CNN（vgg-ish）改成吃 feature：(B,1,H,T)
    會切 x[:, :, :n_mels, :] 當 Mel-spectrogram。
    預設 7 個 block，每層 2x2 pool → freq 維 128 -> 1，因此最後 squeeze(2) OK。
    輸出為 logits（別加 Sigmoid），直接配 CrossEntropyLoss。
    """
    def __init__(self,
                 n_class: int,
                 n_mels: int = 128,
                 n_channels: int = 128,
                 p_drop: float = 0.5):
        super().__init__()
        self.n_mels = int(n_mels)

        # 輸入就是「1 channel 的頻譜圖」，先做一下 BN（等同你原始程式的 spec_bn）
        self.spec_bn = nn.BatchNorm2d(1)

        # CNN backbone（跟你貼的 ShortChunkCNN 結構一致）
        self.layer1 = Conv_2d(1,             n_channels,     pooling=2)
        self.layer2 = Conv_2d(n_channels,    n_channels,     pooling=2)
        self.layer3 = Conv_2d(n_channels,    n_channels * 2, pooling=2)
        self.layer4 = Conv_2d(n_channels * 2,n_channels * 2, pooling=2)
        self.layer5 = Conv_2d(n_channels * 2,n_channels * 2, pooling=2)
        self.layer6 = Conv_2d(n_channels * 2,n_channels * 2, pooling=2)
        self.layer7 = Conv_2d(n_channels * 2,n_channels * 4, pooling=2)

        # Dense head（輸出 logits）
        self.fc1 = nn.Linear(n_channels * 4, n_channels * 4)
        self.bn1 = nn.BatchNorm1d(n_channels * 4)
        self.relu = nn.ReLU(inplace=True)
        self.drop = nn.Dropout(p_drop)
        self.fc2 = nn.Linear(n_channels * 4, n_class)

    def forward(self, x):
        """
        x: (B,1,H,T)，其中最前面的 H 行是 Mel dB（你的 FeatureComputer 產出的第一段）
        我們只取前 n_mels 行來符合 ShortChunk 設計（128 -> 1 after 7 pools）
        """
        B, C, H, T = x.shape
        if H < self.n_mels:
            raise ValueError(f"Input freq bins {H} < n_mels {self.n_mels}. "
                             f"請確認 cfg.n_mels 與特徵維度一致（通常是 128）。")

        x = x[:, :, :self.n_mels, :]  # 取 Mel 部分 (B,1,128,T)
        x = self.spec_bn(x)

        x = self.layer1(x)
        x = self.layer2(x)
        x = self.layer3(x)
        x = self.layer4(x)
        x = self.layer5(x)
        x = self.layer6(x)
        x = self.layer7(x)            # 預期 shape: (B, n_channels*4, 1, T')

        x = x.squeeze(2)              # -> (B, n_channels*4, T')
        if x.size(-1) != 1:
            x = F.max_pool1d(x, kernel_size=x.size(-1))  # Global max pool over time
        x = x.squeeze(-1)             # -> (B, n_channels*4)

        x = self.fc1(x)
        x = self.bn1(x)
        x = self.relu(x)
        x = self.drop(x)
        logits = self.fc2(x)          # raw logits, 不要加 Sigmoid/Softmax
        return logits
    

class SEBlock(nn.Module):
    def __init__(self, ch, r=16):
        super().__init__()
        self.pool = nn.AdaptiveAvgPool2d(1)
        self.fc = nn.Sequential(
            nn.Linear(ch, max(ch // r, 8), bias=False),
            nn.ReLU(inplace=True),
            nn.Linear(max(ch // r, 8), ch, bias=False),
            nn.Sigmoid(),
        )
    def forward(self, x):
        b, c, _, _ = x.size()
        y = self.pool(x).view(b, c)
        y = self.fc(y).view(b, c, 1, 1)
        return x * y

class ResBlock(nn.Module):
    """兩個 3x3 conv，帶殘差；首個 conv 可 stride=2 做下採樣。"""
    def __init__(self, in_ch, out_ch, stride=1, use_se=False, p_drop2d=0.0):
        super().__init__()
        self.conv1 = nn.Conv2d(in_ch, out_ch, 3, stride=stride, padding=1, bias=False)
        self.bn1   = nn.BatchNorm2d(out_ch)
        self.conv2 = nn.Conv2d(out_ch, out_ch, 3, padding=1, bias=False)
        self.bn2   = nn.BatchNorm2d(out_ch)
        self.se    = SEBlock(out_ch) if use_se else nn.Identity()
        self.drop2d = nn.Dropout2d(p_drop2d) if p_drop2d > 0 else nn.Identity()

        if stride != 1 or in_ch != out_ch:
            self.down = nn.Sequential(
                nn.Conv2d(in_ch, out_ch, 1, stride=stride, bias=False),
                nn.BatchNorm2d(out_ch),
            )
        else:
            self.down = nn.Identity()

    def forward(self, x):
        identity = self.down(x)
        out = F.relu(self.bn1(self.conv1(x)), inplace=True)
        out = self.bn2(self.conv2(out))
        out = self.se(out)
        out = out + identity
        out = F.relu(out, inplace=True)
        out = self.drop2d(out)
        return out

class SCNN_gpt(nn.Module):
    """
    改良版 Short-Chunk CNN：輸入 (B,1,M,T)，輸出 logits (B,n_classes)
    """
    def __init__(self, n_classes: int):
        super().__init__()
        # Stem
        self.in_bn = nn.BatchNorm2d(1)

        # 4 個 stage，逐步下採樣 (stride=2)
        self.stage1 = ResBlock(1,   64,  stride=2, use_se=False, p_drop2d=0.05)
        self.stage2 = ResBlock(64,  128, stride=2, use_se=False, p_drop2d=0.10)
        self.stage3 = ResBlock(128, 192, stride=2, use_se=True,  p_drop2d=0.10)
        self.stage4 = ResBlock(192, 256, stride=2, use_se=True,  p_drop2d=0.15)

        # 雙池化（Avg + Max）並聯
        self.gap = nn.AdaptiveAvgPool2d((1, 1))
        self.gmp = nn.AdaptiveMaxPool2d((1, 1))

        # Head：兩層全連接（保持 logits，別加 Sigmoid/Softmax）
        self.head = nn.Sequential(
            nn.Linear(256 * 2, 256),
            nn.BatchNorm1d(256),
            nn.ReLU(inplace=True),
            nn.Dropout(0.4),
            nn.Linear(256, n_classes),
        )

    def forward(self, x):
        # x: (B,1,M,T)
        x = self.in_bn(x)
        x = self.stage1(x)
        x = self.stage2(x)
        x = self.stage3(x)
        x = self.stage4(x)

        a = self.gap(x)                      # (B,256,1,1)
        m = self.gmp(x)                      # (B,256,1,1)
        x = torch.cat([a, m], dim=1).flatten(1)  # (B,512)

        logits = self.head(x)                # (B,n_classes), raw logits
        return logits

