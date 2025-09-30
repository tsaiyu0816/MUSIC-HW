# -*- coding: utf-8 -*-
# file: task2_train_main.py
import os, argparse, json, time
import numpy as np
from collections import defaultdict

import torch
import torch.nn as nn
import torch.optim as optim
from sklearn.metrics import confusion_matrix, classification_report, accuracy_score
import torch.backends.cudnn as cudnn
from task2_dataload import *
from tqdm import tqdm

# =========================
# Model: Short-Chunk CNN
# =========================
class SCNN(nn.Module):
    """
    輕量 short-chunk CNN：輸入 (B,1,M,T)
    """
    def __init__(self, n_classes: int):
        super().__init__()
        self.net = nn.Sequential(
            nn.Conv2d(1, 32, kernel_size=3, padding=1), nn.BatchNorm2d(32), nn.ReLU(),
            nn.MaxPool2d(kernel_size=(2,2)),
            nn.Conv2d(32, 64, kernel_size=3, padding=1), nn.BatchNorm2d(64), nn.ReLU(),
            nn.MaxPool2d(kernel_size=(2,2)),
            nn.Conv2d(64, 128, kernel_size=3, padding=1), nn.BatchNorm2d(128), nn.ReLU(),
            nn.MaxPool2d(kernel_size=(2,2)),
            nn.Conv2d(128, 256, kernel_size=3, padding=1), nn.BatchNorm2d(256), nn.ReLU(),
            nn.AdaptiveAvgPool2d((1,1)),  # 全域平均
        )
        self.head = nn.Sequential(
            nn.Flatten(),
            nn.Dropout(0.3),
            nn.Linear(256, n_classes)
        )

    def forward(self, x):
        h = self.net(x)
        out = self.head(h)
        return out

# =========================
# Train / Eval
# =========================
def train_one_epoch(model, loader, optimizer, criterion, device):
    model.train()
    total, correct = 0, 0
    losses = []

    pbar = tqdm(loader, desc="train", leave=False, dynamic_ncols=True)
    for x, y, _ in pbar:
        x = x.to(device, non_blocking=True)
        y = y.to(device, non_blocking=True)

        # 兼容舊版 torch
        try:
            optimizer.zero_grad(set_to_none=True)
        except TypeError:
            optimizer.zero_grad()

        logits = model(x)
        loss = criterion(logits, y)
        loss.backward()
        optimizer.step()

        losses.append(loss.item())
        with torch.no_grad():
            pred = logits.argmax(1)
            correct += (pred == y).sum().item()
            total   += y.numel()

        # tqdm 即時顯示
        pbar.set_postfix(loss=f"{losses[-1]:.4f}", acc=f"{(correct/total):.4f}")

    return float(np.mean(losses)), correct / total

@torch.no_grad()
def eval_chunks(model, loader, device, n_classes: int):
    model.eval()
    probs, labels, keys = [], [], []
    for x, y, k in loader:
        x = x.to(device, non_blocking=True)
        logits = model(x)
        p = torch.softmax(logits, dim=1).cpu().numpy()
        probs.append(p)
        labels.append(y.numpy())
        keys.extend(k)
    proba = np.vstack(probs)
    labs  = np.concatenate(labels)
    pred1 = proba.argmax(1)
    acc1  = accuracy_score(labs, pred1)
    acc3  = topk_from_proba(labs, proba, k=3)
    return proba, labs, keys, acc1, acc3

# =========================
# Main
# =========================
def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--train_json", type=str, required=True)
    ap.add_argument("--val_json",   type=str, required=True)
    ap.add_argument("--test_json",  type=str, default=None)     # 若需測試
    ap.add_argument("--segment_sec", type=int, default=30)
    ap.add_argument("--overlap",     type=float, default=0.0)
    ap.add_argument("--batch_size",  type=int, default=64)
    ap.add_argument("--epochs",      type=int, default=40)
    ap.add_argument("--lr",          type=float, default=1e-3)
    ap.add_argument("--patience",    type=int, default=6, help="early stopping patience")
    ap.add_argument("--vote_method", type=str, default="mean", choices=["mean","majority"])
    ap.add_argument("--ckpt_path",   type=str, default="checkpoints/task2_scnn.pt")
    ap.add_argument("--report_dir",  type=str, default="reports_task2")
    ap.add_argument("--num_workers", type=int, default=4)
    ap.add_argument("--pin_memory", action="store_true")

    args = ap.parse_args()

    os.makedirs(args.report_dir, exist_ok=True)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    
    # ===== dataloaders =====
    cfg = CFG(segment_sec=args.segment_sec, overlap=args.overlap, batch_size=args.batch_size)
    # 先從 train_json 中建立 classes
    classes = build_classes_from_json(args.train_json)
    dl_tr, dl_va, idx_tr, idx_va, _ = make_loaders(args.train_json, args.val_json, cfg, classes=classes)
    print("Train chunks =", len(dl_tr.dataset))
    print("Val   chunks =", len(dl_va.dataset))
    n_classes = len(classes)

    # ===== model / opt =====
    model = SCNN(n_classes=n_classes).to(device)
    optimizer = optim.Adam(model.parameters(), lr=args.lr)
    criterion = nn.CrossEntropyLoss()

    # ===== train with early stopping =====
    best_acc, best_epoch, wait = 0.0, 0, 0
    best_state = None
    epoch_bar = tqdm(range(1, args.epochs + 1), desc="Epochs", dynamic_ncols=True)
    for epoch in epoch_bar:
        # train
        tr_loss, tr_acc = train_one_epoch(model, dl_tr, optimizer, criterion, device)

        # chunk-level val
        proba_va, y_va, keys_va, acc1_va, acc3_va = eval_chunks(model, dl_va, device, n_classes)

        # track-level voting（val）
        y_track, P_track, _ = aggregate_by_keys(proba_va, y_va, keys_va, n_classes, method=args.vote_method)
        acc1_track = (P_track.argmax(1) == y_track).mean()
        acc3_track = topk_from_proba(y_track, P_track, k=3)

        # 更新 tqdm 顯示
        epoch_bar.set_postfix(
            loss=f"{tr_loss:.4f}",
            tr1=f"{tr_acc:.4f}",
            va1=f"{acc1_va:.4f}",
            va3=f"{acc3_va:.4f}",
            t1=f"{acc1_track:.4f}",
            t3=f"{acc3_track:.4f}",
        )

        # 也印一行（避免 tqdm 捲走）
        tqdm.write(
            f"[{epoch:03d}] loss {tr_loss:.4f} | tr@1 {tr_acc:.4f} | "
            f"va@1 {acc1_va:.4f} va@3 {acc3_va:.4f} | "
            f"track@1 {acc1_track:.4f} track@3 {acc3_track:.4f}"
        )

        # early stopping 以 track@1 為主
        if acc1_track > best_acc:
            best_acc, best_epoch, wait = float(acc1_track), epoch, 0
            best_state = {
                "model": model.state_dict(),
                "classes": classes,
                "cfg": vars(cfg),
            }

            # 立刻存 checkpoint（選配）
            ckpt_dir = getattr(args, "ckpt_dir", "checkpoints")
            os.makedirs(ckpt_dir, exist_ok=True)
            ckpt_path = os.path.join(ckpt_dir, "task2_best.pt")
            torch.save(best_state, ckpt_path)
            tqdm.write(f"[checkpoint] saved: {ckpt_path} (track@1={best_acc:.4f})")
        else:
            wait += 1
            if wait >= args.patience:
                tqdm.write(
                    f"Early stopped at epoch {epoch}, best track@1={best_acc:.4f} (epoch {best_epoch})"
                )
                break

    # ===== save best =====
    if best_state is None:
        best_state = { "model": model.state_dict(), "classes": classes, "cfg": vars(cfg) }
    os.makedirs(os.path.dirname(args.ckpt_path), exist_ok=True)
    torch.save(best_state, args.ckpt_path)
    print(f"[CKPT] saved to {args.ckpt_path}")

    # ===== 最終 validation 報告（用 best state） =====
    model.load_state_dict(best_state["model"])
    proba_va, y_va, keys_va, acc1_va, acc3_va = eval_chunks(model, dl_va, device, n_classes)
    y_track, P_track, _ = aggregate_by_keys(proba_va, y_va, keys_va, n_classes, method=args.vote_method)
    y_pred_chunk = proba_va.argmax(1)
    y_pred_track = P_track.argmax(1)

    print("\n=== VALIDATION (chunk-level) ===")
    print(f"Top-1: {acc1_va:.4f} | Top-3: {acc3_va:.4f}")
    print("Classification report (chunk):")
    print(classification_report(y_va, y_pred_chunk, target_names=classes, digits=4, zero_division=0))

    print("\n=== VALIDATION (track-level, voting) ===")
    acc1_track = (y_pred_track == y_track).mean()
    acc3_track = topk_from_proba(y_track, P_track, k=3)
    print(f"Top-1: {acc1_track:.4f} | Top-3: {acc3_track:.4f}")
    print("Classification report (track):")
    print(classification_report(y_track, y_pred_track, target_names=classes, digits=4, zero_division=0))

    # 混淆矩陣圖片
    import matplotlib.pyplot as plt
    def plot_cm(cm, labels, outpng):
        fig, ax = plt.subplots(figsize=(6,6))
        im = ax.imshow(cm, interpolation='nearest')
        ax.figure.colorbar(im, ax=ax)
        ax.set(xticks=np.arange(cm.shape[1]), yticks=np.arange(cm.shape[0]),
               xticklabels=labels, yticklabels=labels, ylabel='True', xlabel='Predicted')
        plt.setp(ax.get_xticklabels(), rotation=45, ha="right", rotation_mode="anchor")
        ax.set_title("Confusion Matrix")
        plt.tight_layout(); plt.savefig(outpng, dpi=160); plt.close(fig)

    cm_chunk = confusion_matrix(y_va, y_pred_chunk)
    plot_cm(cm_chunk, classes, os.path.join(args.report_dir, "cm_val_chunk.png"))
    cm_track = confusion_matrix(y_track, y_pred_track)
    plot_cm(cm_track, classes, os.path.join(args.report_dir, "cm_val_track.png"))

    # ===== TEST（可選）→ 匯出 top-3 JSON（track-level） =====
    if args.test_json:
        idx_te = build_index(args.test_json, cfg, classes)
        dl_te = DataLoader(ChunkDatasetFixed(idx_te, cfg), batch_size=cfg.batch_size,
                           shuffle=False, num_workers=cfg.num_workers, pin_memory=True)
        proba_te, labs_dummy, keys_te, _, _ = eval_chunks(model, dl_te, device, n_classes)
        # 聚合
        buckets = defaultdict(list)
        for i, k in enumerate(keys_te):
            buckets[k].append(i)
        results: Dict[str, List[str]] = {}
        for k, idxs in buckets.items():
            p = proba_te[idxs].mean(axis=0)
            top3 = np.argsort(p)[-3:][::-1]
            results[k] = [classes[i] for i in top3]
        out_json = os.path.join(args.report_dir, "task2_test_top3.json")
        with open(out_json, "w", encoding="utf-8") as f:
            json.dump(results, f, ensure_ascii=False, indent=2)
        print(f"[TEST] wrote top-3 JSON to: {out_json}")

if __name__ == "__main__":
    main()
