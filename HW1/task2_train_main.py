# -*- coding: utf-8 -*-
# file: task2_train_main.py
import os, argparse, json
import numpy as np
import matplotlib.pyplot as plt
import math
from collections import defaultdict

import torch
import torch.nn as nn
import torch.optim as optim
from sklearn.metrics import confusion_matrix, classification_report, accuracy_score
from task2_dataload import _separate_vocals_demucs_mem, _chunk_quality
from task2_dataload import *
from model import SCNN
from tqdm import tqdm
from sklearn.manifold import TSNE
import subprocess, sys

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

@torch.no_grad()
def collect_logits(model, loader, device):
    model.eval()
    logits_all, labels_all, keys = [], [], []
    for x, y, k in loader:
        x = x.to(device, non_blocking=True)
        z = model(x).detach().cpu()
        logits_all.append(z)
        labels_all.append(y)
        keys.extend(k)
    return torch.cat(logits_all, 0).numpy(), torch.cat(labels_all, 0).numpy(), keys

def _pick_best_chunk(vocal_mono: np.ndarray, cfg: CFG):
    """按照訓練相同切法＋_chunk_quality()，回傳 (best_score, s, e)。"""
    seg_len = int(cfg.segment_sec * cfg.sr)
    step = int(seg_len * (1.0 - cfg.overlap)) if cfg.overlap < 1.0 else 1
    T = vocal_mono.shape[0]
    total = max(1, math.floor((T - seg_len) / step) + 1)
    best = (-1e9, 0, seg_len)
    for k in range(total):
        s = k * step; e = s + seg_len
        seg = vocal_mono[s:e]
        if seg.shape[0] < seg_len:
            seg = np.pad(seg, (0, seg_len - seg.shape[0]))
        sc, _, _, _ = _chunk_quality(seg, cfg)
        if sc > best[0]:
            best = (sc, s, e)
    return best  # (score, s, e)

def _save_mel_png(seg: np.ndarray, cfg: CFG, outpng: str, device: torch.device, start_sample: int = 0):
    """用與訓練一致的 Mel/ToDB 畫圖，x 軸顯示原整段歌曲中的秒數（絕對時間），y=Hz。"""
    comp = FeatureComputer(cfg, device)
    x = torch.from_numpy(seg.astype(np.float32)).unsqueeze(0).to(device)
    with torch.no_grad():
        S = comp.to_db(comp.mel(x)).squeeze(0).cpu().numpy()  # (n_mels, T)

    n_mels, n_frames = S.shape
    # === 絕對時間軸（秒）：chunk 起點換算為秒後加上每個 frame 的位移 ===
    start_sec = start_sample / cfg.sr
    frame_times_sec = start_sec + np.arange(n_frames) * (cfg.hop / cfg.sr)

    # 頻率軸：把 mel bin 對應到 Hz，並挑好讀的 Hz 刻度
    mel_hz = librosa.mel_frequencies(n_mels=n_mels, fmin=cfg.fmin, fmax=cfg.fmax)
    tick_hz_candidates = np.array([100, 200, 500, 1000, 2000, 4000, 8000])
    tick_hz = tick_hz_candidates[(tick_hz_candidates >= cfg.fmin) & (tick_hz_candidates <= cfg.fmax)]
    tick_pos = [int(np.argmin(np.abs(mel_hz - f))) for f in tick_hz]

    # 讓對比更清楚
    vmax = np.max(S)
    vmin = vmax - 80.0

    plt.figure(figsize=(10, 4))
    im = plt.imshow(S, aspect='auto', origin='lower', vmin=vmin, vmax=vmax)
    plt.colorbar(im, label='dB')

    # x 軸：顯示絕對秒數（最多 ~8 個刻度）
    max_xticks = 8
    if n_frames <= max_xticks:
        xtick_pos = np.arange(n_frames)
    else:
        xtick_pos = np.linspace(0, n_frames - 1, num=max_xticks, dtype=int)
    xtick_lab = [f"{frame_times_sec[i]:.1f}" for i in xtick_pos]
    plt.xticks(xtick_pos, xtick_lab)
    plt.xlabel("Time in full track (s)")

    # y 軸：用 Hz 標示
    plt.yticks(tick_pos, [f"{int(f)}" for f in tick_hz])
    plt.ylabel("Frequency (Hz)")

    plt.title("Mel-spectrogram (best chunk)")
    plt.tight_layout()
    plt.savefig(outpng, dpi=160)
    plt.close()



# =========================
# Main
# =========================
def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--train_json", type=str, required=True)
    ap.add_argument("--val_json",   type=str, required=True)
    ap.add_argument("--segment_sec", type=int, default=10)
    ap.add_argument("--overlap",     type=float, default=0.0)
    ap.add_argument("--batch_size",  type=int, default=16)
    ap.add_argument("--epochs",      type=int, default=40)
    ap.add_argument("--lr",          type=float, default=1e-3)
    ap.add_argument("--patience",    type=int, default=10, help="early stopping patience")
    ap.add_argument("--vote_method", type=str, default="mean", choices=["mean","majority"])
    ap.add_argument("--ckpt_path",   type=str, default="checkpoints/task2_scnn.pt")
    ap.add_argument("--report_dir",  type=str, default="reports_task2")
    ap.add_argument("--num_workers", type=int, default=4)
    ap.add_argument("--cache_dir", type=str, default="cache_task2")
    ap.add_argument("--user_drive_url", type=str, default="https://drive.google.com/file/d/1igvMqkJNY4NpH1ZgV2nXPw1AtNT62fNf/view?usp=sharing",
                help="Google Drive 分享連結；若給這個會用 gdown 下載成 user_audio.mp3")

    args = ap.parse_args()

    os.makedirs(args.report_dir, exist_ok=True)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    user_path = os.path.join(args.report_dir, "user_audio.mp3")
    cmd = ["gdown", "--id", "1v15RpbOiRv2GEtGawwQEP6N-oZa_9Agb", "-O", user_path]
    res = subprocess.run(cmd, stdout=sys.stdout, stderr=sys.stderr, check=True)
    
    # ===== dataloaders =====
    cfg = CFG(
    segment_sec=args.segment_sec,
    overlap=args.overlap,
    batch_size=args.batch_size,
    num_workers=args.num_workers,
    cache_dir=args.cache_dir,
    )

    # 先從 train_json 中建立 classes
    classes = build_classes_from_json(args.train_json)
    dl_tr, dl_va, idx_tr, idx_va, _ = make_loaders(args.train_json, args.val_json, cfg, classes=classes)
    print("Train chunks =", len(dl_tr.dataset))
    print("Val   chunks =", len(dl_va.dataset))
    n_classes = len(classes)

    # ===== model / opt ===== 
    model = SCNN(n_class=n_classes, n_mels=cfg.n_mels).to(device)
    optimizer = optim.Adam(model.parameters(), lr=args.lr)
    criterion = nn.CrossEntropyLoss()

    # ===== train with early stopping =====
    best_acc, best_epoch, wait = 0.0, 0, 0
    hist = {'loss':[], 'tr_acc':[], 'va1':[], 'va3':[], 't1':[], 't3':[]}
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

        hist['loss'].append(tr_loss)
        hist['tr_acc'].append(tr_acc)
        hist['va1'].append(acc1_va)
        hist['va3'].append(acc3_va)
        hist['t1'].append(acc1_track)
        hist['t3'].append(acc3_track)

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
            ckpt_path = os.path.join(ckpt_dir, f"task2_best_{acc1_track:.2f}.pt")
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

    def plot_curve(values, ylabel, outpng):
        plt.figure(figsize=(6,4))
        plt.plot(values, marker='o')
        plt.xlabel('epoch'); plt.ylabel(ylabel)
        plt.tight_layout(); plt.savefig(outpng, dpi=160); plt.close()

    plot_curve(hist['loss'],   'train loss', os.path.join(args.report_dir, 'loss_curve.png'))
    plot_curve(hist['tr_acc'], 'train acc',  os.path.join(args.report_dir, 'train_acc.png'))
    plot_curve(hist['va1'],    'val top-1',  os.path.join(args.report_dir, 'val_top1.png'))
    plot_curve(hist['t1'],     'track top-1',os.path.join(args.report_dir, 'track_top1.png'))


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

    # === t-SNE on validation logits（可選併入你的錄音） ===
    logits_va, labs_va, _ = collect_logits(model, dl_va, device)

    # 子樣本（加速 t-SNE）
    max_points = 4000
    if logits_va.shape[0] > max_points:
        sel = np.random.choice(logits_va.shape[0], max_points, replace=False)
        X_val = logits_va[sel]; Y_val = labs_va[sel]
    else:
        X_val = logits_va; Y_val = labs_va

    # ---- (可選) 把你的音檔下載/讀入 → Demucs 分離 → 挑最高分 chunk → 取 logits ----
    user_logit = None
    # 2) 真的有檔案才處理
    if user_path and os.path.isfile(user_path):
        # 用同一流程取得 vocal mono（LRU+Demucs）
        v = _separate_vocals_demucs_mem(user_path, cfg, device)
        sc, s, e = _pick_best_chunk(v, cfg)
        best = v[s:e]
        need = int(cfg.segment_sec * cfg.sr)
        if best.shape[0] < need:
            best = np.pad(best, (0, need - best.shape[0]))

        # 畫 Mel
        mel_png = os.path.join(args.report_dir, "user_best_mel.png")
        _save_mel_png(best, cfg, mel_png, device, start_sample=s)
        print(f"[user_audio] best score={sc:.3f}, time=[{s/cfg.sr:.2f}s ~ {e/cfg.sr:.2f}s] → {mel_png}")

        # 跟驗證一樣的特徵 → 丟 model 取 logits（和 t-SNE 一致的空間）
        comp = FeatureComputer(cfg, device)
        F = comp.compute(best.astype(np.float32), apply_specaug=False)
        x_u = torch.tensor(F, dtype=torch.float32, device=device).unsqueeze(0).unsqueeze(0)
        with torch.no_grad():
            user_logit = model(x_u).detach().cpu().numpy()  # (1, n_classes)

    # ---- 一起做 t-SNE（非參數式，需把 user 一起 fit）----
    X_tsne = X_val if user_logit is None else np.vstack([X_val, user_logit])
    perp = max(5, min(30, (X_tsne.shape[0] - 1) // 3))
    Z = TSNE(n_components=2, init='pca', learning_rate='auto',
            perplexity=perp, n_iter=1000, random_state=42).fit_transform(X_tsne)

    if user_logit is None:
        val_xy, user_xy = Z, None
    else:
        val_xy, user_xy = Z[:-1], Z[-1]

    plt.figure(figsize=(7, 7))
    sc = plt.scatter(val_xy[:, 0], val_xy[:, 1], c=Y_val, s=6, alpha=0.7, cmap="tab20")
    plt.colorbar(sc)
    if user_xy is not None:
        plt.scatter(user_xy[0], user_xy[1], marker="*", s=320,
                    edgecolors="k", linewidths=1.2, c="none", label="Your clip")
        plt.legend(loc="best", frameon=True)
    plt.title("t-SNE on validation logits (+ your clip ★)")
    plt.tight_layout()
    plt.savefig(os.path.join(args.report_dir, "tsne_val_logits.png"), dpi=160)
    plt.close()

if __name__ == "__main__":
    main()
