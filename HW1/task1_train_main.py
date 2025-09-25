# -*- coding: utf-8 -*-
# file: train_main.py
import os, argparse, json
import numpy as np
import matplotlib.pyplot as plt

from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler, FunctionTransformer
from sklearn.svm import SVC
from sklearn.neighbors import KNeighborsClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import confusion_matrix, classification_report, accuracy_score

from collections import defaultdict

from task1_dataload import (
    CFG,
    build_Xy_json,
    build_Xy_json_with_meta,
    build_X_dir_with_meta_unlabeled,   # 測試用（扁平資料夾、無標籤）
)

# ------------------ Utils ------------------
def topk_accuracy_from_proba(y_true: np.ndarray, proba: np.ndarray, k: int = 3) -> float:
    k = min(k, proba.shape[1])
    topk = np.argsort(proba, axis=1)[:, -k:]
    return float(np.mean([y_true[i] in topk[i] for i in range(len(y_true))]))

def aggregate_proba_by_key(proba: np.ndarray, y_true: np.ndarray, keys: np.ndarray):
    """將 chunk 機率依歌曲 key 聚合（平均機率）。回傳 track 真值、平均後機率、唯一 keys。"""
    idxs = defaultdict(list)
    for i, k in enumerate(keys):
        idxs[k].append(i)
    uniq_keys, y_true_track, proba_track = [], [], []
    for k, id_list in idxs.items():
        uniq_keys.append(k)
        y_true_track.append(np.bincount(y_true[id_list]).argmax())
        proba_track.append(proba[id_list].mean(axis=0))
    return np.array(y_true_track), np.vstack(proba_track), np.array(uniq_keys)

def plot_and_save_cm(cm, classes, out_png):
    fig, ax = plt.subplots(figsize=(6, 6))
    im = ax.imshow(cm, interpolation="nearest")
    ax.figure.colorbar(im, ax=ax)
    ax.set(
        xticks=np.arange(cm.shape[1]),
        yticks=np.arange(cm.shape[0]),
        xticklabels=classes,
        yticklabels=classes,
        ylabel="True",
        xlabel="Predicted",
    )
    plt.setp(ax.get_xticklabels(), rotation=45, ha="right", rotation_mode="anchor")
    ax.set_title("Confusion Matrix")
    plt.tight_layout()
    os.makedirs(os.path.dirname(out_png), exist_ok=True)
    plt.savefig(out_png, dpi=160)
    plt.close(fig)

def predict_proba_safe(pipe: Pipeline, X: np.ndarray, n_classes: int):
    """優先用 predict_proba；沒有則用 decision_function+softmax；最後退路 one-hot。"""
    clf = pipe.named_steps["clf"]
    if hasattr(clf, "predict_proba"):
        return pipe.predict_proba(X)
    elif hasattr(clf, "decision_function"):
        s = pipe.decision_function(X)
        if s.ndim == 1:  # 二分類 → 兩類
            s = np.vstack([-s, s]).T
        s = s - np.max(s, axis=1, keepdims=True)
        e = np.exp(s)
        return (e / np.sum(e, axis=1, keepdims=True)).astype(np.float32)
    else:
        pred = pipe.predict(X)
        proba = np.zeros((len(pred), n_classes), dtype=np.float32)
        proba[np.arange(len(pred)), pred] = 1.0
        return proba

# ------------------ Models ------------------
def build_clf(name: str) -> Pipeline:
    name = name.lower()
    if name == "svm":
        clf = SVC(kernel="rbf", C=10, gamma="scale", probability=True, random_state=42)
        return Pipeline([("scaler", StandardScaler()), ("clf", clf)])
    elif name == "knn":
        clf = KNeighborsClassifier(n_neighbors=5, weights="distance")
        return Pipeline([("scaler", StandardScaler()), ("clf", clf)])
    elif name == "rf":
        clf = RandomForestClassifier(n_estimators=400, n_jobs=-1, random_state=42)
        return Pipeline([("scaler", StandardScaler()), ("clf", clf)])
    elif name == "lr":
        # multinomial + lbfgs 較穩；n_jobs 無效但保留兼容
        clf = LogisticRegression(max_iter=2000, multi_class="multinomial", solver="lbfgs", n_jobs=-1)
        return Pipeline([("scaler", StandardScaler()), ("clf", clf)])
    elif name == "xgb":
        # 用 FunctionTransformer 讓 pipeline 形式一致
        xgb = __import__("xgboost").XGBClassifier(
            n_estimators=800,
            max_depth=6,
            learning_rate=0.05,
            subsample=0.8,
            colsample_bytree=0.8,
            eval_metric=["mlogloss", "merror"],
            tree_method="hist",  # 有 GPU 可改 'gpu_hist'
            n_jobs=-1,
            random_state=42,
        )
        return Pipeline([("identity", FunctionTransformer()), ("clf", xgb)])
    else:
        raise ValueError("Unsupported classifier. Choose: svm | knn | rf | lr | xgb")

# ------------------ Main ------------------
def main():
    ap = argparse.ArgumentParser()
    # Data
    ap.add_argument("--train_json", type=str, default="./hw1/artist20/train.json",
                    help="JSON list of train file paths")
    ap.add_argument("--val_json",   type=str, default="./hw1/artist20/val.json",
                    help="JSON list of val file paths")
    ap.add_argument("--test_root",  type=str, default="./hw1/artist20/test",
                    help="Folder that contains flat .wav files (no labels)")

    # Training
    ap.add_argument("--clf", type=str, default="svm", help="svm | knn | rf | lr | xgb")
    ap.add_argument("--segment_sec", type=int, default=30)
    ap.add_argument("--cache_dir", type=str, default="cache")
    ap.add_argument("--report_dir", type=str, default="reports")
    ap.add_argument("--xgb_es_rounds", type=int, default=0,
                    help=">0 to try early stopping for XGB (fallback to no-ES if unsupported).")
    args = ap.parse_args()

    cfg = CFG(segment_sec=args.segment_sec)

    # ---------- Build data ----------
    Xtr, ytr, classes = build_Xy_json(args.train_json, cfg=cfg, cache_dir=args.cache_dir)
    Xva, yva, _, val_srcs = build_Xy_json_with_meta(args.val_json, cfg=cfg, classes=classes, cache_dir=args.cache_dir)

    # ---------- Build & train ----------
    pipe = build_clf(args.clf)

    if args.clf.lower() == "xgb":
        # 嘗試 early stopping；不支援時自動回退
        fit_kwargs = {"clf__eval_set": [(Xva, yva)], "clf__verbose": True}
        if args.xgb_es_rounds and args.xgb_es_rounds > 0:
            fit_kwargs["clf__early_stopping_rounds"] = int(args.xgb_es_rounds)
        try:
            pipe.fit(Xtr, ytr, **fit_kwargs)
        except TypeError:
            # 版本不接受 early_stopping_rounds → 移除後再訓練
            fit_kwargs.pop("clf__early_stopping_rounds", None)
            pipe.fit(Xtr, ytr, **fit_kwargs)
    else:
        pipe.fit(Xtr, ytr)

    # ---------- Validation (chunk-level) ----------
    y_pred = pipe.predict(Xva)
    acc1 = accuracy_score(yva, y_pred)
    print("\n=== VALIDATION (Chunk-level) ===")
    print(f"Top-1 Accuracy : {acc1:.4f}")
    print("\nClassification Report (chunk):")
    print(classification_report(yva, y_pred, target_names=classes, digits=4, zero_division=0))
    cm = confusion_matrix(yva, y_pred)
    os.makedirs(args.report_dir, exist_ok=True)
    cm_png = os.path.join(args.report_dir, "confusion_matrix_val_chunk.png")
    plot_and_save_cm(cm, classes, cm_png)
    print(f"Saved confusion matrix (chunk): {cm_png}")

    # Top-3 (chunk)
    proba_va = predict_proba_safe(pipe, Xva, n_classes=len(classes))
    acc3_chunk = topk_accuracy_from_proba(yva, proba_va, k=3)
    print(f"Top-3 Accuracy (chunk): {acc3_chunk:.4f}")

    # ---------- Validation (track-level, mean_proba) ----------
    y_true_track, proba_track, uniq_keys = aggregate_proba_by_key(proba_va, yva, np.array(val_srcs))
    y_pred_track = proba_track.argmax(axis=1)
    acc1_track = accuracy_score(y_true_track, y_pred_track)
    acc3_track = topk_accuracy_from_proba(y_true_track, proba_track, k=3)
    print("\n=== VALIDATION (Track-level, mean_proba) ===")
    print(f"Track Top-1 Acc : {acc1_track:.4f}")
    print(f"Track Top-3 Acc : {acc3_track:.4f}")
    print("\nClassification Report (track):")
    print(classification_report(y_true_track, y_pred_track, target_names=classes, digits=4, zero_division=0))
    cm_t = confusion_matrix(y_true_track, y_pred_track)
    cmtrack_png = os.path.join(args.report_dir, "confusion_matrix_val_track.png")
    plot_and_save_cm(cm_t, classes, cmtrack_png)
    print(f"Saved confusion matrix (track): {cmtrack_png}")

    # ---------- Test (optional, flat root, unlabeled) ----------
    if args.test_root and os.path.exists(args.test_root):
        print("\n=== TEST PREDICTION (flat .wav under test_root) ===")
        Xt, test_keys = build_X_dir_with_meta_unlabeled(args.test_root, cfg)
        proba_t = predict_proba_safe(pipe, Xt, n_classes=len(classes))

        # 以檔名（不含副檔名）聚合平均機率 → 取 Top-3
        buckets = defaultdict(list)
        for i, k in enumerate(test_keys):
            buckets[k].append(i)

        results = {}
        for k, idxs in buckets.items():
            p = proba_t[idxs].mean(axis=0)
            top3_idx = np.argsort(p)[-3:][::-1]
            results[k] = [classes[i] for i in top3_idx]

        os.makedirs(args.report_dir, exist_ok=True)
        out_json = os.path.join(args.report_dir, "test_top3.json")
        with open(out_json, "w", encoding="utf-8") as f:
            json.dump(results, f, ensure_ascii=False, indent=2)
        print(f"Saved predictions: {out_json}")
    else:
        print("\n[INFO] No test_root provided or folder missing; skip test prediction.")

if __name__ == "__main__":
    main()
