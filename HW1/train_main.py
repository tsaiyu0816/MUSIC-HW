import os, argparse
import numpy as np
import matplotlib.pyplot as plt

from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler, FunctionTransformer
from sklearn.svm import SVC
from sklearn.neighbors import KNeighborsClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import train_test_split
from sklearn.metrics import confusion_matrix, classification_report, accuracy_score
from xgboost import XGBClassifier
from xgboost.callback import EarlyStopping
from collections import defaultdict
from dataload_audio import CFG, build_Xy_json, build_Xy_json_with_meta, discover_classes

def aggregate_proba_by_key(proba: np.ndarray, y_true: np.ndarray, keys: np.ndarray):
    """
    將 chunk-level 機率依歌曲 key 聚合成 track-level 機率（用「平均機率」）
    回傳:
      y_true_track: (T,) 每首歌的真實標籤
      proba_track : (T,C) 每首歌平均後的機率分佈
      uniq_keys   : (T,) 去重後的歌曲 keys（可用來對應）
    """
    from collections import defaultdict
    idxs = defaultdict(list)
    for i, k in enumerate(keys):
        idxs[k].append(i)

    uniq_keys, y_true_track, proba_track = [], [], []
    for k, id_list in idxs.items():
        uniq_keys.append(k)
        # 這首歌的真實標籤（理論上相同；保險取眾數）
        y_true_track.append(np.bincount(y_true[id_list]).argmax())
        # 平均機率
        proba_track.append(proba[id_list].mean(axis=0))

    return np.array(y_true_track), np.vstack(proba_track), np.array(uniq_keys)


def topk_accuracy_from_proba(y_true: np.ndarray, proba: np.ndarray, k: int = 3) -> float:
    k = min(k, proba.shape[1])
    topk = np.argsort(proba, axis=1)[:, -k:]
    return float(np.mean([y_true[i] in topk[i] for i in range(len(y_true))]))


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
        clf = LogisticRegression(max_iter=2000, n_jobs=-1, multi_class="multinomial")
        return Pipeline([("scaler", StandardScaler()), ("clf", clf)])
    elif name == "xgb":
        xgb = XGBClassifier(
            n_estimators=600,
            max_depth=6,
            learning_rate=0.05,
            subsample=0.8,
            colsample_bytree=0.8,
            eval_metric=["mlogloss", "merror"],  # 顯示 loss + 錯誤率
            tree_method="hist",                  # 有 GPU 換 "gpu_hist"
            n_jobs=-1,
            random_state=42,
        )
        return Pipeline([("identity", FunctionTransformer()), ("clf", xgb)])
    else:
        raise ValueError("Unsupported classifier. Choose: svm | knn | rf | lr | xgb")

def topk_accuracy(y_true, proba, k=3):
    k = min(k, proba.shape[1])
    topk = np.argsort(proba, axis=1)[:, -k:]
    return float(np.mean([y_true[i] in topk[i] for i in range(len(y_true))]))

def plot_and_save_cm(cm, classes, out_png):
    fig, ax = plt.subplots(figsize=(6,6))
    im = ax.imshow(cm, interpolation='nearest')
    ax.figure.colorbar(im, ax=ax)
    ax.set(xticks=np.arange(cm.shape[1]), yticks=np.arange(cm.shape[0]),
           xticklabels=classes, yticklabels=classes, ylabel='True', xlabel='Predicted')
    plt.setp(ax.get_xticklabels(), rotation=45, ha="right", rotation_mode="anchor")
    ax.set_title("Confusion Matrix")
    plt.tight_layout()
    os.makedirs(os.path.dirname(out_png), exist_ok=True)
    plt.savefig(out_png, dpi=160)
    plt.close(fig)

def main():
    ap = argparse.ArgumentParser()
    # Choose one input模式：JSON清單 或 資料夾
    ap.add_argument("--train_json", type=str, default="./hw1/artist20/train.json", help="JSON list of train file paths")
    ap.add_argument("--val_json",   type=str, default="./hw1/artist20/val.json", help="JSON list of val file paths")
    ap.add_argument("--test_root",  type=str, default="./hw1/artist20/test", help="Folder mode: data/test")

    # Training opts
    ap.add_argument("--clf", type=str, default="svm", help="svm | knn | rf | lr | xgb")
    ap.add_argument("--segment_sec", type=int, default=30)
    ap.add_argument("--cache_dir", type=str, default="cache")
    ap.add_argument("--report_dir", type=str, default="reports")
    args = ap.parse_args()

    cfg = CFG(segment_sec=args.segment_sec)

    # ---------- 建立資料（優先使用 JSON 模式） ----------
    Xtr, ytr, classes = build_Xy_json(args.train_json, cfg=cfg, cache_dir=args.cache_dir)
    Xva, yva, _, val_srcs = build_Xy_json_with_meta(args.val_json, cfg=cfg, classes=classes, cache_dir=args.cache_dir)
    test_root = args.test_root
    # if os.path.exists(test_root) and any(os.scandir(test_root)):
    #     Xt, yt, _, test_srcs = build_Xy_dir_with_meta(test_root, cfg=cfg, classes=classes, cache_dir=args.cache_dir)

    # ---------- 建模 ----------
    pipe = build_clf(args.clf)

    if args.clf.lower() == "xgb":
        # 直接用 XGBClassifier.fit（新版 API）
        es_rounds = getattr(args, "xgb_es_rounds", 50)
        pipe.fit(
            Xtr, ytr,
            clf__eval_set=[(Xva, yva)],
            clf__verbose= True
        )
    else:
        pipe.fit(Xtr, ytr)


    # ---------- Validation (Chunk-level) ----------
    y_pred = pipe.predict(Xva)
    acc1 = accuracy_score(yva, y_pred)
    print("\n=== VALIDATION (Chunk-level) ===")
    print(f"Top-1 Accuracy : {acc1:.4f}")
    print("\nClassification Report (chunk):")
    print(classification_report(yva, y_pred, target_names=classes, digits=4, zero_division=0))
    cm = confusion_matrix(yva, y_pred)
    cm_png = os.path.join(args.report_dir, "confusion_matrix_val_chunk.png")
    plot_and_save_cm(cm, classes, cm_png)
    print(f"Saved confusion matrix (chunk): {cm_png}")

    # Top-3 (chunk)
    if hasattr(pipe.named_steps["clf"], "predict_proba"):
        proba_va = pipe.predict_proba(Xva)
        acc3_chunk = topk_accuracy_from_proba(yva, proba_va, k=3)
        print(f"Top-3 Accuracy (chunk): {acc3_chunk:.4f}")
    else:
        proba_va = None
        print("[WARN] 此模型沒有 predict_proba，無法計算 Top-3 / 做 track-level 平均機率。")

    # ---------- Validation (Track-level by mean_proba) ----------
    if proba_va is not None:
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

    # ---------- Test (chunk-level) ----------
    if 'Xt' in locals() and Xt is not None:
        pt = pipe.predict(Xt)
        acc1_t = accuracy_score(yt, pt)
        print("\n=== TEST (Chunk-level) ===")
        print(f"Top-1 Accuracy : {acc1_t:.4f}")
        print("\nClassification Report (chunk, TEST):")
        print(classification_report(yt, pt, target_names=classes, digits=4, zero_division=0))
        cm_t = confusion_matrix(yt, pt)
        cmtest_png = os.path.join(args.report_dir, "confusion_matrix_test_chunk.png")
        plot_and_save_cm(cm_t, classes, cmtest_png)
        print(f"Saved confusion matrix (chunk): {cmtest_png}")

        # Top-3 (chunk)
        if hasattr(pipe.named_steps["clf"], "predict_proba"):
            proba_t = pipe.predict_proba(Xt)
            acc3_t = topk_accuracy_from_proba(yt, proba_t, k=3)
            print(f"Top-3 Accuracy (chunk): {acc3_t:.4f}")

            # ---------- Test (Track-level by mean_proba) ----------
            y_true_track_t, proba_track_t, uniq_keys_t = aggregate_proba_by_key(proba_t, yt, np.array(test_srcs))
            y_pred_track_t = proba_track_t.argmax(axis=1)
            acc1_track_t = accuracy_score(y_true_track_t, y_pred_track_t)
            acc3_track_t = topk_accuracy_from_proba(y_true_track_t, proba_track_t, k=3)
            print("\n=== TEST (Track-level, mean_proba) ===")
            print(f"Track Top-1 Acc : {acc1_track_t:.4f}")
            print(f"Track Top-3 Acc : {acc3_track_t:.4f}")
            print("\nClassification Report (track, TEST):")
            print(classification_report(y_true_track_t, y_pred_track_t, target_names=classes, digits=4, zero_division=0))
            cm_tt = confusion_matrix(y_true_track_t, y_pred_track_t)
            cmtest_track_png = os.path.join(args.report_dir, "confusion_matrix_test_track.png")
            plot_and_save_cm(cm_tt, classes, cmtest_track_png)
            print(f"Saved confusion matrix (track): {cmtest_track_png}")
        else:
            print("[WARN] 此模型沒有 predict_proba，無法做 Top-3 / track-level 平均機率。")


if __name__ == "__main__":
    main()
