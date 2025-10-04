import os, argparse, json, joblib
import numpy as np
import matplotlib
matplotlib.use("Agg") 
import matplotlib.pyplot as plt

from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler, FunctionTransformer
from sklearn.svm import SVC
from sklearn.neighbors import KNeighborsClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import confusion_matrix, classification_report, accuracy_score
from xgboost import XGBClassifier
from collections import defaultdict

from task1_dataload import (
    CFG,
    build_X_dir_with_meta_unlabeled, 
    build_Xy_json_fixedchunks_with_meta,  # 測試用（扁平資料夾、無標籤）
)
import os
import numpy as np
import xgboost as xgb

def _detect_xgb_tree_method(force_gpu=None) -> str:
    if force_gpu is True:
        return "gpu_hist"
    if force_gpu is False:
        return "hist"
    # auto
    env = os.environ.get("XGB_USE_GPU", "auto").lower()
    if env in ("1", "true", "yes"):
        return "gpu_hist"
    if env == "auto" and os.environ.get("CUDA_VISIBLE_DEVICES", "") not in ("", "-1"):
        return "gpu_hist"
    return "hist"

from sklearn.base import BaseEstimator, ClassifierMixin

class XGBCompat(BaseEstimator, ClassifierMixin):
    """
    sklearn 相容包裝：
    - 自動 num_class / objective
    - 舊版 XGB 沒有 early_stopping_rounds/callbacks 也能優雅 fallback
    - 提供 get_params/set_params，便於 Pipeline/Joblib
    """
    def __init__(self,
                 early_stopping_rounds=50,
                 **params):
        self.early_stopping_rounds = int(early_stopping_rounds)
        self.params = dict(params)
        self.model = None
        self.classes_ = None
        self.best_iteration_ = None

    # ---- sklearn API ----
    def get_params(self, deep=True):
        # 讓 GridSearch/Joblib 能存取
        out = dict(self.params)
        out["early_stopping_rounds"] = self.early_stopping_rounds
        return out

    def set_params(self, **params):
        if "early_stopping_rounds" in params:
            self.early_stopping_rounds = int(params.pop("early_stopping_rounds"))
        # 其餘都當成 XGB 參數
        self.params.update(params)
        return self

    def fit(self, X, y, eval_set=None, verbose=False):
        import xgboost as xgb
        y = np.asarray(y)
        self.classes_ = np.unique(y)
        n_classes = int(self.classes_.shape[0])

        # 準備參數
        params = dict(self.params)
        params.pop("num_class", None)  # 清掉 None
        if n_classes <= 2:
            params.setdefault("objective", "binary:logistic")
            params.setdefault("eval_metric", "logloss")
        else:
            params.setdefault("objective", "multi:softprob")
            params.setdefault("eval_metric", "mlogloss")
            params["num_class"] = n_classes

        # 預設加速
        params.setdefault("tree_method", _detect_xgb_tree_method())
        params.setdefault("n_estimators", 600)
        params.setdefault("learning_rate", 0.2)
        params.setdefault("max_depth", 5)
        params.setdefault("subsample", 0.7)
        params.setdefault("colsample_bytree", 0.7)
        params.setdefault("n_jobs", os.cpu_count() or 8)

        self.model = xgb.XGBClassifier(**params)

        # 依版本支援度決定是否帶 early_stopping_rounds
        try:
            self.model.fit(
                X, y,
                eval_set=eval_set,
                early_stopping_rounds=self.early_stopping_rounds if eval_set else None,
                verbose=verbose,
            )
        except TypeError:
            # 舊版：拿掉 early stopping、或甚至拿掉 eval_set
            try:
                self.model.fit(X, y, eval_set=eval_set, verbose=verbose)
            except TypeError:
                self.model.fit(X, y)

        # best_iteration（有早停才會有）
        self.best_iteration_ = getattr(self.model, "best_iteration", None)
        return self

    def predict(self, X):
        return self.model.predict(X)

    def predict_proba(self, X):
        if hasattr(self.model, "predict_proba"):
            return self.model.predict_proba(X)
        # 退路
        s = self.model.decision_function(X)
        if s.ndim == 1:
            s = np.vstack([-s, s]).T
        s = s - s.max(axis=1, keepdims=True)
        e = np.exp(s)
        return e / e.sum(axis=1, keepdims=True)


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

# --- keep your other imports / XGBCompat / _detect_xgb_tree_method 定義不變 ---

def build_clf(name: str, xgb_es_rounds: int = 50, force_gpu=None) -> Pipeline:
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
        clf = LogisticRegression(max_iter=2000, multi_class="multinomial",
                                 solver="lbfgs", n_jobs=-1)
        return Pipeline([("scaler", StandardScaler()), ("clf", clf)])

    elif name == "xgb":
        tree_method = _detect_xgb_tree_method(force_gpu)
        clf = XGBCompat(
            early_stopping_rounds=xgb_es_rounds,
            tree_method=tree_method,
            n_estimators=600,
            learning_rate=0.20,
            max_depth=5,
            min_child_weight=2.0,
            subsample=0.7,
            colsample_bytree=0.7,
            max_bin=256,
            n_jobs=os.cpu_count() or 8,
            random_state=42,
            verbosity=0,
        )
        return Pipeline([("identity", FunctionTransformer()), ("clf", clf)])

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
    ap.add_argument("--xgb_es_rounds", type=int, default=50,
                    help=">0 to try early stopping for XGB (fallback to no-ES if unsupported).")
    ap.add_argument("--ckpt_path", type=str, default="checkpoints/model.pkl",
                    help="存模型的路徑（.pkl，sklearn pipeline 序列化）")
    args = ap.parse_args()

    cfg = CFG(segment_sec=args.segment_sec)

    # ---------- Build data ----------
    Xtr, ytr, classes, _ = build_Xy_json_fixedchunks_with_meta(args.train_json, cfg=cfg, cache_dir=args.cache_dir)
    Xva, yva, _, val_srcs = build_Xy_json_fixedchunks_with_meta(args.val_json, cfg=cfg, classes=classes, cache_dir=args.cache_dir)

    # # ---------- Build & train ----------
    pipe = build_clf(args.clf, xgb_es_rounds=args.xgb_es_rounds)
    n_classes = len(classes)
    clf = pipe.named_steps["clf"]

    if args.clf.lower() == "xgb":
        # 直接呼叫 wrapper.fit（它會處理早停/相容性）
        clf.fit(
            Xtr.astype(np.float32, copy=False),
            ytr,
            eval_set=[(Xva.astype(np.float32, copy=False), yva)],
            verbose=True,
        )
        model = pipe
    else:
        pipe.fit(Xtr, ytr)
        model = pipe

    # 不要再呼叫 clf.set_params(...) 來改 n_estimators，已早停就會用最佳迭代
    # 如果一定要知道最佳迭代，可以讀 clf.best_iteration_




    # --- 若是 XGB，用最佳迭代數收斂模型大小（有 early stopping 才會有 best_iteration） ---
    if args.clf.lower() == "xgb" and hasattr(clf, "best_iteration") and clf.best_iteration is not None:
        clf.set_params(n_estimators=clf.best_iteration + 1)

    # --- 存模型（整條 pipeline） ---
    ckpt_path = args.ckpt_path
    os.makedirs(os.path.dirname(ckpt_path), exist_ok=True)
    joblib.dump(model, ckpt_path)
    print(f"[CKPT] Saved model checkpoint to: {ckpt_path}")

    # ---------- Validation (chunk-level) ----------
    y_pred = model.predict(Xva)
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
    proba_va = predict_proba_safe(model, Xva, n_classes=len(classes))
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

    # --- 存 metadata（類別名單 + 特徵參數），方便推論/還原 ---
    meta = {
        "clf": args.clf,
        "classes": classes,
        "top_1_ac":f"{acc1_track:.4f}",
        "top_3_ac":f"{acc3_track:.4f}",
        "cfg": {
            "sr": cfg.sr, "n_fft": cfg.n_fft, "hop": cfg.hop,
            "n_mfcc": cfg.n_mfcc, "segment_sec": cfg.segment_sec
        }
    }
    meta_path = ckpt_path.replace(".pkl", "_meta.json")
    with open(meta_path, "w", encoding="utf-8") as f:
        json.dump(meta, f, ensure_ascii=False, indent=2)
    print(f"[CKPT] Saved metadata to: {meta_path}")

    # ---------- Test (optional, flat root, unlabeled) ----------
    if args.test_root and os.path.exists(args.test_root):
        print("\n=== TEST PREDICTION (flat .wav under test_root) ===")
        Xt, test_keys = build_X_dir_with_meta_unlabeled(args.test_root, cfg)
        proba_t = predict_proba_safe(model, Xt, n_classes=len(classes))

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
