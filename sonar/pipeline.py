"""End-to-end experiment: data -> classical baselines -> denoiser -> CNNs -> ablation/robustness/statistics -> detection mAP."""
import json
import os
import time
from pathlib import Path

import joblib
import numpy as np
import torch
from sklearn.ensemble import GradientBoostingClassifier, RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import balanced_accuracy_score
from sklearn.model_selection import GroupKFold, cross_val_score
from sklearn.neighbors import KNeighborsClassifier
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.svm import SVC

from . import data as D
from .config import NOISE_LEVELS, Config
from .detect import extract_detections, scan_classical, scan_cnn
from .features import feats_stack
from .metrics import (average_precision, cls_metrics, cluster_bootstrap_ci, enl, cnr, image_quality, mcnemar)
from .models import count_params
from .noise import filter_baseline, pseudo_clean_stack, speckle_np
from .training import denoise, predict_probs, train_cnn, train_denoiser

CLASSICAL = ["Logistic Regression", "KNN (k=5)", "SVM (RBF)", "Random Forest", "Gradient Boosting"]
DEEP = ["CNN (raw)", "CNN (noise-aug)", "Denoise + CNN"]
ALL_MODELS = CLASSICAL + DEEP


def make_classical():
    return {
        "Logistic Regression": make_pipeline(StandardScaler(), LogisticRegression(max_iter=3000, class_weight="balanced")),
        "KNN (k=5)": make_pipeline(StandardScaler(), KNeighborsClassifier(5)),
        "SVM (RBF)": make_pipeline(StandardScaler(), SVC(probability=True, class_weight="balanced", random_state=42)),
        "Random Forest": RandomForestClassifier(300, class_weight="balanced", random_state=42, n_jobs=-1),
        "Gradient Boosting": GradientBoostingClassifier(random_state=42),
    }


def _proba3(model, F):
    out = np.zeros((len(F), 3)); out[:, list(model.classes_)] = model.predict_proba(F); return out


def _sub(cb, a, b):
    return lambda f, m: cb(a + (b - a) * f, m)


def run_pipeline(cfg: Config, cb=lambda f, m: None):
    t_start = time.time()
    torch.set_num_threads(max(1, os.cpu_count() or 1))
    out = Path(cfg.out_dir); out.mkdir(parents=True, exist_ok=True)
    paths = D.list_images(cfg.data_root)
    if not paths:
        raise FileNotFoundError(f"No images found in '{cfg.data_root}'")
    if cfg.max_images and len(paths) > cfg.max_images:
        import random
        paths = sorted(random.Random(cfg.seed).sample(paths, cfg.max_images))
    smap = D.split_images(paths, cfg.seed)
    X, y, im, sp = D.build_patches(paths, smap, cfg.milco_id, cfg.bg_ratio, cfg.seed, _sub(cb, 0.0, 0.05))
    tr, va, te = sp == 0, sp == 1, sp == 2
    if min(tr.sum(), va.sum(), te.sum()) < 5 or len(np.unique(y[tr])) < 3:
        raise ValueError("Too few patches per split or missing classes; use more images.")
    Xtr, ytr, Xva, yva, Xte, yte, gte = X[tr], y[tr], X[va], y[va], X[te], y[te], im[te]
    np.savez_compressed(out / "patches.npz", X=X, y=y, img=im, split=sp)
    res = {"meta": {"config": cfg.to_dict(), "n_images": len(paths), "patches": {"train": int(tr.sum()), "val": int(va.sum()), "test": int(te.sum())},
                    "class_counts": {k: np.bincount(y[m], minlength=3).tolist() for k, m in [("train", tr), ("val", va), ("test", te)]},
                    "torch": torch.__version__, "device": "cpu", "n_cpu": os.cpu_count()},
           "image_paths": paths, "test_images": [p for p in paths if smap[p] == 2],
           "train_images": [p for p in paths if smap[p] == 0]}

    # ---------- 1. classical ML baselines ----------
    cb(0.05, "Classical features...")
    Ftr, Fva, Fte = feats_stack(Xtr), feats_stack(Xva), feats_stack(Xte)
    Ftr_a = np.concatenate([Ftr, feats_stack(Xtr[:, :, ::-1])]); ytr_a = np.concatenate([ytr, ytr]); gtr_a = np.concatenate([im[tr], im[tr]])
    classical, probs, cls_rows, val_scores = make_classical(), {}, {}, {}
    for k, (name, m) in enumerate(classical.items()):
        t0 = time.time()
        cv = float("nan") if cfg.fast else float(cross_val_score(m, Ftr_a, ytr_a, groups=gtr_a, cv=GroupKFold(3), scoring="balanced_accuracy").mean())
        m.fit(Ftr_a, ytr_a)
        val_scores[name] = balanced_accuracy_score(yva, m.predict(Fva)); probs[name] = _proba3(m, Fte)
        cls_rows[name] = {"cv_bal_acc": cv, "train_time_s": time.time() - t0}
        cb(0.05 + 0.10 * (k + 1) / 5, f"Classical: {name}")
    best_classical = max(val_scores, key=val_scores.get)
    joblib.dump({"models": classical, "best": best_classical}, out / "classical.joblib")

    # ---------- 2. deconvolutional denoiser ----------
    dn, dn_hist = train_denoiser(Xtr, Xva, cfg, _sub(cb, 0.15, 0.40))
    torch.save(dn.state_dict(), out / "denoiser.pt")

    # ---------- 3. CNNs ----------
    cnn_raw, h_raw = train_cnn(Xtr, ytr, Xva, yva, cfg, None, False, "CNN (raw)", _sub(cb, 0.40, 0.55))
    cnn_aug, h_aug = train_cnn(Xtr, ytr, Xva, yva, cfg, None, True, "CNN (noise-aug)", _sub(cb, 0.55, 0.70))
    cnn_dn, h_dn = train_cnn(Xtr, ytr, Xva, yva, cfg, dn, False, "Denoise + CNN", _sub(cb, 0.70, 0.85))
    for n, m in [("cnn_raw", cnn_raw), ("cnn_aug", cnn_aug), ("cnn_dn", cnn_dn)]:
        torch.save(m.state_dict(), out / f"{n}.pt")
    deep = {"CNN (raw)": (cnn_raw, None), "CNN (noise-aug)": (cnn_aug, None), "Denoise + CNN": (cnn_dn, dn)}
    for name, (m, d) in deep.items():
        probs[name] = predict_probs(m, Xte, d)

    # ---------- 4. evaluation ----------
    cb(0.85, "Evaluation, confidence intervals, McNemar...")
    table = {}
    for name in ALL_MODELS:
        pred = probs[name].argmax(1); r = cls_metrics(yte, pred)
        r["ci"] = cluster_bootstrap_ci(yte, pred, gte); r.update(cls_rows.get(name, {}))
        table[name] = r
    table["_best_classical"] = best_classical
    res["classification"] = table
    pairs = [(best_classical, "CNN (raw)"), ("CNN (raw)", "Denoise + CNN"), ("CNN (noise-aug)", "Denoise + CNN"),
             (best_classical, "Denoise + CNN")]
    res["mcnemar"] = [{"A": a, "B": b, **mcnemar(yte, probs[a].argmax(1), probs[b].argmax(1))} for a, b in pairs]

    # robustness: extra synthetic speckle on the *test* patches
    rob = {}
    for L in ["clean", *NOISE_LEVELS]:
        row = {}
        rng = np.random.default_rng(123)
        Xn = Xte if L == "clean" else speckle_np(Xte, float(L), rng)
        row[f"Classical ({best_classical})"] = probs[best_classical].argmax(1) if L == "clean" else _proba3(classical[best_classical], feats_stack(Xn)).argmax(1)
        for name, (m, d) in deep.items():
            row[name] = probs[name].argmax(1) if L == "clean" else predict_probs(m, Xte, d, looks=L, seed=123).argmax(1)
        rob[str(L)] = {n: {k: v for k, v in cls_metrics(yte, p).items() if k in ("acc", "bal_acc", "macro_f1", "milco_recall")} for n, p in row.items()}
    res["robustness"] = rob

    # denoiser quality
    ref_te = pseudo_clean_stack(Xte); dq = []
    for L in NOISE_LEVELS:
        rng = np.random.default_rng(7); noisy = speckle_np(ref_te, float(L), rng); den = denoise(dn, noisy)
        base = np.stack([filter_baseline(p, float(L)) for p in noisy])
        q0 = [image_quality(r, n) for r, n in zip(ref_te, noisy)]; q1 = [image_quality(r, d) for r, d in zip(ref_te, den)]
        q2 = [image_quality(r, b) for r, b in zip(ref_te, base)]
        mean = lambda q, k: float(np.mean([v[k] for v in q]))
        dq.append({"looks": L, **{f"{k}_before": mean(q0, k) for k in ("SNR", "PSNR", "SSIM")},
                   **{f"{k}_after": mean(q1, k) for k in ("SNR", "PSNR", "SSIM")},
                   "SNR_filter_after": mean(q2, "SNR"), "PSNR_filter_after": mean(q2, "PSNR"), "SSIM_filter_after": mean(q2, "SSIM"),
                   "SNR_improvement": mean(q1, "SNR") - mean(q0, "SNR"), "SNR_improvement_filter": mean(q2, "SNR") - mean(q0, "SNR")})
    real = {}
    den_real = denoise(dn, Xte); base_real = np.stack([filter_baseline(p, 4.0) for p in Xte])
    for lab, sel in [("ENL_background", y[te] == 0), ("CNR_objects", y[te] > 0)]:
        f = enl if lab.startswith("ENL") else cnr
        if sel.sum():
            real[lab] = {"raw": float(np.mean([f(p) for p in Xte[sel]])), "denoised": float(np.mean([f(p) for p in den_real[sel]])),
                         "lee_filter": float(np.mean([f(p) for p in base_real[sel]]))}
    res["denoise"] = {"by_noise": dq, "real_no_reference": real, "params": count_params(dn)}
    res["params"] = {"denoiser": count_params(dn), "cnn": count_params(cnn_raw)}
    res["curves"] = {"denoiser": dn_hist, "CNN (raw)": h_raw, "CNN (noise-aug)": h_aug, "Denoise + CNN": h_dn}
    np.savez_compressed(out / "preds.npz", y=yte, img=gte, **{n.replace(" ", "_"): p for n, p in probs.items()})

    # ---------- 5. detection mAP@0.5 ----------
    test_imgs = res["test_images"]
    if cfg.det_eval_images and len(test_imgs) > cfg.det_eval_images:
        import random
        test_imgs = random.Random(1).sample(test_imgs, cfg.det_eval_images)
    anchors = {}
    for c in (1, 2):
        wh = [(x2 - x1, y2 - y1) for p in res["train_images"] for c_, x1, y1, x2, y2 in D.yolo_to_px(D.read_boxes(p, cfg.milco_id), *reversed(D.load_gray(p).shape)) if c_ == c]
        anchors[c] = tuple(np.median(np.array(wh), 0).tolist()) if wh else (96.0, 96.0)
    dets = {"Classical sliding-window": {}, "CNN heatmap (raw)": {}, "CNN heatmap (Denoise + CNN)": {}}; gts = {}; tm = {k: 0.0 for k in dets}
    cm = classical[best_classical]
    for i, p in enumerate(test_imgs):
        arr = D.load_gray(p); H, W = arr.shape; gts[p] = D.yolo_to_px(D.read_boxes(p, cfg.milco_id), W, H)
        for k, fn in [("Classical sliding-window", lambda: scan_classical(arr, cm, 96, cfg.det_stride)),
                      ("CNN heatmap (raw)", lambda: scan_cnn(arr, cnn_raw, None, cfg.det_scales, cfg.det_stride)),
                      ("CNN heatmap (Denoise + CNN)", lambda: scan_cnn(arr, cnn_dn, dn, cfg.det_scales, cfg.det_stride))]:
            t0 = time.time(); dets[k][p] = extract_detections(fn(), anchors=anchors); tm[k] += time.time() - t0
        cb(0.85 + 0.14 * (i + 1) / len(test_imgs), f"Detection eval {i + 1}/{len(test_imgs)}")
    det = {}
    for k, d in dets.items():
        ap1, r1, n1 = average_precision(d, gts, 1); ap2, r2, n2 = average_precision(d, gts, 2)
        aps = [a for a in (ap1, ap2) if not np.isnan(a)]
        a1, a2 = average_precision(d, gts, 1, 0.3)[0], average_precision(d, gts, 2, 0.3)[0]; aps3 = [a for a in (a1, a2) if not np.isnan(a)]
        det[k] = {"AP50_MILCO": ap1, "AP50_NOMBO": ap2, "mAP50": float(np.mean(aps)) if aps else float("nan"), "mAP30": float(np.mean(aps3)) if aps3 else float("nan"),
                  "recall_MILCO": r1, "n_gt_MILCO": n1, "n_gt_NOMBO": n2, "sec_per_image": tm[k] / max(1, len(test_imgs))}
    res["detection"] = {"n_images": len(test_imgs), "methods": det, "scales": list(cfg.det_scales), "stride": cfg.det_stride, "anchors": {C: list(v) for C, v in zip(("MILCO", "NOMBO"), (anchors[1], anchors[2]))}}
    res["meta"]["total_time_s"] = time.time() - t_start

    def clean(o):
        if isinstance(o, dict): return {str(k): clean(v) for k, v in o.items()}
        if isinstance(o, (list, tuple)): return [clean(v) for v in o]
        if isinstance(o, (np.floating, float)): return None if np.isnan(o) else float(o)
        if isinstance(o, np.integer): return int(o)
        return o
    (out / "results.json").write_text(json.dumps(clean(res), indent=1))
    cb(1.0, "Done")
    return res
