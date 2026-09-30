import numpy as np
from scipy.ndimage import uniform_filter
from scipy.stats import binomtest
from sklearn.metrics import (accuracy_score, balanced_accuracy_score, f1_score,
                             precision_recall_fscore_support, recall_score)

EPS = 1e-12


def snr_db(ref, x):
    return float(10 * np.log10((ref ** 2).sum() / (((x - ref) ** 2).sum() + EPS) + EPS))


def psnr_db(ref, x, data_range=1.0):
    mse = ((x - ref) ** 2).mean()
    return float(10 * np.log10(data_range ** 2 / (mse + EPS)))


def ssim(ref, x, data_range=1.0, win=7):
    c1, c2 = (0.01 * data_range) ** 2, (0.03 * data_range) ** 2
    mu1, mu2 = uniform_filter(ref, win), uniform_filter(x, win)
    s11 = uniform_filter(ref * ref, win) - mu1 ** 2
    s22 = uniform_filter(x * x, win) - mu2 ** 2
    s12 = uniform_filter(ref * x, win) - mu1 * mu2
    m = ((2 * mu1 * mu2 + c1) * (2 * s12 + c2)) / ((mu1 ** 2 + mu2 ** 2 + c1) * (s11 + s22 + c2))
    return float(m.mean())


def image_quality(ref, x):
    return {"SNR": snr_db(ref, x), "PSNR": psnr_db(ref, x), "SSIM": ssim(ref, x)}


def enl(p):
    """Equivalent number of looks on a (nominally homogeneous) patch = mean^2 / variance. Higher = smoother speckle."""
    c = p[8:-8, 8:-8]
    return float(c.mean() ** 2 / (c.var() + EPS))


def cnr(p):
    """Contrast-to-noise ratio: centre (object) vs surrounding ring (background)."""
    h, w = p.shape
    m = np.zeros_like(p, bool); m[h // 2 - 8:h // 2 + 8, w // 2 - 8:w // 2 + 8] = True
    ring = ~np.zeros_like(p, bool); ring[h // 2 - 12:h // 2 + 12, w // 2 - 12:w // 2 + 12] = False
    return float(abs(p[m].mean() - p[ring].mean()) / (p[ring].std() + EPS))


# ---------------- classification ----------------
def cls_metrics(y, pred):
    p, r, f, s = precision_recall_fscore_support(y, pred, labels=[0, 1, 2], zero_division=0)
    return {"acc": accuracy_score(y, pred), "bal_acc": balanced_accuracy_score(y, pred),
            "macro_f1": f1_score(y, pred, average="macro", labels=[0, 1, 2], zero_division=0),
            "milco_recall": float(r[1]), "milco_precision": float(p[1]),
            "per_class": {"precision": p.tolist(), "recall": r.tolist(), "f1": f.tolist(), "support": s.tolist()}}


def _scalar(fn_name, y, pred):
    if fn_name == "bal_acc":
        return balanced_accuracy_score(y, pred)
    if fn_name == "macro_f1":
        return f1_score(y, pred, average="macro", labels=[0, 1, 2], zero_division=0)
    if fn_name == "milco_recall":
        return recall_score(y, pred, labels=[1], average="macro", zero_division=0)
    return accuracy_score(y, pred)


def cluster_bootstrap_ci(y, pred, groups, names=("acc", "bal_acc", "macro_f1", "milco_recall"), n=400, seed=0):
    """95% CI by resampling *images* (patches from the same image are correlated)."""
    rng = np.random.default_rng(seed)
    ug = np.unique(groups)
    idx = {g: np.where(groups == g)[0] for g in ug}
    samples = {k: [] for k in names}
    for _ in range(n):
        pick = np.concatenate([idx[g] for g in rng.choice(ug, len(ug))])
        for k in names:
            samples[k].append(_scalar(k, y[pick], pred[pick]))
    return {k: (float(np.percentile(v, 2.5)), float(np.percentile(v, 97.5))) for k, v in samples.items()}


def mcnemar(y, pa, pb):
    """Exact McNemar test on paired correctness. b = A right & B wrong, c = A wrong & B right."""
    ca, cb = pa == y, pb == y
    b, c = int((ca & ~cb).sum()), int((~ca & cb).sum())
    p = 1.0 if b + c == 0 else float(binomtest(b, b + c, 0.5).pvalue)
    return {"A_only_correct": b, "B_only_correct": c, "p_value": p}


# ---------------- detection ----------------
def iou_matrix(a, b):
    a, b = np.asarray(a, float).reshape(-1, 4), np.asarray(b, float).reshape(-1, 4)
    x1 = np.maximum(a[:, None, 0], b[None, :, 0]); y1 = np.maximum(a[:, None, 1], b[None, :, 1])
    x2 = np.minimum(a[:, None, 2], b[None, :, 2]); y2 = np.minimum(a[:, None, 3], b[None, :, 3])
    inter = np.clip(x2 - x1, 0, None) * np.clip(y2 - y1, 0, None)
    aa = (a[:, 2] - a[:, 0]) * (a[:, 3] - a[:, 1]); ab = (b[:, 2] - b[:, 0]) * (b[:, 3] - b[:, 1])
    return inter / (aa[:, None] + ab[None, :] - inter + EPS)


def average_precision(dets, gts, cls, iou_thr=0.5):
    """VOC all-point AP. dets: {img: [(cls, score, x1,y1,x2,y2)]}, gts: {img: [(cls, x1,y1,x2,y2)]}."""
    n_gt = sum(1 for v in gts.values() for g in v if g[0] == cls)
    if n_gt == 0:
        return float("nan"), 0, 0
    recs = [(s, im, np.array(b)) for im, v in dets.items() for c, s, *b in v if c == cls]
    recs.sort(key=lambda r: -r[0])
    used = {im: np.zeros(sum(1 for g in gts.get(im, []) if g[0] == cls), bool) for im in gts}
    tp = np.zeros(len(recs)); fp = np.zeros(len(recs))
    for i, (s, im, b) in enumerate(recs):
        g = np.array([x[1:] for x in gts.get(im, []) if x[0] == cls])
        if len(g) == 0:
            fp[i] = 1; continue
        ious = iou_matrix(b[None], g)[0]; j = int(ious.argmax())
        if ious[j] >= iou_thr and not used[im][j]:
            tp[i] = 1; used[im][j] = True
        else:
            fp[i] = 1
    tp, fp = np.cumsum(tp), np.cumsum(fp)
    rec = tp / n_gt; prec = tp / np.maximum(tp + fp, EPS)
    mrec = np.concatenate([[0], rec, [1]]); mpre = np.concatenate([[0], prec, [0]])
    for i in range(len(mpre) - 2, -1, -1):
        mpre[i] = max(mpre[i], mpre[i + 1])
    k = np.where(mrec[1:] != mrec[:-1])[0]
    ap = float(((mrec[k + 1] - mrec[k]) * mpre[k + 1]).sum())
    return ap, float(rec[-1]) if len(rec) else 0.0, int(n_gt)
