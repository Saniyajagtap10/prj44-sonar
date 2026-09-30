"""Sliding-window detection. CNN heat-maps over several window scales -> per-class NMS -> boxes."""
import numpy as np
import torch
import torch.nn.functional as F

from .config import SZ
from .features import feats
from .metrics import iou_matrix


@torch.no_grad()
def scan_cnn(img, cnn, denoiser=None, scales=(64, 96, 128), stride=48, bs=512):
    """img: (H,W) float in [0,1]. Each scale s: downsample image by s/SZ (matches training patches), optionally
    denoise the whole down-sampled image (fully convolutional), unfold SZ x SZ windows, classify.
    Returns list of dicts {scale, probs (ny,nx,3), cx (nx), cy (ny)} (centres in original pixel coords)."""
    H, W = img.shape; t = torch.from_numpy(np.ascontiguousarray(img))[None, None]; res = []
    for s in scales:
        f = s / SZ; Hs, Ws = int(round(H / f)), int(round(W / f))
        if Hs < SZ or Ws < SZ:
            continue
        ts = F.interpolate(t, size=(Hs, Ws), mode="area")
        if denoiser is not None:
            ts = denoiser.eval()(ts).clamp(0, 1)
        st = max(1, int(round(stride / f)))
        win = ts.unfold(2, SZ, st).unfold(3, SZ, st); ny, nx = win.shape[2:4]
        w = win.reshape(-1, 1, SZ, SZ)
        pr = torch.cat([torch.softmax(cnn.eval()(w[i:i + bs]), 1) for i in range(0, len(w), bs)]).numpy()
        res.append({"scale": s, "probs": pr.reshape(ny, nx, 3), "cx": (np.arange(nx) * st + SZ / 2) * f,
                    "cy": (np.arange(ny) * st + SZ / 2) * f})
    return res


def scan_classical(img, model, scale=96, stride=48):
    H, W = img.shape; f = scale / SZ
    ts = F.interpolate(torch.from_numpy(np.ascontiguousarray(img))[None, None], size=(int(round(H / f)), int(round(W / f))), mode="area")
    st = max(1, int(round(stride / f)))
    win = ts.unfold(2, SZ, st).unfold(3, SZ, st); ny, nx = win.shape[2:4]
    w = win.reshape(-1, SZ, SZ).numpy()
    F_ = np.stack([feats(p) for p in w]); pr = np.zeros((len(w), 3))
    pr[:, list(model.classes_)] = model.predict_proba(F_)
    return [{"scale": scale, "probs": pr.reshape(ny, nx, 3), "cx": (np.arange(nx) * st + SZ / 2) * f,
             "cy": (np.arange(ny) * st + SZ / 2) * f}]


def nms(boxes, scores, thr=0.3):
    order = np.argsort(-scores); keep = []
    while len(order):
        i = order[0]; keep.append(i)
        if len(order) == 1:
            break
        order = order[1:][iou_matrix(boxes[i][None], boxes[order[1:]])[0] <= thr]
    return keep


def _refine_centres(p, cx, cy):
    """Soft-argmax over the 3x3 neighbourhood of each window -> sub-grid centre estimate."""
    ny, nx = p.shape; sx = cx[1] - cx[0] if nx > 1 else 0; sy = cy[1] - cy[0] if ny > 1 else 0
    pp = np.pad(p, 1); num_x = np.zeros_like(p); num_y = np.zeros_like(p); den = np.zeros_like(p)
    for dy in (-1, 0, 1):
        for dx in (-1, 0, 1):
            w = pp[1 + dy:1 + dy + ny, 1 + dx:1 + dx + nx]; den += w; num_x += w * dx * sx; num_y += w * dy * sy
    return cx[None, :] + num_x / (den + 1e-8), cy[:, None] + num_y / (den + 1e-8)


def extract_detections(scan, min_score=0.05, nms_thr=0.3, top=300, anchors=None):
    """-> [(cls, score, x1, y1, x2, y2)] for cls in {1 MILCO, 2 NOMBO}.
    anchors {cls: (w, h)} (median training-box size, px): box = anchor * scale/96 centred on the refined window
    centre. Without anchors a square window box is used."""
    dets = []
    for c in (1, 2):
        B, S = [], []
        for r in scan:
            p = r["probs"][..., c]; yy, xx = np.where(p >= min_score)
            if len(yy) == 0:
                continue
            rx, ry = _refine_centres(p, r["cx"], r["cy"])
            bw, bh = ((anchors[c][0] * r["scale"] / 96, anchors[c][1] * r["scale"] / 96) if anchors else (r["scale"], r["scale"]))
            for y, x in zip(yy, xx):
                B.append([rx[y, x] - bw / 2, ry[y, x] - bh / 2, rx[y, x] + bw / 2, ry[y, x] + bh / 2]); S.append(p[y, x])
        if not B:
            continue
        B, S = np.array(B), np.array(S)
        if len(S) > top:
            k = np.argsort(-S)[:top]; B, S = B[k], S[k]
        dets += [(c, float(S[i]), *B[i].tolist()) for i in nms(B, S, nms_thr)]
    return dets


def heatmap(scan, cls=1, scale_idx=None):
    """Return (prob grid, cx, cy) of the middle scale for display."""
    r = scan[len(scan) // 2 if scale_idx is None else scale_idx]
    return r["probs"][..., cls], r["cx"], r["cy"]
