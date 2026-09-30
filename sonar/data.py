"""Dataset IO: images, YOLO boxes, patch extraction, image-wise split."""
import random
from pathlib import Path

import numpy as np
from PIL import Image

from .config import SZ, WIN

IMG_EXT = {".jpg", ".jpeg", ".png"}


def list_images(root):
    r = Path(root)
    return sorted(str(p) for p in r.rglob("*") if p.suffix.lower() in IMG_EXT) if r.exists() else []


def read_boxes(p, milco_id=0):
    """YOLO txt -> [(label, xc, yc, w, h)] normalised; label 1 = MILCO, 2 = NOMBO."""
    t = Path(p).with_suffix(".txt")
    out = []
    if t.exists():
        for ln in t.read_text().splitlines():
            v = ln.split()
            if len(v) >= 5:
                out.append((1 if int(float(v[0])) == milco_id else 2, *map(float, v[1:5])))
    return out


def load_gray(p):
    return np.asarray(Image.open(p).convert("L"), dtype=np.float32) / 255.0


def resize_area(c, sz=SZ):
    if c.shape[0] == sz and c.shape[1] == sz:
        return np.ascontiguousarray(c, dtype=np.float32)
    return np.asarray(Image.fromarray(np.ascontiguousarray(c, dtype=np.float32)).resize((sz, sz), Image.BOX), dtype=np.float32)


def crop_patch(arr, cx, cy, win=WIN, sz=SZ):
    l, t = int(round(cx - win / 2)), int(round(cy - win / 2))
    H, W = arr.shape
    pad = max(0, -l, -t, l + win - W, t + win - H)
    if pad:
        arr = np.pad(arr, pad, mode="edge"); l += pad; t += pad
    return resize_area(arr[t:t + win, l:l + win], sz)


def yolo_to_px(boxes, W, H):
    """-> [(label, x1, y1, x2, y2)] in pixels."""
    return [(c, (xc - w / 2) * W, (yc - h / 2) * H, (xc + w / 2) * W, (yc + h / 2) * H) for c, xc, yc, w, h in boxes]


def split_images(paths, seed=42, frac=(0.6, 0.15, 0.25)):
    """Image-wise split -> {path: 0 train / 1 val / 2 test}. No patch of a test image ever reaches training."""
    idx = list(range(len(paths))); random.Random(seed).shuffle(idx)
    n_tr, n_va = int(frac[0] * len(idx)), int(frac[1] * len(idx))
    m = {}
    for k, i in enumerate(idx):
        m[paths[i]] = 0 if k < n_tr else (1 if k < n_tr + n_va else 2)
    return m


def build_patches(paths, split_map, milco_id=0, bg_ratio=1.0, seed=42, cb=None):
    """Object patches (label 1/2) centred on annotated boxes + random background patches (label 0)."""
    rnd = random.Random(seed)
    X, y, im_id, sp = [], [], [], []
    boxes_cache = {}
    n_obj = 0
    for i, p in enumerate(paths):
        bx = read_boxes(p, milco_id); boxes_cache[p] = bx
        n_obj += len(bx)
    n_bg = max(30, int(bg_ratio * n_obj))
    bg_count = {}
    for _ in range(n_bg):
        i = rnd.randrange(len(paths)); bg_count[i] = bg_count.get(i, 0) + 1
    for i, p in enumerate(paths):
        bx = boxes_cache[p]
        if not bx and i not in bg_count:
            continue
        arr = load_gray(p); H, W = arr.shape
        for c, xc, yc, _, _ in bx:
            X.append(crop_patch(arr, xc * W, yc * H)); y.append(c); im_id.append(i); sp.append(split_map[p])
        got, tries = 0, 0
        while got < bg_count.get(i, 0) and tries < 40:
            tries += 1
            if W <= WIN or H <= WIN:
                break
            cx, cy = rnd.randint(WIN // 2, W - WIN // 2), rnd.randint(WIN // 2, H - WIN // 2)
            if any(abs(xc * W - cx) < 0.75 * WIN and abs(yc * H - cy) < 0.75 * WIN for _, xc, yc, _, _ in bx):
                continue
            X.append(crop_patch(arr, cx, cy)); y.append(0); im_id.append(i); sp.append(split_map[p]); got += 1
        if cb and i % 25 == 0:
            cb(i / len(paths), f"Extracting patches ({i}/{len(paths)} images)")
    return (np.stack(X).astype(np.float32), np.array(y), np.array(im_id), np.array(sp))
