"""Synthetic side-scan-like sonar generator, ONLY for smoke-testing the code when the real dataset is not present.
It is NOT a substitute for the real Gavia dataset and results on it mean nothing scientifically."""
from pathlib import Path

import numpy as np
from PIL import Image


def _ellipse(yy, xx, cy, cx, ry, rx, ang=0.0):
    c, s = np.cos(ang), np.sin(ang); u = (xx - cx) * c + (yy - cy) * s; v = -(xx - cx) * s + (yy - cy) * c
    return (u / rx) ** 2 + (v / ry) ** 2 <= 1


def make_synthetic(out_dir, n=80, size=384, seed=0):
    out = Path(out_dir); out.mkdir(parents=True, exist_ok=True); rng = np.random.default_rng(seed)
    yy, xx = np.mgrid[0:size, 0:size].astype(np.float32)
    for k in range(n):
        base = 0.22 + 0.10 * np.exp(-xx / (0.4 * size)) + 0.03 * np.sin(xx / rng.uniform(6, 14) + rng.uniform(0, 6))
        img = base.astype(np.float32); lines = []
        for _ in range(rng.integers(0, 4)):
            cls = int(rng.random() < 0.55)            # 0 = MILCO, 1 = NOMBO (YOLO ids, same as the real dataset)
            cx, cy = rng.uniform(40, size - 60), rng.uniform(40, size - 40)
            if cls == 0:
                ry, rx = rng.uniform(7, 11), rng.uniform(9, 14)
                hi = _ellipse(yy, xx, cy, cx, ry, rx, rng.uniform(-.2, .2)); sh = _ellipse(yy, xx, cy, cx + rx + 14, ry * .9, 16)
                img[hi] += 0.55
            else:
                hi = np.zeros_like(img, bool)
                for _ in range(rng.integers(2, 5)):
                    hi |= _ellipse(yy, xx, cy + rng.uniform(-8, 8), cx + rng.uniform(-8, 8), rng.uniform(4, 9), rng.uniform(5, 12), rng.uniform(0, 3))
                sh = _ellipse(yy, xx, cy, cx + 20, 6, 10); img[hi] += 0.32
            img[sh & ~hi] *= 0.25
            w, h = (2 * rx + 32, 2 * ry) if cls == 0 else (44, 30)
            lines.append(f"{cls} {(cx + w / 4) / size:.6f} {cy / size:.6f} {w / size:.6f} {h / size:.6f}")
        L = 3.0
        img = np.clip(img * rng.gamma(L, 1 / L, img.shape), 0, 1)
        Image.fromarray((img * 255).astype(np.uint8)).save(out / f"synthetic_{k:04d}.jpg", quality=95)
        (out / f"synthetic_{k:04d}.txt").write_text("\n".join(lines))
    return str(out)
