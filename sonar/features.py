"""Hand-crafted sonar features for the classical-ML baselines (48x48 patch -> 67-d vector)."""
import numpy as np


def feats(p):
    gy, gx = np.gradient(p); mag = np.hypot(gx, gy); ang = np.arctan2(gy, gx)
    m, s = p.mean(), p.std() + 1e-6
    f = [m, s, p.max(), *np.percentile(p, [5, 25, 50, 75, 95]), (p > m + 2 * s).mean(),   # highlight fraction
         (p < m - 1.5 * s).mean(), mag.mean(), mag.max()]                                   # acoustic-shadow fraction
    f += list(np.histogram(p, 16, (0, 1))[0] / p.size)
    f += list(np.histogram(ang, 8, (-np.pi, np.pi), weights=mag)[0] / (mag.sum() + 1e-6))
    g = p.reshape(4, 12, 4, 12)
    f += list(g.mean((1, 3)).ravel()) + list(g.std((1, 3)).ravel())
    return np.array(f, dtype=np.float32)


def feats_stack(X):
    return np.stack([feats(np.ascontiguousarray(p)) for p in X])
