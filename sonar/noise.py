"""Sonar speckle simulation + pseudo-clean reference generation."""
import numpy as np
import torch
from scipy.ndimage import gaussian_filter, median_filter, uniform_filter


def speckle_t(x, looks, rng, sigma_g=0.0):
    """Multiplicative gamma speckle (mean 1, variance 1/L) + optional additive Gaussian noise.
    x: tensor (B,1,H,W) in [0,1]; looks: float or array of length B; sigma_g: float or array of length B."""
    B = x.shape[0]
    L = np.broadcast_to(np.asarray(looks, dtype=np.float64), (B,)).reshape(B, 1, 1, 1)
    n = rng.gamma(L, 1.0 / L, size=tuple(x.shape)).astype(np.float32)
    y = x * torch.from_numpy(n)
    sg = np.broadcast_to(np.asarray(sigma_g, dtype=np.float64), (B,)).reshape(B, 1, 1, 1)
    if np.any(sg > 0):
        y = y + torch.from_numpy((rng.standard_normal(tuple(x.shape)) * sg).astype(np.float32))
    return y.clamp(0, 1)


def speckle_np(x, looks, rng):
    """numpy version for single images / stacks (any shape)."""
    n = rng.gamma(looks, 1.0 / looks, size=x.shape).astype(np.float32)
    return np.clip(x * n, 0, 1)


def pseudo_clean(p):
    """There is no real clean ground truth for side-scan sonar. We build a *pseudo-clean* reference by
    light median + Gaussian smoothing of the real patch. All SNR/PSNR/SSIM numbers are w.r.t. this proxy."""
    return gaussian_filter(median_filter(p, size=3, mode="reflect"), 1.0, mode="reflect").astype(np.float32)


def pseudo_clean_stack(X):
    return np.stack([pseudo_clean(p) for p in X])


def lee_filter(p, looks=4.0, size=5):
    """Classical Lee despeckle filter (standard baseline for sonar/SAR). Different from the pseudo-clean
    reference, so the comparison with the CNN denoiser is not circular."""
    mean = uniform_filter(p, size); var = np.maximum(uniform_filter(p * p, size) - mean ** 2, 0)
    ci2 = var / (mean ** 2 + 1e-8); w = np.clip(1 - (1.0 / looks) / (ci2 + 1e-8), 0, 1)
    return (mean + w * (p - mean)).astype(np.float32)


def filter_baseline(p, looks=4.0):
    return lee_filter(p, looks)
