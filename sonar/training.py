"""Training loops (CPU-friendly) for the deconvolutional denoiser and the CNN classifiers."""
import copy
import time

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F

from .metrics import psnr_db
from .models import DenoiseNet, SonarCNN
from .noise import pseudo_clean_stack, speckle_t
from sklearn.metrics import balanced_accuracy_score

LOOKS = np.array([1, 2, 4, 8, 16])


def _t(a):
    return torch.from_numpy(np.ascontiguousarray(a, dtype=np.float32))[:, None]


@torch.no_grad()
def denoise(net, X, bs=256):
    """X: (N,H,W) numpy -> denoised numpy (N,H,W)."""
    net.eval(); out = []
    for i in range(0, len(X), bs):
        out.append(net(_t(X[i:i + bs])).clamp(0, 1)[:, 0].numpy())
    return np.concatenate(out) if out else np.zeros_like(X)


def train_denoiser(Xtr, Xva, cfg, cb=lambda f, m: None, p_real=0.3):
    """Noisy sonar -> clean image.
    'n2c': input = pseudo-clean x synthetic speckle, target = pseudo-clean.
    'n2n': Noise2Noise, input & target are two independent speckle realisations of the pseudo-clean image.
    With prob. p_real the *real* (already noisy) patch is used as input and pseudo-clean as target, so the
    network also sees genuine sonar noise."""
    torch.manual_seed(cfg.seed); rng = np.random.default_rng(cfg.seed)
    ref_tr, ref_va = pseudo_clean_stack(Xtr), pseudo_clean_stack(Xva)
    net = DenoiseNet(); opt = torch.optim.AdamW(net.parameters(), 2e-3, weight_decay=1e-4)
    spe = int(np.ceil(len(Xtr) / cfg.batch))
    E = max(2, cfg.dn_epochs // (2 if cfg.fast else 1), int(np.ceil(400 / spe))); steps = E * spe   # >=400 optimisation steps even on tiny datasets
    sched = torch.optim.lr_scheduler.OneCycleLR(opt, 4e-3, total_steps=steps)
    rn = np.random.default_rng(0); Lv = LOOKS[rn.integers(0, len(LOOKS), len(Xva))]
    val_in = speckle_t(_t(ref_va), Lv, np.random.default_rng(1))
    hist = {"epoch": [], "train_loss": [], "val_psnr": []}; best, best_state = -1e9, None
    for ep in range(E):
        net.train(); perm = rng.permutation(len(Xtr)); tl = []
        for i in range(0, len(perm), cfg.batch):
            b = perm[i:i + cfg.batch]
            ref = _t(ref_tr[b]); real = _t(Xtr[b])
            if rng.random() < 0.5:
                ref, real = ref.flip(-1), real.flip(-1)
            L = LOOKS[rng.integers(0, len(LOOKS), len(b))]; sg = rng.uniform(0, 0.03, len(b))
            noisy = speckle_t(ref, L, rng, sg)
            target = speckle_t(ref, L, rng, sg) if cfg.dn_mode == "n2n" else ref
            if rng.random() < p_real:          # real-noise step: real patch -> pseudo-clean
                noisy, target = real, ref
            pred = net(noisy)
            loss = F.l1_loss(pred, target) + 0.5 * F.mse_loss(pred, target)
            opt.zero_grad(); loss.backward(); opt.step(); sched.step(); tl.append(loss.item())
        with torch.no_grad():
            net.eval(); pv = net(val_in).clamp(0, 1)[:, 0].numpy()
        vp = float(np.mean([psnr_db(r, p) for r, p in zip(ref_va, pv)]))
        hist["epoch"].append(ep + 1); hist["train_loss"].append(float(np.mean(tl))); hist["val_psnr"].append(vp)
        if vp > best:
            best, best_state = vp, copy.deepcopy(net.state_dict())
        cb((ep + 1) / E, f"Denoiser epoch {ep + 1}/{E}  val PSNR {vp:.2f} dB")
    net.load_state_dict(best_state); net.eval()
    return net, hist


def _prep(x, denoiser, aug_p, rng, train):
    if train and aug_p > 0 and rng.random() < aug_p:
        L = LOOKS[rng.integers(0, len(LOOKS), x.shape[0])]
        x = speckle_t(x, L, rng, rng.uniform(0, 0.02, x.shape[0]))
    if denoiser is not None:
        with torch.no_grad():
            x = denoiser(x).clamp(0, 1)
    return x


@torch.no_grad()
def predict_probs(cnn, X, denoiser=None, looks=None, seed=0, bs=256):
    cnn.eval(); rng = np.random.default_rng(seed); out = []
    for i in range(0, len(X), bs):
        x = _t(X[i:i + bs])
        if looks:
            x = speckle_t(x, float(looks), rng)
        if denoiser is not None:
            x = denoiser(x).clamp(0, 1)
        out.append(torch.softmax(cnn(x), 1).numpy())
    return np.concatenate(out)


def train_cnn(Xtr, ytr, Xva, yva, cfg, denoiser=None, noise_aug=False, tag="cnn", cb=lambda f, m: None):
    """CNN classifier. denoiser=None -> raw input; otherwise input is D(x) (denoiser frozen)."""
    torch.manual_seed(cfg.seed); rng = np.random.default_rng(cfg.seed + 1)
    net = SonarCNN(); opt = torch.optim.AdamW(net.parameters(), 2e-3, weight_decay=1e-3)
    spe = int(np.ceil(len(Xtr) / cfg.batch))
    E = max(3, cfg.cnn_epochs // (2 if cfg.fast else 1), int(np.ceil(500 / spe))); steps = E * spe   # >=500 steps so BatchNorm statistics settle
    sched = torch.optim.lr_scheduler.OneCycleLR(opt, 4e-3, total_steps=steps)
    cw = torch.tensor(len(ytr) / (3 * np.maximum(np.bincount(ytr, minlength=3), 1)), dtype=torch.float32)
    aug_p = cfg.noise_aug_p if (noise_aug or denoiser is not None) else 0.0
    hist = {"epoch": [], "train_loss": [], "val_loss": [], "val_bal_acc": []}; best, best_state = -1, None
    if denoiser is not None:
        denoiser.eval()
    for ep in range(E):
        cnn_train = net.train(); perm = rng.permutation(len(Xtr)); tl = []
        for i in range(0, len(perm), cfg.batch):
            b = perm[i:i + cfg.batch]; x = _t(Xtr[b]); yb = torch.from_numpy(ytr[b])
            if rng.random() < 0.5:
                x = x.flip(-1)                                   # port/starboard flip is physically valid
            x = (x * torch.from_numpy(rng.uniform(0.9, 1.1, (len(b), 1, 1, 1)).astype(np.float32))).clamp(0, 1)
            x = _prep(x, denoiser, aug_p, rng, True)
            loss = F.cross_entropy(net(x), yb, weight=cw, label_smoothing=0.05)
            opt.zero_grad(); loss.backward(); opt.step(); sched.step(); tl.append(loss.item())
        pv = predict_probs(net, Xva, denoiser)
        vl = float(F.nll_loss(torch.log(torch.from_numpy(pv) + 1e-8), torch.from_numpy(yva)))
        vb = float(balanced_accuracy_score(yva, pv.argmax(1)))
        for k, v in zip(hist, [ep + 1, float(np.mean(tl)), vl, vb]):
            hist[k].append(v)
        if vb >= best:
            best, best_state = vb, copy.deepcopy(net.state_dict())
        cb((ep + 1) / E, f"{tag} epoch {ep + 1}/{E}  val bal.acc {vb:.3f}")
    net.load_state_dict(best_state); net.eval()
    return net, hist
