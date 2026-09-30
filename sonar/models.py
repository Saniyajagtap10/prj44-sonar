import torch
import torch.nn as nn
import torch.nn.functional as F


def cbr(i, o, k=3, s=1):
    return nn.Sequential(nn.Conv2d(i, o, k, s, k // 2, bias=False), nn.BatchNorm2d(o), nn.ReLU(inplace=False))


class DenoiseNet(nn.Module):
    """Deconvolutional encoder-decoder (U-Net style) for sonar despeckling.
    Encoder: strided convs (48 -> 24 -> 12). Decoder: *transposed convolutions* (12 -> 24 -> 48) with skip
    connections. Residual output: y = x + f(x) so the net only has to learn the noise."""

    def __init__(self, c=16):
        super().__init__()
        self.e1 = cbr(1, c); self.e2 = cbr(c, 2 * c, s=2); self.e3 = cbr(2 * c, 4 * c, s=2); self.mid = cbr(4 * c, 4 * c)
        self.up1 = nn.ConvTranspose2d(4 * c, 2 * c, 4, 2, 1); self.d1 = cbr(4 * c, 2 * c)
        self.up2 = nn.ConvTranspose2d(2 * c, c, 4, 2, 1); self.d2 = cbr(2 * c, c)
        self.out = nn.Conv2d(c, 1, 1)

    def forward(self, x):
        H, W = x.shape[-2:]
        ph, pw = (-H) % 4, (-W) % 4
        xp = F.pad(x, (0, pw, 0, ph), mode="reflect") if (ph or pw) else x
        e1 = self.e1(xp); e2 = self.e2(e1); e3 = self.mid(self.e3(e2))
        d1 = self.d1(torch.cat([self.up1(e3), e2], 1))
        d2 = self.d2(torch.cat([self.up2(d1), e1], 1))
        y = xp + self.out(d2)
        return y[..., :H, :W]


class SonarCNN(nn.Module):
    """Small CPU-friendly CNN: Background / MILCO / NOMBO from a 48x48 patch (~100k params)."""

    def __init__(self, nc=3, w=16, drop=0.3):
        super().__init__()
        self.features = nn.Sequential(cbr(1, w), nn.MaxPool2d(2), cbr(w, 2 * w), nn.MaxPool2d(2),
                                      cbr(2 * w, 4 * w), nn.MaxPool2d(2), cbr(4 * w, 4 * w))
        self.head = nn.Sequential(nn.AdaptiveAvgPool2d(1), nn.Flatten(), nn.Dropout(drop), nn.Linear(4 * w, nc))

    def forward(self, x):
        return self.head(self.features(x))


def count_params(m):
    return sum(p.numel() for p in m.parameters())
