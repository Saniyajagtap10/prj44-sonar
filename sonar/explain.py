import torch
import torch.nn.functional as F


def gradcam(cnn, x, target=None, denoiser=None):
    """Grad-CAM on the last conv block. x: tensor (1,1,H,W). Returns (cam in [0,1] HxW numpy, class probs, class idx)."""
    cnn.eval()
    if denoiser is not None:
        with torch.no_grad():
            x = denoiser.eval()(x).clamp(0, 1)
    store = {}
    layer = cnn.features[-1]

    def fwd(_, __, out):
        store["a"] = out; out.register_hook(lambda g: store.__setitem__("g", g))
    h = layer.register_forward_hook(fwd)
    try:
        cnn.zero_grad()
        logits = cnn(x.clone().requires_grad_(False))
        c = int(logits.argmax(1)) if target is None else int(target)
        logits[0, c].backward()
    finally:
        h.remove()
    w = store["g"].mean((2, 3), keepdim=True)
    cam = F.relu((w * store["a"]).sum(1, keepdim=True))
    cam = F.interpolate(cam, size=x.shape[-2:], mode="bilinear", align_corners=False)[0, 0].detach()
    cam = (cam - cam.min()) / (cam.max() - cam.min() + 1e-8)
    return cam.numpy(), torch.softmax(logits, 1)[0].detach().numpy(), c
