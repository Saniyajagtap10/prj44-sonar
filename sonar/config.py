from dataclasses import dataclass, asdict

CLASSES = ["Background", "MILCO", "NOMBO"]   # label ids 0 / 1 / 2
CLASS_LONG = ["Background", "MILCO (mine-like contact)", "NOMBO (non-mine bottom object)"]
WIN, SZ = 96, 48          # crop window (px in the original image) and network patch size
NOISE_LEVELS = (16, 8, 4, 2, 1)   # speckle "looks" L (smaller L = noisier)


@dataclass
class Config:
    data_root: str = "data"
    out_dir: str = "artifacts"
    milco_id: int = 0            # YOLO class id that means MILCO (the other one is NOMBO)
    seed: int = 42
    bg_ratio: float = 1.0        # background patches per object patch
    max_images: int = 0          # 0 = all images (use a small number for a quick smoke test)
    dn_epochs: int = 20
    cnn_epochs: int = 25
    batch: int = 64
    noise_aug_p: float = 0.5     # prob. of adding synthetic speckle during CNN training (aug + denoise variants)
    dn_mode: str = "n2c"         # "n2c" = noisy -> pseudo-clean, "n2n" = noisy -> independently-noisy (Noise2Noise)
    det_scales: tuple = (64, 96, 128)
    det_stride: int = 48
    det_eval_images: int = 60    # test images used for detection mAP (0 = all)
    fast: bool = False           # halves epochs, skips CV for classical models

    def to_dict(self):
        d = asdict(self)
        d["det_scales"] = list(self.det_scales)
        return d
