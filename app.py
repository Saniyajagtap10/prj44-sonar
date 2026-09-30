"""PRJ-44 | Deconvolutional Feature-Denoising CNN for AUV side-scan sonar hazard detection.
Run:  python -m streamlit run app.py     (train first with `python train.py`, or press "Train now" in the app)"""
import io
import json
import os
import time
import zipfile
from pathlib import Path

import joblib
import matplotlib
import numpy as np
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
import streamlit as st
import torch
import torch.nn.functional as F
from PIL import Image, ImageDraw
from sklearn.metrics import auc, confusion_matrix, precision_recall_curve, roc_curve

from sonar import data as D
from sonar.config import CLASSES, Config, NOISE_LEVELS, SZ, WIN
from sonar.detect import extract_detections, heatmap, scan_cnn
from sonar.explain import gradcam
from sonar.metrics import image_quality
from sonar.models import DenoiseNet, SonarCNN
from sonar.noise import filter_baseline, pseudo_clean, speckle_np
from sonar.pipeline import ALL_MODELS, run_pipeline
from sonar.report import (ablation_df, build_pdf, comparison_df, denoise_df, detection_df, perclass_df, robustness_df)
from sonar.training import denoise

st.set_page_config(page_title="PRJ-44 Sonar Denoising & Hazard Detection", page_icon="🌊", layout="wide")
st.markdown("""<style>.block-container{padding-top:1.6rem}
.hero{background:linear-gradient(90deg,#0b3c5d,#1d7a9c);padding:1.2rem 1.6rem;border-radius:12px;color:#fff;margin-bottom:.8rem}
.hero h1{margin:0;font-size:1.7rem;color:#fff}.hero p{margin:.3rem 0 0;opacity:.92;color:#fff}
.note{border-left:4px solid #1d7a9c;padding:.4rem .8rem;border-radius:4px;background:rgba(29,122,156,.10)}</style>
<div class="hero"><h1>🌊 PRJ-44 · Deconvolutional Feature-Denoising CNN for AUV Sonar</h1>
<p>Speckle denoising with SNR improvement → CNN classification (Background / MILCO / NOMBO) → Grad-CAM → CNN-heatmap detector</p></div>""",
            unsafe_allow_html=True)

C3 = ["Background", "MILCO", "NOMBO"]
READONLY = True   # public demo: no training / deleting


def plot(fig, **kw):
    fig.update_layout(margin=dict(l=10, r=10, t=40, b=10)); st.plotly_chart(fig, width="stretch", **kw)


# ---------------- sidebar ----------------
with st.sidebar:
    st.header("⚙️ Settings")
    root = st.text_input("Dataset folder", "data", disabled=READONLY)
    art = st.text_input("Artifacts folder", "artifacts", disabled=READONLY)
    milco_id = st.radio("YOLO class id of MILCO", [0, 1], horizontal=True, disabled=READONLY,
                        help="Check the dataset README. The other id is treated as NOMBO. Changing it requires re-training.")
    st.caption("Dataset: Pessanha Santos et al., Data in Brief 2024 (CC BY 4.0), doi:10.6084/m9.figshare.24574879")
    st.caption("Theme: use ⋮ (top-right) → Settings → Theme for light / dark.")
    if not READONLY:
      with st.expander("🧪 No dataset yet? Synthetic smoke-test data"):
        st.caption("Only checks that the code runs. Results on synthetic data have NO scientific meaning.")
        if st.button("Generate 80 synthetic images into dataset folder"):
            from sonar.synth import make_synthetic
            make_synthetic(root, 80); st.success("Done. Now press *Train now*.")
    if READONLY:
        st.info("Read-only demo: shows pre-trained results. Training and dataset upload are disabled.")

ART = Path(art)
if not ART.is_absolute() and not ART.exists():
    ART = Path(__file__).parent / art
RES = ART / "results.json"


# ---------------- training panel (if no artifacts yet) ----------------
def train_panel():
    if READONLY:
        st.error("This demo has no trained artifacts. Deployment is incomplete (artifacts/ folder missing).")
        return
    st.warning("No trained artifacts found. Train once (CPU is fine), the app will then load everything from the artifacts folder.")
    imgs = D.list_images(root)
    if not imgs:
        st.error(f"No images in '{root}'. Download the Figshare dataset (doi:10.6084/m9.figshare.24574879), unzip it into that folder and refresh.")
        return
    st.write(f"Found **{len(imgs)}** images in `{root}`.")
    c = st.columns(4)
    fast = c[0].checkbox("Fast mode (half the epochs)", True); dn_e = c[1].number_input("Denoiser epochs", 2, 100, 20)
    cnn_e = c[2].number_input("CNN epochs", 3, 100, 25); mode = c[3].selectbox("Denoiser training", ["n2c", "n2n"], help="n2c: noisy→pseudo-clean; n2n: Noise2Noise")
    limit = st.number_input("Limit images (0 = all; use ~100 for a quick test)", 0, 5000, 0)
    if st.button("🚀 Train now", type="primary"):
        bar = st.progress(0.0, text="Starting..."); cfg = Config(data_root=root, out_dir=art, milco_id=milco_id, fast=fast, dn_epochs=int(dn_e),
                                                                cnn_epochs=int(cnn_e), dn_mode=mode, max_images=int(limit))
        try:
            run_pipeline(cfg, lambda f, m: bar.progress(min(1.0, f), text=m))
        except Exception as e:
            st.error(f"Training failed: {e}"); return
        load_bundle.clear(); st.rerun()


@st.cache_resource(show_spinner="Loading models...")
def load_bundle(art_dir, stamp):
    a = Path(art_dir); res = json.loads((a / "results.json").read_text())
    dn = DenoiseNet(); dn.load_state_dict(torch.load(a / "denoiser.pt")); dn.eval()
    cnn = {}
    for k, f in [("CNN (raw)", "cnn_raw"), ("CNN (noise-aug)", "cnn_aug"), ("Denoise + CNN", "cnn_dn")]:
        m = SonarCNN(); m.load_state_dict(torch.load(a / f"{f}.pt")); m.eval(); cnn[k] = m
    P = np.load(a / "preds.npz"); patches = np.load(a / "patches.npz")
    return {"res": res, "dn": dn, "cnn": cnn, "classical": joblib.load(a / "classical.joblib"),
            "preds": {n: P[n.replace(" ", "_")] for n in ALL_MODELS}, "y": P["y"], "img": P["img"],
            "X": patches["X"], "yy": patches["y"], "split": patches["split"]}


if not RES.exists():
    train_panel(); st.stop()
B = load_bundle(str(ART), RES.stat().st_mtime); res = B["res"]; dn = B["dn"]
best_cl = res["classification"]["_best_classical"]
if not READONLY and res["meta"]["config"]["milco_id"] != milco_id:
    st.info(f"Artifacts were trained with MILCO id = {res['meta']['config']['milco_id']} (sidebar shows {milco_id}). Re-train to change it.")
milco_id = res["meta"]["config"]["milco_id"]

tabs = st.tabs(["📊 Dataset", "🔇 Denoising & SNR", "🧪 Ablation", "📈 Evaluation", "🔍 Explainability",
                "🎯 Detector", "📉 Training & Export", "ℹ️ About / Methodology"])


def find_path(name_or_path):
    p = Path(name_or_path); return str(p) if p.exists() else None


def draw(img, gt=(), det=()):
    im = Image.fromarray((np.clip(img, 0, 1) * 255).astype(np.uint8)).convert("RGB"); d = ImageDraw.Draw(im)
    for c, x1, y1, x2, y2 in gt:
        d.rectangle([x1, y1, x2, y2], outline="red" if c == 1 else "orange", width=3)
    for c, s, x1, y1, x2, y2 in det:
        col = "cyan" if c == 1 else "magenta"; d.rectangle([x1, y1, x2, y2], outline=col, width=2); d.text((x1 + 3, y1 + 3), f"{C3[c]} {s:.0%}", fill=col)
    return im


def up(p, k=4):
    return Image.fromarray((np.clip(p, 0, 1) * 255).astype(np.uint8)).resize((p.shape[1] * k, p.shape[0] * k), Image.NEAREST)


# ============ 1. DATASET ============
with tabs[0]:
    m = res["meta"]; cc = m["class_counts"]; tot = np.array(cc["train"]) + np.array(cc["val"]) + np.array(cc["test"])
    c = st.columns(5)
    c[0].metric("Sonar images", m["n_images"]); c[1].metric("Object patches", int(tot[1] + tot[2])); c[2].metric("MILCO patches", int(tot[1]))
    c[3].metric("NOMBO patches", int(tot[2])); c[4].metric("Background patches", int(tot[0]))
    st.markdown(f"<div class='note'>Split is <b>by image</b>: train {m['patches']['train']} / val {m['patches']['val']} / test {m['patches']['test']} patches. "
                "No patch of a test image is ever seen in training.</div>", unsafe_allow_html=True)
    st.subheader("🖼️ Sample gallery (real patches, 48×48)")
    g1, g2 = st.columns([1, 3]); cl = g1.radio("Class", [0, 1, 2], format_func=lambda i: C3[i], horizontal=False, key="gal_c")
    sp = g1.selectbox("Split", ["test", "train", "val"], key="gal_s"); sid = {"train": 0, "val": 1, "test": 2}[sp]
    sel = np.where((B["yy"] == cl) & (B["split"] == sid))[0][:12]
    if len(sel):
        cols = g2.columns(6)
        for k, i in enumerate(sel):
            cols[k % 6].image(up(B["X"][i], 3), caption=f"#{i}")
    else:
        g2.info("No patches for this selection.")
    imgs = [p for p in res["image_paths"] if Path(p).exists()]
    if imgs:
        st.subheader("Browse full images with ground truth")
        ann = [p for p in imgs if D.read_boxes(p, milco_id)] or imgs
        pick = st.selectbox("Annotated image", ann, format_func=lambda s: Path(s).name)
        arr = D.load_gray(pick); H, W = arr.shape
        st.image(draw(arr, D.yolo_to_px(D.read_boxes(pick, milco_id), W, H)), caption="Red = MILCO, orange = NOMBO", width="stretch")
    else:
        st.info("Original images are not available on this machine (only artifacts). Gallery above still works.")

# ============ 2. DENOISING & SNR ============
with tabs[1]:
    st.subheader("Deconvolutional denoising network → SNR improvement")
    st.markdown("Encoder (strided conv) → decoder (**transposed convolutions**) with skip connections. Speckle is simulated with a gamma "
                "model (L looks; smaller L = noisier) because **real clean ground truth does not exist** for side-scan sonar. "
                "SNR / PSNR / SSIM are measured against a *pseudo-clean* reference (median+Gaussian smoothed image).")
    dq = pd.DataFrame(res["denoise"]["by_noise"])
    c = st.columns(3)
    mid = dq.iloc[len(dq) // 2]
    c[0].metric(f"SNR improvement @ L={int(mid['looks'])}", f"{mid['SNR_improvement']:+.2f} dB", f"{mid['SNR_improvement'] - mid['SNR_improvement_filter']:+.2f} dB vs Lee filter")
    c[1].metric(f"PSNR after @ L={int(mid['looks'])}", f"{mid['PSNR_after']:.2f} dB", f"{mid['PSNR_after'] - mid['PSNR_before']:+.2f} dB")
    c[2].metric(f"SSIM after @ L={int(mid['looks'])}", f"{mid['SSIM_after']:.3f}", f"{mid['SSIM_after'] - mid['SSIM_before']:+.3f}")
    l, r = st.columns(2)
    f1 = go.Figure()
    f1.add_bar(x=dq.looks.astype(str), y=dq.SNR_improvement, name="Deconv denoiser"); f1.add_bar(x=dq.looks.astype(str), y=dq.SNR_improvement_filter, name="Lee filter")
    f1.update_layout(title="SNR improvement (dB) vs speckle level (test patches)", xaxis_title="looks L (left = cleaner)", yaxis_title="dB", barmode="group"); l.plotly_chart(f1, width="stretch")
    f2 = go.Figure()
    for k in ("SNR", "PSNR"):
        f2.add_scatter(x=dq.looks.astype(str), y=dq[f"{k}_before"], name=f"{k} before", mode="lines+markers", line=dict(dash="dot"))
        f2.add_scatter(x=dq.looks.astype(str), y=dq[f"{k}_after"], name=f"{k} after", mode="lines+markers")
    f2.update_layout(title="Before vs after (dB)", xaxis_title="looks L"); r.plotly_chart(f2, width="stretch")
    with st.expander("Full table (SNR, PSNR, SSIM, before/after)"):
        st.dataframe(dq.style.format({k: "{:.3f}" for k in dq.columns if k != "looks"}), width="stretch")
    rr = res["denoise"]["real_no_reference"]
    if rr:
        st.markdown("**No-reference metrics on the real (un-modified) test patches** (higher = smoother speckle / clearer target)")
        st.dataframe(pd.DataFrame(rr).T.rename(columns={"raw": "Raw", "denoised": "Deconv denoiser", "lee_filter": "Lee filter"}).style.format("{:.3f}"), width="stretch")
    st.divider(); st.subheader("Before / after slider")
    timgs = [p for p in res["test_images"] if Path(p).exists()]
    up_f = st.file_uploader("Upload your own sonar image (optional)", type=["jpg", "jpeg", "png"], key="dn_up")
    if up_f:
        arr = np.asarray(Image.open(up_f).convert("L"), dtype=np.float32) / 255
    elif timgs:
        arr = D.load_gray(st.selectbox("Unseen test image", timgs, format_func=lambda s: Path(s).name, key="dn_img"))
    else:
        arr = None; st.info("Original images not available; upload one to use the slider.")
    if arr is not None:
        s = WIN // SZ; ts = F.interpolate(torch.from_numpy(arr)[None, None], scale_factor=1 / s, mode="area")[0, 0].numpy()
        a, b = st.columns(2)
        mode = a.radio("Input", ["Simulate extra speckle", "Use image as-is"], horizontal=True)
        L = b.select_slider("Speckle looks L (noisier ←)", options=[1, 2, 4, 8, 16], value=2, disabled=mode != "Simulate extra speckle")
        if mode == "Simulate extra speckle":
            ref = pseudo_clean(ts); noisy = speckle_np(ref, float(L), np.random.default_rng(3))
        else:
            ref, noisy = None, ts
        den = denoise(dn, noisy[None])[0]
        pos = st.slider("Wipe position (left = noisy input, right = denoised)", 0, 100, 50)
        comp = noisy.copy(); x = int(noisy.shape[1] * pos / 100); comp[:, x:] = den[:, x:]
        im = Image.fromarray((comp * 255).astype(np.uint8)).convert("RGB"); ImageDraw.Draw(im).line([(x, 0), (x, im.height)], fill=(255, 220, 0), width=2)
        st.image(im, caption="◀ noisy | denoised ▶", width="stretch")
        if ref is not None:
            q0, q1, q2 = image_quality(ref, noisy), image_quality(ref, den), image_quality(ref, filter_baseline(noisy, float(L)))
            t = pd.DataFrame({"Noisy": q0, "Deconv denoiser": q1, "Lee filter": q2}).T
            t["SNR improvement (dB)"] = t["SNR"] - q0["SNR"]; st.dataframe(t.style.format("{:.3f}"), width="stretch")
        else:
            st.caption("SNR/PSNR/SSIM need a clean reference, so they are only computed in simulation mode.")

# ============ 3. ABLATION ============
with tabs[2]:
    st.subheader("Ablation: Classical ML vs CNN vs Denoise + CNN")
    ab = ablation_df(res); st.dataframe(ab.style.format({k: "{:.1f}" for k in ab.columns if "(%)" in k}), width="stretch")
    st.caption("All systems share the same image-wise split. 'CNN + noise augmentation' is a control: it sees the same synthetic speckle as the denoise variant, "
               "so any remaining gain of 'Denoise + CNN' is attributable to the deconvolutional denoiser itself. 95% CI: image-level bootstrap.")
    mm = ab.melt(id_vars="System", value_vars=["Balanced Acc (%)", "Macro F1 (%)", "MILCO Recall (%)"])
    plot(px.bar(mm, x="System", y="value", color="variable", barmode="group", title="Clean test set"))
    st.markdown("#### Robustness to additional speckle (test patches corrupted after the split)")
    rb = robustness_df(res); mt = st.radio("Metric", ["Balanced Acc (%)", "MILCO Recall (%)", "Macro F1 (%)"], horizontal=True)
    plot(px.line(rb, x="Noise (looks L)", y=mt, color="System", markers=True, title=f"{mt} vs noise level (clean → noisier)"))
    st.markdown("#### McNemar exact test (paired predictions)")
    mc = pd.DataFrame(res["mcnemar"]); mc["Significant (p<0.05)"] = mc.p_value < 0.05
    st.dataframe(mc.style.format({"p_value": "{:.4f}"}), width="stretch")
    st.caption("A_only_correct = patches only A got right; B_only_correct = only B got right. If p ≥ 0.05, the difference is not statistically distinguishable on this test set. Report honestly either way.")
    st.markdown("#### All models")
    cmp_ = comparison_df(res); st.dataframe(cmp_.style.format({k: "{:.2f}" for k in cmp_.columns if k != "Model"}).background_gradient(cmap="Blues", subset=["Balanced Acc (%)", "MILCO Recall (%)"]), width="stretch")
    st.download_button("⬇️ Ablation (CSV)", ab.to_csv(index=False), "ablation.csv"); st.download_button("⬇️ Model comparison (CSV)", cmp_.to_csv(index=False), "model_comparison.csv")

# ============ 4. EVALUATION ============
with tabs[3]:
    y, P = B["y"], B["preds"]
    st.subheader("Per-class metrics, ROC / PR, MILCO focus")
    sel = st.selectbox("Model", ALL_MODELS, index=ALL_MODELS.index("Denoise + CNN"))
    pc = res["classification"][sel]["per_class"]; ci = res["classification"][sel]["ci"]
    t = pd.DataFrame({"Class": C3, "Precision": pc["precision"], "Recall": pc["recall"], "F1": pc["f1"], "Support": pc["support"]})
    l, r = st.columns(2); l.dataframe(t.style.format({"Precision": "{:.3f}", "Recall": "{:.3f}", "F1": "{:.3f}"}), width="stretch")
    l.markdown("**95% CI (image bootstrap):** " + " · ".join(f"{k} {v[0] * 100:.1f}–{v[1] * 100:.1f}%" for k, v in ci.items()))
    cm = confusion_matrix(y, P[sel].argmax(1), labels=[0, 1, 2])
    r.plotly_chart(px.imshow(cm, text_auto=True, x=C3, y=C3, labels=dict(x="Predicted", y="Actual"), color_continuous_scale="Blues", title="Confusion matrix"), width="stretch")
    st.markdown("##### ROC & Precision-Recall (one-vs-rest)")
    a, b = st.columns([1, 3]); cls = a.radio("Class", [1, 2, 0], format_func=lambda i: C3[i])
    ms = b.multiselect("Models", ALL_MODELS, default=[best_cl, "CNN (raw)", "Denoise + CNN"])
    f_roc, f_pr = go.Figure(), go.Figure()
    for n in ms:
        yt = (y == cls).astype(int); s = P[n][:, cls]
        if yt.sum() == 0 or yt.sum() == len(yt):
            continue
        fp, tp, _ = roc_curve(yt, s); pr, rc, _ = precision_recall_curve(yt, s)
        f_roc.add_scatter(x=fp, y=tp, name=f"{n} (AUC {auc(fp, tp):.3f})"); f_pr.add_scatter(x=rc, y=pr, name=f"{n} (AP {-np.trapezoid(pr, rc):.3f})")
    f_roc.add_scatter(x=[0, 1], y=[0, 1], line=dict(dash="dot", color="grey"), showlegend=False)
    f_roc.update_layout(title=f"ROC · {C3[cls]}", xaxis_title="FPR", yaxis_title="TPR"); f_pr.update_layout(title=f"PR · {C3[cls]}", xaxis_title="Recall", yaxis_title="Precision")
    l, r = st.columns(2); l.plotly_chart(f_roc, width="stretch"); r.plotly_chart(f_pr, width="stretch")
    st.markdown("##### 🎯 MILCO recall focus: choose the operating point")
    thr = st.slider("MILCO probability threshold", 0.05, 0.95, 0.5, 0.05)
    rows = []
    for n in ms or [sel]:
        pm = P[n][:, 1] >= thr; tm = y == 1
        rows.append({"Model": n, "MILCO recall": (pm & tm).sum() / max(tm.sum(), 1), "MILCO precision": (pm & tm).sum() / max(pm.sum(), 1),
                     "Missed mines (FN)": int((~pm & tm).sum()), "False alarms (FP)": int((pm & ~tm).sum())})
    st.dataframe(pd.DataFrame(rows).style.format({"MILCO recall": "{:.3f}", "MILCO precision": "{:.3f}"}), width="stretch")
    st.caption("For mine countermeasures a missed mine (FN) is far costlier than a false alarm: lower the threshold to trade precision for recall.")
    pcd = perclass_df(res); st.download_button("⬇️ Per-class metrics (CSV)", pcd.to_csv(index=False), "per_class_metrics.csv")

# ============ 5. EXPLAINABILITY ============
with tabs[4]:
    st.subheader("Grad-CAM: where is the CNN looking?")
    st.caption("Heat-map of the last conv block for the chosen class. For a sensible model MILCO should light up on the bright highlight and its acoustic shadow, not on random speckle.")
    a, b, c = st.columns(3); cl = a.radio("True class", [1, 2, 0], format_func=lambda i: C3[i], horizontal=True, key="gc_c")
    mk = b.selectbox("Model", ["Denoise + CNN", "CNN (raw)", "CNN (noise-aug)"]); tgt = c.selectbox("Explain class", ["Predicted", *C3])
    idx = np.where((B["yy"] == cl) & (B["split"] == 2))[0]
    if len(idx) == 0:
        st.info("No test patches of this class.")
    else:
        k = st.slider("Test patch", 1, len(idx), 1); i = idx[k - 1]; p = B["X"][i]
        x = torch.from_numpy(p)[None, None]
        cam, pr, pc_ = gradcam(B["cnn"][mk], x, None if tgt == "Predicted" else C3.index(tgt), dn if mk == "Denoise + CNN" else None)
        with torch.no_grad():
            dpatch = dn(x).clamp(0, 1)[0, 0].numpy()
        base = dpatch if mk == "Denoise + CNN" else p
        heat = matplotlib.colormaps["jet"](cam)[..., :3]; ov = 0.55 * np.stack([base] * 3, -1) + 0.45 * heat
        cols = st.columns(4)
        cols[0].image(up(p, 5), caption="Input (real patch)"); cols[1].image(up(dpatch, 5), caption="Denoised")
        cols[2].image(Image.fromarray((ov * 255).astype(np.uint8)).resize((240, 240), Image.BILINEAR), caption=f"Grad-CAM → {C3[pc_]}")
        cols[3].plotly_chart(px.bar(x=C3, y=pr, range_y=[0, 1], labels=dict(x="", y="probability"), title="Model output"), width="stretch")
        ok = "✅ correct" if int(np.argmax(pr)) == cl else "❌ wrong"; st.markdown(f"True: **{C3[cl]}** · Predicted: **{C3[int(np.argmax(pr))]}** ({ok})")

# ============ 6. DETECTOR ============
with tabs[5]:
    st.subheader("CNN-heatmap detector (multi-scale) + mAP@0.5")
    dm = res["detection"]; mt = pd.DataFrame(dm["methods"]).T
    st.markdown(f"**Detection benchmark** on {dm['n_images']} unseen test images, IoU ≥ 0.5, ground truth = YOLO boxes; windows {dm['scales']} px, stride {dm['stride']} px.")
    show = mt.rename(columns={"AP50_MILCO": "AP@0.5 MILCO", "AP50_NOMBO": "AP@0.5 NOMBO", "mAP50": "mAP@0.5", "mAP30": "mAP@0.3", "recall_MILCO": "MILCO recall", "sec_per_image": "s / image"})[["AP@0.5 MILCO", "AP@0.5 NOMBO", "mAP@0.5", "mAP@0.3", "MILCO recall", "s / image"]]
    st.dataframe(show.style.format("{:.3f}"), width="stretch")
    plot(px.bar(show.reset_index().melt(id_vars="index", value_vars=["AP@0.5 MILCO", "AP@0.5 NOMBO", "mAP@0.5", "mAP@0.3"]), x="index", y="value", color="variable", barmode="group", labels=dict(index="Detector")))
    st.caption("Boxes use the median training-box size per class (anchor) centred by soft-argmax; there is no learned box regression, so mAP@0.5 is conservative. mAP@0.3 is shown as a looser localisation reference.")
    st.divider(); st.markdown("#### Interactive scan")
    a, b, c = st.columns(3); dmk = a.selectbox("Detector", ["Denoise + CNN", "CNN (raw)"], key="det_m")
    thr = b.slider("Score threshold", 0.1, 0.99, 0.6, 0.05); cls_show = c.multiselect("Classes", ["MILCO", "NOMBO"], default=["MILCO"])
    up2 = st.file_uploader("Upload a sonar image (optional)", type=["jpg", "jpeg", "png"], key="det_up"); gt = []
    timgs = [p for p in res["test_images"] if Path(p).exists()]
    if up2:
        arr = np.asarray(Image.open(up2).convert("L"), dtype=np.float32) / 255
    elif timgs:
        pth = st.selectbox("Unseen test image", timgs, format_func=lambda s: Path(s).name, key="det_img"); arr = D.load_gray(pth)
        gt = D.yolo_to_px(D.read_boxes(pth, milco_id), arr.shape[1], arr.shape[0])
    else:
        arr = None; st.info("Original images not on this machine; upload one.")
    if arr is not None:
        t0 = time.time(); scan = scan_cnn(arr, B["cnn"][dmk], dn if dmk == "Denoise + CNN" else None, tuple(dm["scales"]), dm["stride"])
        dets = [d for d in extract_detections(scan, min_score=thr, anchors={1: tuple(dm["anchors"]["MILCO"]), 2: tuple(dm["anchors"]["NOMBO"])}) if C3[d[0]] in cls_show]; el = time.time() - t0
        l, r = st.columns(2); l.image(draw(arr, gt, dets), caption="Red/orange = ground truth · cyan = MILCO · magenta = NOMBO", width="stretch")
        hc = 1 if "MILCO" in cls_show or not cls_show else 2; hm, cx, cy = heatmap(scan, hc)
        r.plotly_chart(px.imshow(hm, x=cx, y=cy, color_continuous_scale="Hot", zmin=0, zmax=1, title=f"{C3[hc]} probability heat-map", aspect="equal"), width="stretch")
        c = st.columns(3); c[0].metric("Detections", len(dets)); c[1].metric("Ground-truth objects", len(gt)); c[2].metric("Scan time", f"{el:.2f} s")

# ============ 7. TRAINING & EXPORT ============
with tabs[6]:
    cv = res["curves"]; st.subheader("Training curves")
    l, r = st.columns(2)
    d = cv["denoiser"]; f = go.Figure(); f.add_scatter(x=d["epoch"], y=d["train_loss"], name="train loss", yaxis="y"); f.add_scatter(x=d["epoch"], y=d["val_psnr"], name="val PSNR (dB)", yaxis="y2")
    f.update_layout(title="Deconvolutional denoiser", xaxis_title="epoch", yaxis=dict(title="loss"), yaxis2=dict(title="PSNR dB", overlaying="y", side="right")); l.plotly_chart(f, width="stretch")
    f = go.Figure()
    for k in ("CNN (raw)", "CNN (noise-aug)", "Denoise + CNN"):
        f.add_scatter(x=cv[k]["epoch"], y=cv[k]["val_bal_acc"], name=k)
    f.update_layout(title="CNN validation balanced accuracy", xaxis_title="epoch"); r.plotly_chart(f, width="stretch")
    f = go.Figure()
    for k in ("CNN (raw)", "CNN (noise-aug)", "Denoise + CNN"):
        f.add_scatter(x=cv[k]["epoch"], y=cv[k]["train_loss"], name=f"{k} train"); f.add_scatter(x=cv[k]["epoch"], y=cv[k]["val_loss"], name=f"{k} val", line=dict(dash="dot"))
    f.update_layout(title="CNN loss", xaxis_title="epoch"); plot(f)
    st.info(f"Denoiser: {res['params']['denoiser']:,} params · CNN: {res['params']['cnn']:,} params · trained on CPU in {res['meta'].get('total_time_s', 0) / 60:.1f} min")
    st.subheader("📦 Export")
    zb = io.BytesIO()
    with zipfile.ZipFile(zb, "w", zipfile.ZIP_DEFLATED) as z:
        for fn in ["denoiser.pt", "cnn_raw.pt", "cnn_aug.pt", "cnn_dn.pt", "classical.joblib", "results.json"]:
            z.write(ART / fn, fn)
        z.write(Path(__file__).parent / "sonar" / "models.py", "models.py")
    c = st.columns(3)
    c[0].download_button("⬇️ Trained models (.zip)", zb.getvalue(), "prj44_models.zip", help="PyTorch state_dicts + classical models + models.py (architecture)")
    if c[1].button("📄 Build PDF report"):
        with st.spinner("Rendering PDF..."):
            st.session_state["pdf"] = build_pdf(res)
    if "pdf" in st.session_state:
        c[2].download_button("⬇️ Download PDF report", st.session_state["pdf"], "prj44_report.pdf", mime="application/pdf")
    st.markdown("**CSV exports**")
    d = st.columns(5)
    for col, (nm, df) in zip(d, [("ablation", ablation_df(res)), ("comparison", comparison_df(res)), ("per_class", perclass_df(res)), ("denoise", denoise_df(res)), ("detection", detection_df(res))]):
        col.download_button(f"⬇️ {nm}.csv", df.to_csv(index=False), f"{nm}.csv")
    with st.expander("Load a saved model in your own code"):
        st.code("import torch\nfrom models import DenoiseNet, SonarCNN\ndn = DenoiseNet(); dn.load_state_dict(torch.load('denoiser.pt')); dn.eval()\n"
                "cnn = SonarCNN(); cnn.load_state_dict(torch.load('cnn_dn.pt')); cnn.eval()\n# x: float tensor (N,1,48,48) in [0,1]\nlogits = cnn(dn(x).clamp(0, 1))", language="python")
    if not READONLY and D.list_images(root) and st.button("🔁 Re-train (delete artifacts)"):
        for f_ in ART.glob("*"):
            f_.unlink()
        load_bundle.clear(); st.rerun()

# ============ 8. ABOUT ============
with tabs[7]:
    st.markdown(f"""
## About / Methodology
**Problem.** Side-scan sonar images are corrupted by multiplicative *speckle*. This project tests whether a **deconvolutional denoising CNN** improves
automatic recognition of mine-like contacts (MILCO) vs non-mine bottom objects (NOMBO) on real AUV data.

**Data.** {res['meta']['n_images']} real Gavia AUV side-scan images with YOLO boxes (Pessanha Santos et al., *Data in Brief* 2024, CC BY 4.0). Patches are cut at 96 px around
each annotated object and around random background locations, then area-averaged to 48×48. **Split by image** into train/val/test (60/15/25%), so no leakage.

**Denoiser (deconvolutional).** U-Net-style encoder-decoder; decoder uses `ConvTranspose2d` (transposed conv / "deconvolution", cf. Noh et al. 2015) with skip connections and a residual output.
Trained with L1+MSE on synthetic gamma speckle (L ∈ {{1,2,4,8,16}}) + Gaussian noise. Two modes: **n2c** (noisy → pseudo-clean) and **n2n** (Noise2Noise: two independent noisy versions, Lehtinen et al. 2018).
A fraction of steps uses *real* patches as input so the network also sees genuine sonar noise.

**Pseudo-clean reference (important caveat).** No real clean sonar exists. The reference is a median(3)+Gaussian(σ=1) smoothed patch, so SNR/PSNR/SSIM measure agreement with a *smoothed proxy*,
not the physical truth. We therefore also report a Lee-filter baseline (a standard despeckle filter, deliberately different from the reference) and no-reference metrics (ENL, CNR) on real patches.

**Metrics.** SNR = 10·log10(Σref² / Σ(x−ref)²); SNR improvement = SNR(after) − SNR(before); PSNR (peak = 1); SSIM (7×7 window). ENL = mean²/variance on background patches. CNR = |μ_obj − μ_ring| / σ_ring.

**Classifiers.** (1) 5 classical models on 67 hand-made features (best chosen on the *validation* set); (2) CNN on raw patches; (3) CNN with noise augmentation (control);
(4) **Denoise + CNN** (frozen denoiser → CNN, same noise exposure as 3). Class-weighted loss, flip augmentation (port/starboard symmetry).

**Statistics.** 95% CIs by image-level cluster bootstrap; McNemar exact test for paired model comparison; robustness by corrupting test patches with extra speckle.

**Explainability.** Grad-CAM on the last conv block. **Detection.** Multi-scale CNN heat-maps → soft-argmax centre refinement → per-class NMS (IoU 0.3) → anchor-sized boxes (median training-box size); mAP@0.5 with VOC all-point AP against YOLO ground truth, compared with the classical sliding-window baseline.

**Limitations (be upfront in your report).** Pseudo-clean reference; small dataset (results have wide CIs); no learned box regression limits IoU; single split (run several seeds for a paper); the denoiser may not help on already-clean real data and mainly pays off under stronger noise. Whatever the tables show is the result.

**References.** Pessanha Santos et al. 2024 (dataset) · Noh, Hong, Han, ICCV 2015 (deconvolution network) · Lehtinen et al., ICML 2018 (Noise2Noise) · Selvaraju et al., ICCV 2017 (Grad-CAM).
""")
st.divider(); st.caption("PyTorch (CPU) + scikit-learn + Streamlit · Data: Pessanha Santos et al. (2024), Data in Brief 53:110132")
