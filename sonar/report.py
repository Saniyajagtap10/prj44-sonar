"""PDF + CSV report export (reportlab + matplotlib)."""
import io

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from reportlab.lib import colors
from reportlab.lib.pagesizes import A4
from reportlab.lib.styles import getSampleStyleSheet
from reportlab.lib.units import cm
from reportlab.platypus import Image as RLImage, PageBreak, Paragraph, SimpleDocTemplate, Spacer, Table, TableStyle

from .pipeline import ALL_MODELS


def ablation_df(res):
    c = res["classification"]; bc = c["_best_classical"]
    rows = [(f"Classical ML ({bc})", bc), ("CNN, no denoising", "CNN (raw)"),
            ("CNN + noise augmentation (control)", "CNN (noise-aug)"), ("Deconv denoiser + CNN (proposed)", "Denoise + CNN")]
    out = []
    for lab, k in rows:
        r = c[k]; ci = r["ci"]
        out.append({"System": lab, "Accuracy (%)": 100 * r["acc"], "Balanced Acc (%)": 100 * r["bal_acc"],
                    "Bal. Acc 95% CI": f"{100 * ci['bal_acc'][0]:.1f}-{100 * ci['bal_acc'][1]:.1f}",
                    "Macro F1 (%)": 100 * r["macro_f1"], "MILCO Recall (%)": 100 * r["milco_recall"],
                    "MILCO Recall 95% CI": f"{100 * ci['milco_recall'][0]:.1f}-{100 * ci['milco_recall'][1]:.1f}"})
    return pd.DataFrame(out)


def comparison_df(res):
    c = res["classification"]; rows = []
    for k in ALL_MODELS:
        r = c[k]; rows.append({"Model": k, "CV Bal. Acc (%)": 100 * (r.get("cv_bal_acc") or float("nan")), "Accuracy (%)": 100 * r["acc"],
                               "Balanced Acc (%)": 100 * r["bal_acc"], "Macro F1 (%)": 100 * r["macro_f1"],
                               "MILCO Recall (%)": 100 * r["milco_recall"], "MILCO Precision (%)": 100 * r["milco_precision"]})
    return pd.DataFrame(rows).sort_values("Balanced Acc (%)", ascending=False).reset_index(drop=True)


def perclass_df(res):
    rows = []
    for k in ALL_MODELS:
        pc = res["classification"][k]["per_class"]
        for i, cn in enumerate(["Background", "MILCO", "NOMBO"]):
            rows.append({"Model": k, "Class": cn, "Precision": pc["precision"][i], "Recall": pc["recall"][i], "F1": pc["f1"][i], "Support": pc["support"][i]})
    return pd.DataFrame(rows)


def robustness_df(res):
    rows = []
    for lvl, d in res["robustness"].items():
        for m, v in d.items():
            rows.append({"Noise (looks L)": lvl, "System": m, "Balanced Acc (%)": 100 * v["bal_acc"], "MILCO Recall (%)": 100 * v["milco_recall"], "Macro F1 (%)": 100 * v["macro_f1"]})
    return pd.DataFrame(rows)


def denoise_df(res):
    return pd.DataFrame(res["denoise"]["by_noise"])


def detection_df(res):
    return pd.DataFrame(res["detection"]["methods"]).T.reset_index().rename(columns={"index": "Method"})


def _fig_png(fig):
    b = io.BytesIO(); fig.savefig(b, format="png", dpi=130, bbox_inches="tight"); plt.close(fig); b.seek(0); return b


def _tbl(df, fmt=2, font=7):
    data = [list(df.columns)] + [[(f"{v:.{fmt}f}" if isinstance(v, (float, np.floating)) else str(v)) for v in r] for r in df.values]
    t = Table(data, repeatRows=1)
    t.setStyle(TableStyle([("BACKGROUND", (0, 0), (-1, 0), colors.HexColor("#0b3c5d")), ("TEXTCOLOR", (0, 0), (-1, 0), colors.white),
                           ("FONTSIZE", (0, 0), (-1, -1), font), ("GRID", (0, 0), (-1, -1), .25, colors.grey),
                           ("ROWBACKGROUNDS", (0, 1), (-1, -1), [colors.white, colors.HexColor("#eef4f8")])]))
    return t


def build_pdf(res, title="PRJ-44 Report"):
    ss = getSampleStyleSheet(); buf = io.BytesIO(); doc = SimpleDocTemplate(buf, pagesize=A4, leftMargin=1.5 * cm, rightMargin=1.5 * cm, topMargin=1.5 * cm, bottomMargin=1.5 * cm)
    S = []; m = res["meta"]
    S += [Paragraph(title, ss["Title"]),
          Paragraph("Deconvolutional feature-denoising CNN for AUV side-scan sonar hazard detection (MILCO / NOMBO)", ss["Heading3"]),
          Paragraph(f"Images: {m['n_images']} | patches train/val/test: {m['patches']['train']}/{m['patches']['val']}/{m['patches']['test']} | "
                    f"split: image-wise (no leakage) | device: CPU | denoiser params: {res['params']['denoiser']:,} | CNN params: {res['params']['cnn']:,}", ss["BodyText"]),
          Spacer(1, 8), Paragraph("1. Ablation: does denoising help?", ss["Heading2"]), _tbl(ablation_df(res), 1),
          Paragraph("95% CIs come from an image-level (cluster) bootstrap. Metrics are on held-out test images.", ss["Italic"]), Spacer(1, 6)]
    mc = pd.DataFrame(res["mcnemar"]); mc["p_value"] = mc["p_value"].map(lambda v: f"{v:.4f}")
    S += [Paragraph("McNemar exact test (paired, test patches; A_only = A right & B wrong)", ss["Heading4"]), _tbl(mc, 3, 8)]
    S += [Paragraph("2. Robustness to extra speckle", ss["Heading2"])]
    r = robustness_df(res); fig, ax = plt.subplots(figsize=(6, 3))
    for s_, g in r.groupby("System"):
        ax.plot(g["Noise (looks L)"], g["Balanced Acc (%)"], marker="o", label=s_)
    ax.set_xlabel("speckle looks L (right = noisier)"); ax.set_ylabel("Balanced acc (%)"); ax.legend(fontsize=6); ax.grid(alpha=.3)
    S += [RLImage(_fig_png(fig), width=13 * cm, height=6.5 * cm), PageBreak(), Paragraph("3. Denoising quality (vs pseudo-clean reference)", ss["Heading2"])]
    d = denoise_df(res)[["looks", "SNR_before", "SNR_after", "SNR_improvement", "SNR_improvement_filter", "PSNR_before", "PSNR_after", "SSIM_before", "SSIM_after"]]
    S += [_tbl(d, 2, 7), Paragraph("SNR_improvement_filter = Lee-filter baseline. No real clean ground truth exists; the reference is a smoothed proxy.", ss["Italic"]), Spacer(1, 6)]
    S += [Paragraph("4. Model comparison (test set)", ss["Heading2"]), _tbl(comparison_df(res), 1), Spacer(1, 6),
          Paragraph("5. Detection (mAP@0.5, YOLO-style boxes)", ss["Heading2"]), _tbl(detection_df(res), 3, 7)]
    c = res["curves"]; fig, ax = plt.subplots(1, 2, figsize=(9, 3))
    ax[0].plot(c["denoiser"]["epoch"], c["denoiser"]["val_psnr"]); ax[0].set_title("Denoiser val PSNR (dB)")
    for k in ("CNN (raw)", "CNN (noise-aug)", "Denoise + CNN"):
        ax[1].plot(c[k]["epoch"], c[k]["val_bal_acc"], label=k)
    ax[1].set_title("CNN val balanced acc"); ax[1].legend(fontsize=6)
    S += [Spacer(1, 6), Paragraph("6. Training curves", ss["Heading2"]), RLImage(_fig_png(fig), width=16 * cm, height=5.3 * cm)]
    doc.build(S); return buf.getvalue()
