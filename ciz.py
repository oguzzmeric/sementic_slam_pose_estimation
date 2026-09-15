"""
LinkedIn kapak gorseli.

Bbox tabanli irtifa kestirimi ile referans irtifayi karsilastirir.
Sol panel: NED donusumu oncesi -- negatif korelasyon.
Sag panel: donusum sonrasi -- y = x etrafinda toplanma.

Veri dogrudan detections.json ve ground-truth.csv'den uretilir,
pipeline calistirmaya gerek yok.

Kullanim:
    python ciz.py
"""

import csv
import json
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

# ----------------------------------------------------------------------
# ayarlar -- config.yaml ile ayni olmali
# ----------------------------------------------------------------------
FX, FY = 2680.51, 2683.39
CX, CY = 1908.48, 1139.71
Z_OFFSET = 41.6                      # kalkis irtifasi
MIN_CONF = 0.7
MIN_BBOX_PX = 12
REF_OBJECTS = {2: 1.5, 3: 4.5, 4: 7.0, 5: 12.0}

GT_PATH = "data/ground-truth.csv"
DET_PATH = "data/detections.json"
OUT_PATH = "linkedin_kapak.png"

NAVY = "#152238"; ACC = "#C9A227"; RED = "#B3413E"
GRN = "#3E7C5A"; GREY = "#6B7280"; LGREY = "#E3E7EB"

plt.rcParams["font.family"] = "DejaVu Sans"


# ----------------------------------------------------------------------
# veri toplama
# ----------------------------------------------------------------------
def collect():
    """Her kare icin (bbox_irtifa, ham_gt_z) ciftleri uretir."""
    gt = {}
    with open(GT_PATH, encoding="utf-8") as f:
        for r in csv.DictReader(f):
            z = r.get("translation_z", "") or 0.0
            gt[r["frame_numbers"].strip()] = float(z)

    with open(DET_PATH, encoding="utf-8") as f:
        dets = json.load(f)

    f_geo = np.sqrt(FX * FY)
    pairs = []

    for name, detections in dets.items():
        stem = name.rsplit(".", 1)[0]
        if stem not in gt:
            continue

        best = None
        for d in detections:
            cid = d["class_id"]
            if cid not in REF_OBJECTS or d["confidence"] < MIN_CONF:
                continue

            x1, y1, x2, y2 = d["bbox"]
            w, h = abs(x2 - x1), abs(y2 - y1)
            if w < MIN_BBOX_PX or h < MIN_BBOX_PX:
                continue

            d_pixel = max(w, h)
            uc, vc = (x1 + x2) / 2.0, (y1 + y2) / 2.0
            radial = np.hypot(uc - CX, vc - CY)
            cos2 = np.cos(np.arctan2(radial, f_geo)) ** 2

            z_est = (FY * REF_OBJECTS[cid]) / d_pixel * cos2
            if not (1.0 <= z_est <= 500.0):
                continue

            quality = d["confidence"] * cos2
            if best is None or quality > best[0]:
                best = (quality, z_est)

        if best is not None:
            pairs.append((best[1], gt[stem]))

    return np.array(pairs)


# ----------------------------------------------------------------------
# cizim
# ----------------------------------------------------------------------
def plot(bbox_z, gt_raw, gt_corr):
    rho_raw = float(np.corrcoef(bbox_z, gt_raw)[0, 1])
    ratio = float(np.median(bbox_z / np.maximum(gt_corr, 1e-9)))
    std = float(np.std(bbox_z / np.maximum(gt_corr, 1e-9)))
    n = len(bbox_z)

    fig, (a1, a2) = plt.subplots(1, 2, figsize=(11, 4.7))

    # --- sol: donusum oncesi ---
    a1.scatter(gt_raw, bbox_z, s=13, alpha=0.4, color=RED, edgecolors="none")
    a1.set_xlabel("Referans Z (ham, NED)", fontsize=9.5, color=GREY)
    a1.set_ylabel("Kestirilen irtifa (m)", fontsize=9.5, color=GREY)
    a1.set_title("Dönüşüm öncesi", fontsize=11.8, fontweight="bold",
                 color=NAVY, pad=11)
    a1.text(0.04, 0.92, f"ρ = {rho_raw:+.4f}", transform=a1.transAxes,
            fontsize=14, fontweight="bold", color=RED)
    a1.text(0.04, 0.83, "fiziksel olarak tutarsız", transform=a1.transAxes,
            fontsize=8.6, color=GREY, style="italic")
    a1.grid(alpha=0.2, ls="--", lw=0.6)

    # --- sag: donusum sonrasi ---
    a2.scatter(gt_corr, bbox_z, s=13, alpha=0.4, color=GRN, edgecolors="none")
    lo = min(gt_corr.min(), bbox_z.min())
    hi = max(gt_corr.max(), bbox_z.max())
    a2.plot([lo, hi], [lo, hi], color=NAVY, lw=1.5, ls="--", label="y = x")
    a2.set_xlabel("Referans irtifa (m)", fontsize=9.5, color=GREY)
    a2.set_ylabel("Kestirilen irtifa (m)", fontsize=9.5, color=GREY)
    a2.set_title("NED dönüşümü sonrası", fontsize=11.8, fontweight="bold",
                 color=NAVY, pad=11)
    a2.text(0.04, 0.92, f"oran {ratio:.3f}", transform=a2.transAxes,
            fontsize=14, fontweight="bold", color=GRN)
    a2.text(0.04, 0.83, f"σ = {std:.3f},  n = {n}", transform=a2.transAxes,
            fontsize=8.6, color=GREY, style="italic")
    a2.legend(fontsize=8.6, loc="lower right")
    a2.grid(alpha=0.2, ls="--", lw=0.6)

    for ax in (a1, a2):
        ax.tick_params(labelsize=8, colors=GREY)
        for s in ax.spines.values():
            s.set_color(LGREY)

    fig.suptitle("Monoküler irtifa kestirimi — koordinat çerçevesi doğrulaması",
                 fontsize=12.5, fontweight="bold", color=NAVY, y=1.02)

    plt.tight_layout()
    plt.savefig(OUT_PATH, dpi=200, bbox_inches="tight", facecolor="white")
    plt.close()

    print(f"kaydedildi : {OUT_PATH}")
    print(f"ornek      : {n}")
    print(f"korelasyon : {rho_raw:+.4f}  (ham)")
    print(f"oran       : {ratio:.4f}")
    print(f"std        : {std:.4f}")


if __name__ == "__main__":
    data = collect()
    if len(data) < 10:
        raise SystemExit(f"yetersiz ornek: {len(data)}")

    bbox_z = data[:, 0]
    gt_raw = data[:, 1]
    gt_corr = Z_OFFSET - gt_raw        # NED: asagi pozitif

    plot(bbox_z, gt_raw, gt_corr)