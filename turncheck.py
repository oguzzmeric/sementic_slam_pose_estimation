"""
Yon hatasi ile GT donus hizi (turn rate) arasindaki korelasyonu olcer.

Hipotez: yon hatasi rastgele degil, drone donerken (translation kucuk,
rotation buyukken) H/E ayristirmasi sistematik olarak bozuluyor.

Kod degisikligi gerektirmez -- mevcut data/trajectory_output.csv ve
data/ground-truth.csv'den hesaplanir.
"""

import csv
import numpy as np

TRAJ = "data/trajectory_output.csv"
GT = "data/ground-truth.csv"


def load():
    gt = {}
    with open(GT, encoding="utf-8") as f:
        for r in csv.DictReader(f):
            z = r.get("translation_z", "") or 0.0
            gt[r["frame_numbers"].strip()] = (
                float(r["translation_x"]), float(r["translation_y"]), float(z))

    rows = []
    with open(TRAJ, encoding="utf-8") as f:
        for r in csv.DictReader(f):
            if r["is_valid"].strip().lower() != "true":
                continue
            stem = r["frame_name"].rsplit(".", 1)[0]
            if stem not in gt:
                continue
            rows.append((stem, (float(r["x"]), float(r["y"]), float(r["z"])), gt[stem]))
    rows.sort()
    return rows


def umeyama(P, Q):
    P = P.T; Q = Q.T
    n = P.shape[1]
    mu_p = P.mean(axis=1, keepdims=True)
    mu_q = Q.mean(axis=1, keepdims=True)
    Pc = P - mu_p; Qc = Q - mu_q
    W = Qc @ Pc.T / n
    U, D, Vt = np.linalg.svd(W)
    S = np.eye(3)
    if np.linalg.det(U) * np.linalg.det(Vt) < 0:
        S[2, 2] = -1
    R = U @ S @ Vt
    var_p = (Pc ** 2).sum() / n
    s = np.trace(np.diag(D) @ S) / var_p if var_p > 1e-12 else 1.0
    t = mu_q - s * R @ mu_p
    return s, R, t


rows = load()
names = [r[0] for r in rows]
est = np.array([r[1] for r in rows])
gt = np.array([r[2] for r in rows])

s, R, t = umeyama(est, gt)
est_al = (s * R @ est.T + t).T

# --- GT donus hizi: ardisik GT adim vektorleri arasindaki aci ---
d_gt = np.diff(gt, axis=0)
sg = np.linalg.norm(d_gt, axis=1)
m_valid = sg > 1e-6

turn_rate = np.full(len(d_gt), np.nan)
for i in range(1, len(d_gt)):
    if sg[i] < 1e-6 or sg[i - 1] < 1e-6:
        continue
    cosang = np.dot(d_gt[i], d_gt[i - 1]) / (sg[i] * sg[i - 1])
    cosang = np.clip(cosang, -1.0, 1.0)
    turn_rate[i] = np.degrees(np.arccos(cosang))

# --- kare basi yon hatasi (hizalanmis) ---
d_est = np.diff(est_al, axis=0)
d_gt2 = np.diff(gt, axis=0)
se = np.linalg.norm(d_est, axis=1)
m_ok = (se > 1e-6) & (sg > 1e-6)
h_est = np.arctan2(d_est[:, 1], d_est[:, 0])
h_gt = np.arctan2(d_gt2[:, 1], d_gt2[:, 0])
hd = np.degrees((h_gt - h_est + np.pi) % (2 * np.pi) - np.pi)
hd_abs = np.abs(hd)

# turn_rate[i] frame i-1 -> i -> i+1 arasindaki donus; hd[i] de ayni kare
# icin adim yon hatasi (est_al[i]-est_al[i-1] vs gt[i]-gt[i-1]).
# turn_rate index kaymis (turn_rate[i] iki adimlik aciyi, hd[i] tek adimi
# olcuyor) -- turn_rate[1:] ile hd[1:] hizalanir (her ikisi de "i" nokta
# etrafinda tanimli).
n = min(len(turn_rate), len(hd)) - 1
tr = turn_rate[1:n + 1]
hda = hd_abs[1:n + 1]
valid = ~np.isnan(tr)
tr = tr[valid]; hda = hda[valid]
names_v = np.array(names[2:n + 2])[valid]

print("=" * 66)
print(f"  n={len(tr)}  korelasyon(turn_rate, |yon_hatasi|) = "
      f"{np.corrcoef(tr, hda)[0, 1]:+.4f}")
print("=" * 66)

# --- dusuk vs yuksek turn-rate karsilastirmasi ---
med_tr = np.median(tr)
low = hda[tr <= med_tr]
high = hda[tr > med_tr]
print()
print(f"  turn_rate medyan (esik)      : {med_tr:.2f} deg")
print(f"  dusuk turn-rate  |yon_hata|  : medyan {np.median(low):6.2f}  n={len(low)}")
print(f"  yuksek turn-rate |yon_hata|  : medyan {np.median(high):6.2f}  n={len(high)}")
print()

# --- turn-rate'e gore 10 dilim profili ---
order = np.argsort(tr)
tr_s = tr[order]; hda_s = hda[order]
print("TURN-RATE DILIM PROFILI (dusukten yukseğe siralanmis)")
print(f"{'dilim':>6} {'turn_rate':>10} {'yon_hata_medyan':>16} {'n':>5}")
n_s = len(tr_s); step = max(1, n_s // 10)
for i in range(0, n_s, step):
    sl = slice(i, min(i + step, n_s))
    print(f"{i:6d} {np.median(tr_s[sl]):10.2f} {np.median(hda_s[sl]):16.2f} {len(tr_s[sl]):5d}")

print()
print("EN YUKSEK TURN-RATE 10 KARE")
worst_tr = np.argsort(tr)[-10:][::-1]
for i in worst_tr:
    print(f"  {names_v[i]:20s} turn_rate={tr[i]:6.2f} deg   yon_hata={hda[i]:6.2f} deg")
