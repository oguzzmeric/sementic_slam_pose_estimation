"""
Hata profilini kare bazinda cikarir.

Mevcut trajectory_output.csv ve ground-truth.csv'den calisir,
pipeline calistirmaya gerek yok.

Umeyama (Sim3) hizalamasi uygulanip kare bazinda hata hesaplanir,
ardindan hatanin nerede patladigi ve o bolgede ne oldugu incelenir.
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
            rows.append((
                stem,
                (float(r["x"]), float(r["y"]), float(r["z"])),
                gt[stem],
                float(r["scale"]),
                r["mode"].strip(),
            ))
    rows.sort()
    return rows


def umeyama(P, Q, with_scale=True):
    """P -> Q hizalamasi. Dondurur: s, R, t"""
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
    if with_scale:
        var_p = (Pc ** 2).sum() / n
        s = np.trace(np.diag(D) @ S) / var_p if var_p > 1e-12 else 1.0
    else:
        s = 1.0
    t = mu_q - s * R @ mu_p
    return s, R, t


rows = load()
names = [r[0] for r in rows]
est = np.array([r[1] for r in rows])
gt = np.array([r[2] for r in rows])
scales = np.array([r[3] for r in rows])
modes = [r[4] for r in rows]

s, R, t = umeyama(est, gt, with_scale=True)
est_al = (s * R @ est.T + t).T
err = np.linalg.norm(gt - est_al, axis=1)

n = len(err)
print("=" * 62)
print(f"  {n} kare  |  olcek {s:.4f}  |  RMSE {np.sqrt((err**2).mean()):.2f} m")
print("=" * 62)

# --- hata profili, 12 dilim ---
print()
print("HATA PROFILI")
print(f"{'kare':>7} {'medyan':>9} {'maks':>9} {'adim':>8} {'GT adim':>9} {'mod'}")
k = max(1, n // 12)
for i in range(0, n, k):
    sl = slice(i, min(i + k, n))
    e = err[sl]
    d_est = np.linalg.norm(np.diff(est_al[sl], axis=0), axis=1)
    d_gt = np.linalg.norm(np.diff(gt[sl], axis=0), axis=1)
    mode_counts = {}
    for m in modes[sl]:
        mode_counts[m] = mode_counts.get(m, 0) + 1
    dom = max(mode_counts, key=mode_counts.get) if mode_counts else "-"
    print(f"{i:7d} {np.median(e):9.1f} {e.max():9.1f} "
          f"{np.median(d_est) if len(d_est) else 0:8.2f} "
          f"{np.median(d_gt) if len(d_gt) else 0:9.2f}  {dom}")

# --- en kotu bolge ---
print()
print("EN KOTU 5 KARE")
worst = np.argsort(err)[-5:][::-1]
for i in worst:
    print(f"  {names[i]:22s} hata={err[i]:7.1f} m  olcek={scales[i]:.4f}  mod={modes[i]}")

# --- hata artis hizi ---
print()
print("HATA ARTIS HIZI (ardisik kareler arasi)")
de = np.diff(err)
print(f"  medyan artis : {np.median(de):+7.3f} m/kare")
print(f"  maks artis   : {de.max():+7.3f} m/kare  (kare {names[int(np.argmax(de))+1]})")
print(f"  maks dusus   : {de.min():+7.3f} m/kare  (kare {names[int(np.argmin(de))+1]})")

# --- sicrama noktalari ---
th = np.percentile(np.abs(de), 97)
jumps = np.where(np.abs(de) > th)[0]
print()
print(f"SICRAMA NOKTALARI  (|delta| > {th:.2f} m)")
for j in jumps[:12]:
    print(f"  {names[j+1]:22s} delta={de[j]:+7.2f} m  "
          f"hata {err[j]:6.1f} -> {err[j+1]:6.1f}  olcek={scales[j+1]:.4f}")

# --- yon hatasi ile korelasyon ---
d_est = np.diff(est_al, axis=0)
d_gt = np.diff(gt, axis=0)
m_ok = (np.linalg.norm(d_est, axis=1) > 1e-6) & (np.linalg.norm(d_gt, axis=1) > 1e-6)
h_est = np.arctan2(d_est[m_ok, 1], d_est[m_ok, 0])
h_gt = np.arctan2(d_gt[m_ok, 1], d_gt[m_ok, 0])
hd = np.degrees((h_gt - h_est + np.pi) % (2 * np.pi) - np.pi)

print()
print("YON HATASI (hizalama sonrasi)")
print(f"  medyan : {np.median(hd):+7.1f} deg")
print(f"  std    : {np.std(hd):7.1f} deg")
print(f"  |fark| > 45 deg olan kare orani : {100*np.mean(np.abs(hd) > 45):.1f}%")

# --- adim uzunlugu karsilastirmasi ---
se = np.linalg.norm(d_est, axis=1)
sg = np.linalg.norm(d_gt, axis=1)
print()
print("ADIM UZUNLUGU")
print(f"  tahmin medyan : {np.median(se):.3f} m")
print(f"  GT medyan     : {np.median(sg):.3f} m")
print(f"  oran          : {np.median(sg)/max(np.median(se),1e-9):.3f}")