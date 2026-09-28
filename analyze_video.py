"""
analyze_video.py -- extract the foam statistics that bubble_sim.py reproduces.

Run it on the source clip to see where the model's numbers come from, and on a
rendered simulation to check how close the simulation lands:

    python analyze_video.py bubble.mp4
    python analyze_video.py bubble.mp4 foam_sim.mp4        # side by side
    python analyze_video.py bubble.mp4 --frames=600        # use more frames

Measured quantities
    scale       characteristic cell diameter vs image row, from the radial
                power spectrum of square patches (robust to the fact that
                individual bubbles in a jammed foam are hard to segment)
    anisotropy  structure period along x vs y -> the perspective squash b/a
    size        log-normal fit to LoG blob radii, per depth band
    motion      Farneback optical flow: instantaneous speed vs time-averaged
                drift, which separates turbulent jitter from bulk current
    lifetime    intensity autocorrelation vs lag, and a nearest-neighbour
                tracker giving the survival curve
    appearance  intensity percentiles and the mean radial profile of a bubble

Requires numpy, opencv-python, scikit-image.
"""

from __future__ import annotations

import sys

import cv2
import numpy as np


# ---------------------------------------------------------------------------

def load(path, max_frames=300):
    cap = cv2.VideoCapture(path)
    fps = cap.get(cv2.CAP_PROP_FPS) or 30.0
    col, grey = [], []
    while len(grey) < max_frames:
        ok, f = cap.read()
        if not ok:
            break
        col.append(f)
        grey.append(cv2.cvtColor(f, cv2.COLOR_BGR2GRAY))
    cap.release()
    if not grey:
        raise SystemExit(f"could not read frames from {path}")
    return np.array(col), np.array(grey), fps


def active_rows(F):
    """Rows that actually contain moving foam, so two clips of different
    framing are compared over comparable regions."""
    act = F.astype(np.float32).std(axis=0).mean(axis=1)
    thr = 0.45 * act.max()
    rows = np.where(act > thr)[0]
    return rows.min(), rows.max()


# --- 1. characteristic cell size from the radial power spectrum ------------

def char_scale(patch):
    p = patch.astype(np.float32)
    p -= p.mean()
    win = np.outer(np.hanning(p.shape[0]), np.hanning(p.shape[1]))
    P = np.abs(np.fft.fftshift(np.fft.fft2(p * win))) ** 2
    cy, cx = np.array(P.shape) // 2
    yy, xx = np.mgrid[0:P.shape[0], 0:P.shape[1]]
    rr = np.hypot(yy - cy, xx - cx).astype(int)
    radial = np.bincount(rr.ravel(), P.ravel()) / np.maximum(
        np.bincount(rr.ravel()), 1)
    k = np.arange(len(radial))
    lam = np.where(k > 0, p.shape[0] / np.maximum(k, 1e-9), 1e9)
    m = (lam > 4) & (lam < 90)
    best = k[m][np.argmax(radial[m] * k[m])]
    return p.shape[0] / best


def report_scale(F, y0, y1, size=128):
    print("\n  cell diameter vs image row (radial power spectrum)")
    for yy in range(y0, y1 - size, max(1, (y1 - y0 - size) // 7)):
        vals = [char_scale(F[fi, yy:yy + size, x:x + size])
                for x in (60, 170, 280)
                for fi in range(20, len(F) - 1, max(1, len(F) // 12))
                if F[fi, yy:yy + size, x:x + size].shape == (size, size)]
        if vals:
            print(f"    row {yy:4d}   {np.median(vals):5.1f} px")


# --- 2. anisotropy ---------------------------------------------------------

def period(lines):
    acs = []
    for s in lines:
        s = s - s.mean()
        ac = np.correlate(s, s, "full")[len(s) - 1:]
        acs.append(ac / (ac[0] + 1e-9))
    ac = np.mean(acs, axis=0)
    d = np.diff(ac)
    i = 1
    while i < len(d) - 1 and not (d[i] > 0 and d[i - 1] <= 0):
        i += 1
    j = i + 1
    while j < len(d) - 1 and not (d[j] < 0 and d[j - 1] >= 0):
        j += 1
    return float(j)


def report_anisotropy(F, y0, y1):
    print("\n  structure period along x vs y (perspective squash)")
    for yy in np.linspace(y0 + 80, y1 - 80, 4).astype(int):
        xs = [F[fi, yy:yy + 40:4, 40:440] for fi in range(40, len(F), 90)]
        ys = [F[fi, yy - 60:yy + 60, 60:420:8].T for fi in range(40, len(F), 90)]
        px = np.median([period(a) for a in xs])
        py = np.median([period(a) for a in ys])
        print(f"    row {yy:4d}   x {px:5.1f} px   y {py:5.1f} px   b/a {py / px:.2f}")


# --- 3. size distribution --------------------------------------------------

def report_sizes(F, y0, y1):
    from skimage.feature import blob_log
    print("\n  log-normal size fit per depth band (LoG blobs)")
    edges = np.linspace(y0, y1, 5).astype(int)
    for a, b in zip(edges[:-1], edges[1:]):
        R = []
        for fi in range(30, len(F), max(1, len(F) // 10)):
            sub = F[fi, a:b, 20:460].astype(np.float32)
            im = (sub - sub.mean()) / (sub.std() + 1e-6)
            blobs = blob_log(-im, min_sigma=1.4, max_sigma=11.0, num_sigma=6,
                             threshold=0.06, overlap=0.3)
            if len(blobs):
                R += list(blobs[:, 2] * np.sqrt(2) * 1.9)
        if len(R) > 50:
            R = np.array(R)
            print(f"    rows {a:4d}-{b:4d}   median r {np.median(R):5.2f} px   "
                  f"sigma(ln r) {np.std(np.log(R)):.3f}")


# --- 4. motion -------------------------------------------------------------

def report_motion(F, y0, y1, fps):
    print("\n  motion")
    acc = np.zeros(F.shape[1:] + (2,), np.float32)
    inst, n = [], 0
    for i in range(20, len(F) - 1, 4):
        fl = cv2.calcOpticalFlowFarneback(F[i], F[i + 1], None,
                                          0.5, 3, 21, 3, 5, 1.2, 0)
        acc += fl
        n += 1
        inst.append(np.linalg.norm(fl[y0:y1], axis=2).mean())
    drift = np.linalg.norm(acc[y0:y1] / n, axis=2).mean()
    print(f"    instantaneous  {np.mean(inst):.2f} px/frame  "
          f"({np.mean(inst) * fps:.0f} px/s)")
    print(f"    mean drift     {drift:.2f} px/frame  "
          f"-> {100 * (1 - drift / np.mean(inst)):.0f}% of the motion is jitter")


# --- 5. lifetime -----------------------------------------------------------

def report_lifetime(F, y0, y1, fps):
    print("\n  temporal autocorrelation of intensity (lag in frames)")
    reg = F[:, y0:y1, 80:400].astype(np.float32)
    reg -= reg.mean(axis=0, keepdims=True)
    reg /= reg.std(axis=0, keepdims=True) + 1e-6
    out = []
    for lag in (1, 2, 3, 5, 8, 12, 20, 30):
        out.append(f"{lag}:{(reg[:-lag] * reg[lag:]).mean():.2f}")
    print("    " + "  ".join(out))

    from skimage.feature import blob_log
    lifes = []
    for start in range(30, len(F) - 26, 90):
        seq = []
        for k in range(22):
            sub = F[start + k, y0:y1, 40:440].astype(np.float32)
            im = (sub - sub.mean()) / (sub.std() + 1e-6)
            b = blob_log(-im, min_sigma=2.5, max_sigma=10.0, num_sigma=5,
                         threshold=0.08, overlap=0.2)
            seq.append(b[:, :2] if len(b) else np.zeros((0, 2)))
        active, nxt, lens = {}, 0, {}
        for b in seq:
            new = {}
            used = set()
            for y, x in b:
                best, bd = None, 1e9
                for tid, (py, px) in active.items():
                    d = np.hypot(y - py, x - px)
                    if d < bd and d < 9.0 and tid not in used:
                        bd, best = d, tid
                if best is None:
                    best = nxt
                    nxt += 1
                    lens[best] = 0
                else:
                    used.add(best)
                lens[best] += 1
                new[best] = (y, x)
            active = new
        lifes += list(lens.values())
    lifes = np.array(lifes)
    surv = {k: round(float((lifes >= k).mean()), 3) for k in (1, 2, 3, 5, 8, 12)}
    print(f"\n  tracked lifetime: mean {lifes.mean() / fps:.3f} s   "
          f"p90 {np.percentile(lifes, 90) / fps:.3f} s")
    print(f"    survival {surv}")


# --- 6. appearance ---------------------------------------------------------

def report_appearance(C, F, y0, y1):
    print("\n  appearance")
    g = F[::10, y0:y1, 40:440].ravel()
    p = np.percentile(g, [1, 10, 50, 90, 99])
    print(f"    intensity p1 {p[0]:.0f}  p10 {p[1]:.0f}  median {p[2]:.0f}  "
          f"p90 {p[3]:.0f}  p99 {p[4]:.0f}  std {g.std():.0f}")
    b, gg, r = [C[::20, y0:y1, 40:440, i].mean() for i in range(3)]
    print(f"    mean BGR ({b:.0f}, {gg:.0f}, {r:.0f})")

    from skimage.feature import blob_log
    prof, cnt = np.zeros(20), 0
    for fi in range(30, len(F), max(1, len(F) // 6)):
        sub = F[fi, max(y0, y1 - 220):y1, 60:420].astype(np.float32)
        im = (sub - sub.mean()) / (sub.std() + 1e-6)
        for y, x, s in blob_log(-im, min_sigma=3, max_sigma=9, num_sigma=5,
                                threshold=0.10, overlap=0.1):
            R = s * np.sqrt(2) * 1.7
            if not 5 < R < 18:
                continue
            k = int(1.6 * R)
            yy, xx = np.mgrid[-k:k + 1, -k:k + 1]
            Y, X = int(y) + yy, int(x) + xx
            if Y.min() < 0 or X.min() < 0 or Y.max() >= sub.shape[0] \
                    or X.max() >= sub.shape[1]:
                continue
            d = np.hypot(yy, xx) / R
            h, _ = np.histogram(d, bins=20, range=(0, 1.6), weights=sub[Y, X])
            c, _ = np.histogram(d, bins=20, range=(0, 1.6))
            prof += h / np.maximum(c, 1)
            cnt += 1
    if cnt:
        prof /= cnt
        print(f"    mean radial profile over {cnt} bubbles "
              f"(u = r/R, the rim is u = 1)")
        for i in range(0, 20, 2):
            u = (i + 0.5) * 1.6 / 20
            print(f"      u {u:4.2f}  I {prof[i]:5.1f}  "
                  + "#" * max(0, int((prof[i] - 85) / 2.5)))


# ---------------------------------------------------------------------------

def analyse(path, max_frames=300):
    print("=" * 66)
    print(path)
    print("=" * 66)
    C, F, fps = load(path, max_frames)
    y0, y1 = active_rows(F)
    print(f"  {len(F)} frames, {F.shape[2]}x{F.shape[1]} @ {fps:.0f} fps, "
          f"foam occupies rows {y0}-{y1}")
    report_scale(F, y0, y1)
    report_anisotropy(F, y0, y1)
    report_sizes(F, y0, y1)
    report_motion(F, y0, y1, fps)
    report_lifetime(F, y0, y1, fps)
    report_appearance(C, F, y0, y1)


if __name__ == "__main__":
    if len(sys.argv) < 2:
        raise SystemExit(__doc__)
    paths = [a for a in sys.argv[1:] if not a.startswith("-")]
    nf = 300
    for a in sys.argv[1:]:
        if a.startswith("--frames="):
            nf = int(a.split("=")[1])
    for p in paths:
        analyse(p, nf)
