# Boiling Foam Simulation

A physically-motivated simulation of the foam on a boiling liquid surface, in which every operator and every constant is fitted to statistics measured from a single handheld video.

The methodology takes the opposite route from hand-tuned particle systems: first extract target statistics from footage, then construct operators pinned to those statistics, then re-run the identical measurement pipeline on the generated output to verify the match.

## How it works

The simulation maintains a population of ~3,000 surface cells under seven operators per step:

1. **Nucleation** — Poisson births over the pan base, 40% concentrated over the burner; birth radius scales with local superheat so coarse foam only appears near-side
2. **Advection** — convection-roll surface flow plus Ornstein–Uhlenbeck turbulent jitter, tuned to the measured 0.41/1.43 px-per-frame drift/jitter split
3. **Packing** — position-based contact relaxation against the 7 nearest neighbours
4. **Condensation** — `dr/dt = −c` stalling at `r_eq`; this is what narrows the log-radius spread from σ = 0.65 (near) to 0.40 (far)
5. **Coalescence** — volume-conserving merges, rate climbing with film age
6. **Bursting** — hazard `h ∝ r³·² (1 + churn · age)`, with a churn term over the burner that prevents persistent rafts from building there
7. **Oscillation** — Rayleigh mode-2 capillary wobble with viscous damping (undamped it aliases badly at 30 fps)

A pinhole camera projects each cell to its exact first-order ellipse via the local Jacobian, so perspective magnification and the measured `b/a` squash fall out of the geometry rather than being faked.

### Why not coalescence or advection?

Three candidate mechanisms can produce the observed near/far size gradient; two are falsified by the measurements:

- **Coalescence** — predicts old foam is *coarser*; the footage shows it is *finer* and more monodisperse. Ruled out.
- **Advection** — requires a surface current of 0.08 m/s; the measured drift is 0.011 m/s. Ruled out.
- **Local terms** — birth size scales with substrate superheat; a churn term over the burner destroys the raft there. No transport needed. This survives.

## Installation

```bash
python -m venv env
source env/bin/activate
pip install -r req.txt
```

For saving MP4 output, `ffmpeg` must be on your PATH:

```bash
# macOS
brew install ffmpeg
# Debian/Ubuntu
sudo apt install ffmpeg
# conda
conda install -c conda-forge ffmpeg
```

On a headless Linux box with no GUI backend, use `--save` to write a file instead of opening a window; or `pip install PyQt5` for a live window inside a venv.

## Usage

### Simulation

```bash
# Live animation window (default 10 s)
python bubble_sim.py

# Save to MP4
python bubble_sim.py --save out/foam.mp4 --seconds 15

# Save to GIF
python bubble_sim.py --save out/foam.gif --seconds 6 --fps 20

# Adjust boil intensity (0.5 = simmer, 1.0 = as filmed, 2.0 = furious)
python bubble_sim.py --vigour 1.6 --seed 7 --live

# Higher resolution
python bubble_sim.py --scale 1.5 --save out/foam_hd.mp4
```

A HUD overlay shows live statistics: bubble count, median/p95 radius, σ(ln r), and pops per frame.

### Video analysis

```bash
# Measure the source clip
python analyze_video.py in/bubble.mp4

# Compare source and simulation side-by-side
python analyze_video.py in/bubble.mp4 out/foam_sim.mp4

# Use more frames for better statistics
python analyze_video.py in/bubble.mp4 --frames=600
```

The analysis pipeline reports: characteristic cell diameter vs image row, perspective squash `b/a`, log-normal size fit per depth band, motion (instantaneous speed vs time-averaged drift), temporal autocorrelation and tracked lifetime, and intensity percentiles with mean radial profile.

## Files

```
bubble_sim.py      full simulation: 7-operator physics, pinhole camera, software renderer
analyze_video.py   measurement pipeline; run on source or generated footage
main.py            early prototype (simple scatter-plot animation, 300 bubbles)
req.txt            pip dependencies
in/bubble.mp4      source clip (478×850, 30 fps, 23.5 s)
out/               rendered outputs
paper/             write-up of the methodology and results
```

## Simulation results

The generated sequence, measured by the unmodified analysis pipeline, matches the source on:

| Metric | Source | Simulation |
|--------|--------|------------|
| Cell diameter range | 9.1–42.7 px | 8.7–36.9 px |
| Median intensity | 114 | 110 |
| Intensity std | 36 | 35 |
| 3-frame survival | 0.51 | 0.48 |

## Dependencies

| Package | Used for |
|---------|----------|
| `numpy >= 1.24` | all array computation |
| `scipy >= 1.10` | `cKDTree` (packing), `gaussian_filter` (lens blur) |
| `matplotlib >= 3.7` | animation and display |
| `opencv-python >= 4.10` | video decode, Farneback optical flow |
| `scikit-image >= 0.21` | `blob_log` blob detection |
