"""
bubble_sim.py -- a physically-motivated simulation of the boiling-foam pattern
in bubble.mp4, with a dynamic visualisation.

WHAT THE SOURCE VIDEO ACTUALLY SHOWS
------------------------------------
Not a single soap bubble: a pan of liquid at a rolling boil, filmed from above
at an oblique angle (478x850 @ 30 fps, 23.5 s).  The "pattern" is therefore the
statistics of a *bubble population* -- a jammed 2-D foam that is continuously
born, condensed, jostled and destroyed.

MEASURED FROM THE VIDEO  (analyze_video.py reproduces every number)
-------------------------------------------------------------------
  shape       bubbles project to ellipses with axis ratio b/a = 0.83 - 0.94
              (oblique camera); jammed cells flatten against their neighbours.
  surface     the radial intensity profile rises monotonically from a dark core
              to a bright rim peaking at u = r/R ~ 1.0, then decays outside.
              p1 51, median 114, p99 223 (8-bit), std 36; near-neutral grey,
              faintly warm (mean BGR 118/116/122).  Light from above-left.
  size        cell diameter 9.1 px at the far rim -> 42.7 px at the near rim.
              Sizes are log-normal, and the spread *narrows* with distance from
              the burner: sigma(ln r) 0.65 near -> 0.40 far.
  movement    instantaneous 1.43 px/frame (43 px/s); time-averaged drift only
              0.41 px/frame, so ~71% of the motion is chaotic jitter and the
              surface barely flows.
  lifetime    tracked mean 0.141 s, p90 0.333 s; survival 0.51 at 3 frames and
              0.31 at 5.  Tracked bubbles do not visibly grow -- they appear at
              near-final size and burst.
  renewal     17.5% of pixels change by >30 grey levels every frame.

FINDING THE RIGHT MECHANISM
---------------------------
Three candidate explanations for the near/far size gradient were tried; two are
ruled out by the measurements themselves, which is the most useful thing the
analysis did.

  coalescence   REJECTED.  Merging neighbours makes *old* foam coarser, but the
                video shows the old far-side foam is both finer and markedly
                more monodisperse.  Coalescence broadens sigma; the data narrow
                it.  It is kept here only as a weak background process.
  advection     REJECTED.  Sweeping fresh coarse foam toward the far rim
                reproduces the gradient, but requires a surface current of
                ~0.08 m/s.  Validation caught it: that model moved at 7.6
                px/frame against a measured 1.43.  Any mechanism needing a
                bubble to cross the pan within its 0.1 s life is impossible.
  local terms   ACCEPTED.  Two position-dependent terms, no transport needed:
                  * birth size scales with local superheat, so vapour bubbles
                    leave the base largest directly over the burner and much
                    smaller at the cooler edges;
                  * the burst hazard gains a "churn" term over the burner,
                    where rising vapour continuously tears the raft open, so no
                    persistent foam accumulates there while the fine raft away
                    from it survives for seconds and packs solid.
                Nucleation *position* is spread over the whole heated base and
                is deliberately decoupled from nucleation *size*.

Condensation supplies the fining: a steam cap loses volume through its exposed
surface, and since flux scales as r^2 while volume scales as r^3, dr/dt is
constant -- until the Laplace overpressure of a small cap balances the driving
force and shrinking stalls at r_eq.  Old foam therefore converges on one fine
size, which is exactly what narrows sigma(ln r) from 0.65 to 0.40.

THE MODEL
---------
N ~ 3000 surface cells inside a circular pan.  Per step:
  1. NUCLEATION  Poisson births over the whole base, 40% concentrated over the
     burner; radius log-normal, scaled by local superheat; births are rejected
     where there is no free surface, so packing is self-limiting.
  2. ADVECTION   a weak convection-roll surface flow plus an Ornstein-Uhlenbeck
     turbulent fluctuation, tuned to the measured 0.41 / 1.43 px-per-frame
     drift/jitter split.
  3. PACKING     position-based relaxation against the 7 nearest neighbours.
  4. CONDENSATION  dr/dt = -c, stalling at r_eq (see above).
  5. COALESCENCE weak, volume-conserving: r_new = (r1^3 + r2^3)^(1/3).
  6. BURSTING    h(r, age) = h0 (r/r_ref)^3.2 (1 + churn * age / t_drain).
  7. OSCILLATION Rayleigh mode-2 wobble, f = sqrt(8 sigma / rho r^3) / 2pi,
     with viscous damping -- undamped it aliases badly at 30 fps and strobes.

RENDERING
---------
A small software rasteriser, not matplotlib patches.  A pinhole camera projects
each cell to its exact first-order ellipse via the local Jacobian, so the
perspective size gradient and the b/a squash fall out of the geometry rather
than being faked.  The measured radial profile is baked into a sprite lookup
once, then each cell is a gather and an alpha blend.

Cells are drawn largest-first, smallest-last.  Sorting by image row instead
gives a convincing but wrong fish-scale look: real foam cells are coplanar, and
the small ones nestle into the gaps between the big ones.

USAGE
-----
    python bubble_sim.py                      # live animation window
    python bubble_sim.py --save foam.mp4 --seconds 15
    python bubble_sim.py --save foam.gif --seconds 6 --fps 20
    python bubble_sim.py --vigour 1.6 --seed 7 --live

Requires numpy, scipy, matplotlib (ffmpeg or pillow only for --save).
"""

from __future__ import annotations

import argparse
import math
from dataclasses import dataclass, field

import numpy as np
from scipy.spatial import cKDTree


# ---------------------------------------------------------------------------
# Parameters.  SI units: metres, seconds, kilograms.
# ---------------------------------------------------------------------------

@dataclass
class Params:
    # --- vessel -----------------------------------------------------------
    pan_radius: float = 0.105          # 21 cm pan
    hot_spot: tuple = (0.006, -0.050)  # burner sits toward the near side (-y)
    hot_sigma: float = 0.062           # width of the hottest zone
    hot_bias: float = 0.40             # fraction of births drawn over the burner;
                                       # the rest are spread over the whole base

    # --- fluid ------------------------------------------------------------
    wobble_amp: float = 0.030          # mode-2 amplitude at birth
    wobble_damp: float = 0.075         # s, viscous damping of the oscillation
    surface_tension: float = 0.059     # N/m, water at 100 C
    density: float = 958.0             # kg/m^3, water at 100 C

    # --- nucleation -------------------------------------------------------
    birth_rate: float = 17000.0        # bubbles per second over the whole pan
    r_birth_median: float = 0.00880    # 2.5 mm: the coarse, fresh near-side foam
    r_birth_sigma: float = 0.46        #
    r_birth_edge: float = 0.40         # birth size away from the burner, as a fraction measured sigma(ln r) where foam is fresh
    r_max: float = 0.0230              # films above this always break
    crowd_block: float = 0.42          # no nucleation inside this*r of a neighbour

    # --- surface flow -----------------------------------------------------
    sweep: float = 0.0062              # m/s surface current toward the far rim
    outflow: float = 0.0085            # m/s radial outflow from the plume
    swirl: float = 0.12                # tangential fraction of the outflow
    rim_brake: float = 0.85            # flow slowdown as foam packs at the wall
    turb_speed: float = 0.0070         # m/s r.m.s. turbulent fluctuation
    turb_tau: float = 0.130            # s, correlation time of the fluctuation

    # --- packing ----------------------------------------------------------
    neighbours: int = 7                # nearest neighbours used for contact
    stiffness: float = 0.42            # 0..1 positional correction per pass
    relax_passes: int = 3

    # --- condensation -----------------------------------------------------
    condense_rate: float = 5.0e-3      # m/s; dr/dt = -c while r >> r_eq
    r_eq: float = 0.00131              # equilibrium cap size, where shrinking stalls
    r_min: float = 0.00103             # below this the cap collapses

    # --- coarsening -------------------------------------------------------
    coalesce_rate: float = 0.18        # per second for a touching, freshly born pair
    coalesce_gap: float = 0.03         # pairs within this fraction of contact touch
    coalesce_age_gain: float = 0.6     # drained (older) films rupture this much faster

    # --- bursting ---------------------------------------------------------
    burst_rate: float = 0.26           # per second at r = r_ref, age 0
    burst_r_ref: float = 0.00206
    burst_r_exp: float = 3.2           # hazard ~ r^3.2: big films are far weaker
    drain_time: float = 2.2            #
    burst_churn: float = 45.0          # extra hazard over the burner, where rising
                                       # vapour churns the surface and tears the raft s, age scale of film thinning

    # --- camera -----------------------------------------------------------
    cam_distance: float = 0.185        # m from pan centre -- deliberately close
    cam_elevation: float = 52.0        # degrees above the liquid plane
    img_w: int = 478                   # same frame size as the source video
    img_h: int = 850
    fov_fill: float = 1.70             # pan diameters across the frame width
    cy_frac: float = 0.395             # image row the pan centre projects to

    # --- appearance (8-bit levels, from the measured radial profile) ------
    liquid_level: float = 130.0        # wet surface between the cells
    core_level: float = 78.0           # measured: cores sit just below the liquid
    rim_level: float = 180.0           # measured rim peak
    spec_level: float = 250.0
    border_level: float = 70.0          # dark Plateau border between cells
    rim_u: float = 0.905               # where the bright rim peaks
    rim_width: float = 0.050           # measured: the ring is thin
    draw_scale: float = 1.04           # cells deform slightly into the interstices
    ripple: float = 26.0               # amplitude of the surface-ripple highlights
    light_dir: tuple = (-0.46, -0.89)  # unit vector, image coords, from up-left
    grain: float = 3.0                 # per-pixel sensor noise, grey levels
    softness: float = 0.75             # lens/compression blur, pixels

    # --- integration ------------------------------------------------------
    substeps: int = 6                  # physics steps per rendered frame
    max_bubbles: int = 9000


# ---------------------------------------------------------------------------
# Pinhole camera looking obliquely down at the liquid plane z = 0.
# ---------------------------------------------------------------------------

class Camera:
    """World (x, y, 0) -> image (px, py), plus the local 2x2 Jacobian.

    Projecting the Jacobian rather than assuming a circle is what gives the
    simulation its perspective size gradient and the measured b/a squash
    without any extra fudge factors.
    """

    def __init__(self, p: Params):
        el = math.radians(p.cam_elevation)
        self.eye = np.array([0.0, -p.cam_distance * math.cos(el),
                             p.cam_distance * math.sin(el)])
        fwd = -self.eye / np.linalg.norm(self.eye)          # look at the origin
        self.right = np.array([1.0, 0.0, 0.0])
        self.up = np.cross(self.right, fwd)
        self.fwd = fwd
        self.cx, self.cy = p.img_w * 0.5, p.img_h * p.cy_frac

        # choose focal length so the pan spans fov_fill frame widths
        self.f = 1.0
        a, _ = self.project(np.array([[-p.pan_radius, 0.0], [p.pan_radius, 0.0]]))
        self.f = p.fov_fill * p.img_w / abs(a[1] - a[0])

    def project(self, xy: np.ndarray):
        """xy: (n, 2) world points on z = 0 -> (px, py), each (n,)."""
        v = np.empty((len(xy), 3))
        v[:, 0] = xy[:, 0] - self.eye[0]
        v[:, 1] = xy[:, 1] - self.eye[1]
        v[:, 2] = -self.eye[2]
        zc = v @ self.fwd
        zc = np.maximum(zc, 1e-4)
        px = self.f * (v @ self.right) / zc + self.cx
        py = self.cy - self.f * (v @ self.up) / zc
        return px, py

    def jacobian(self, xy: np.ndarray, eps: float = 1e-4):
        """d(image)/d(world) at each point: (n, 2, 2)."""
        p0x, p0y = self.project(xy)
        dx = xy.copy(); dx[:, 0] += eps
        dy = xy.copy(); dy[:, 1] += eps
        ax, ay = self.project(dx)
        bx, by = self.project(dy)
        J = np.empty((len(xy), 2, 2))
        J[:, 0, 0] = (ax - p0x) / eps
        J[:, 1, 0] = (ay - p0y) / eps
        J[:, 0, 1] = (bx - p0x) / eps
        J[:, 1, 1] = (by - p0y) / eps
        return p0x, p0y, J


# ---------------------------------------------------------------------------
# The foam itself.
# ---------------------------------------------------------------------------

class BoilingFoam:
    """Population of surface bubbles: birth, advection, packing, coalescence,
    bursting.  All state lives in parallel numpy arrays; dead bubbles are
    compacted out each step."""

    def __init__(self, p: Params, rng: np.random.Generator, vigour: float = 1.0):
        self.p = p
        self.rng = rng
        self.vigour = vigour
        n0 = 0
        self.pos = np.zeros((n0, 2))
        self.r = np.zeros(n0)
        self.age = np.zeros(n0)
        self.turb = np.zeros((n0, 2))
        self.phase = np.zeros(n0)
        self.tilt = np.zeros(n0)
        self.shade = np.zeros(n0)
        self.generation = np.zeros(n0, dtype=np.int32)  # coalescence count
        self.t = 0.0
        self.stats = {}

    # -- helpers -----------------------------------------------------------

    @property
    def n(self):
        return len(self.r)

    def _append(self, pos, r):
        k = len(r)
        if k == 0:
            return
        self.pos = np.vstack([self.pos, pos])
        self.r = np.concatenate([self.r, r])
        self.age = np.concatenate([self.age, np.zeros(k)])
        self.turb = np.vstack([self.turb, self.rng.normal(
            0.0, self.p.turb_speed, (k, 2))])
        self.phase = np.concatenate(
            [self.phase, self.rng.uniform(0, 2 * np.pi, k)])
        self.tilt = np.concatenate(
            [self.tilt, self.rng.uniform(0, np.pi, k)])
        self.shade = np.concatenate(
            [self.shade, self.rng.normal(1.0, 0.075, k)])
        self.generation = np.concatenate(
            [self.generation, np.zeros(k, dtype=np.int32)])

    def _keep(self, mask):
        self.pos = self.pos[mask]
        self.r = self.r[mask]
        self.age = self.age[mask]
        self.turb = self.turb[mask]
        self.phase = self.phase[mask]
        self.tilt = self.tilt[mask]
        self.shade = self.shade[mask]
        self.generation = self.generation[mask]

    # -- 1. nucleation -----------------------------------------------------

    def nucleate(self, dt):
        p = self.p
        lam = p.birth_rate * self.vigour * dt
        if self.n > p.max_bubbles:
            lam *= 0.25
        k = self.rng.poisson(lam)
        if k == 0:
            return
        # Rejection-free sampling: gaussian around the hot spot, clipped to pan.
        # The whole pan base nucleates, but bubbles are concentrated over the
        # burner.  Position and birth size are set independently: the hot spot
        # controls how *big* a bubble is, not only where it forms.
        hx, hy = p.hot_spot
        pos = self.rng.normal(0.0, p.hot_sigma, (k, 2))
        pos[:, 0] += hx
        pos[:, 1] += hy
        spread = self.rng.random(k) > p.hot_bias
        ns = int(spread.sum())
        if ns:
            th = self.rng.uniform(0, 2 * np.pi, ns)
            rad = p.pan_radius * np.sqrt(self.rng.random(ns))
            pos[spread, 0] = rad * np.cos(th)
            pos[spread, 1] = rad * np.sin(th)
        d = np.linalg.norm(pos, axis=1)
        over = d > p.pan_radius * 0.985
        if over.any():                       # fold escapees back inside the wall
            pos[over] *= (p.pan_radius * 0.985 / d[over])[:, None]
        # Vapour bubbles leave the base largest directly over the burner,
        # where the wall superheat is highest, and much smaller out at the
        # cooler edges -- so the far side is never seeded with coarse bubbles.
        d2 = ((pos - np.array(p.hot_spot)) ** 2).sum(1)
        hot = np.exp(-d2 / (2.0 * p.hot_sigma ** 2))
        r = np.exp(self.rng.normal(
            math.log(p.r_birth_median), p.r_birth_sigma, k))
        r *= p.r_birth_edge + (1.0 - p.r_birth_edge) * hot

        # A bubble can only surface where there is free liquid: reject
        # candidates that land well inside an existing bubble.  This is what
        # makes the packing fraction self-limiting instead of runaway.
        if self.n > 8:
            tree = cKDTree(self.pos)
            d, j = tree.query(pos, k=1)
            free = d > p.crowd_block * self.r[j]
            pos, r = pos[free], r[free]
        self._append(pos, r)

    # -- 2. advection ------------------------------------------------------

    def surface_velocity(self, pos):
        """Surface signature of an off-centre convection roll: liquid rises over
        the burner, spreads, and runs along the surface toward the far rim,
        where it piles up and sinks."""
        p = self.p
        d = pos - np.array(p.hot_spot)
        rad = np.linalg.norm(d, axis=1) + 1e-6
        u = d / rad[:, None]

        # radial spreading of the rising plume
        profile = np.tanh(rad / 0.020) * np.exp(-rad / 0.10)
        v = u * (p.outflow * profile)[:, None]
        # swirl: rotate the radial direction by 90 degrees
        v[:, 0] += -u[:, 1] * p.outflow * p.swirl * profile
        v[:, 1] += u[:, 0] * p.outflow * p.swirl * profile
        # the steady sweep that carries ageing foam to the far side
        v[:, 1] += p.sweep

        # brake near the rim, where the foam has nowhere left to go
        wall = np.linalg.norm(pos, axis=1) / p.pan_radius
        v *= (1.0 - p.rim_brake * np.clip((wall - 0.70) / 0.30, 0, 1))[:, None]
        return v

    def advect(self, dt):
        p = self.p
        if self.n == 0:
            return
        # Ornstein-Uhlenbeck turbulence: correlated in time, white in space.
        a = math.exp(-dt / p.turb_tau)
        sig = p.turb_speed * self.vigour * math.sqrt(1 - a * a)
        self.turb = a * self.turb + self.rng.normal(0, sig, self.turb.shape)
        v = self.surface_velocity(self.pos) * self.vigour + self.turb
        self.pos += v * dt
        # reflect off the pan wall
        d = np.linalg.norm(self.pos, axis=1)
        lim = p.pan_radius - 0.25 * self.r
        out = d > lim
        if out.any():
            self.pos[out] *= (lim[out] / d[out])[:, None]
            self.turb[out] *= -0.3

    # -- 3. packing --------------------------------------------------------

    def _neighbours(self):
        k = min(self.p.neighbours + 1, self.n)
        tree = cKDTree(self.pos)
        dist, idx = tree.query(self.pos, k=k)
        return dist[:, 1:], idx[:, 1:]          # drop self

    def relax(self, dist, idx):
        """Position-based contact resolution against the k nearest neighbours.
        Heavier (larger) bubbles move less, which is what lets big bubbles
        shoulder the fine foam aside."""
        p = self.p
        w = 1.0 / np.maximum(self.r ** 2, 1e-12)      # inverse 'mass'
        for _ in range(p.relax_passes):
            i = np.repeat(np.arange(self.n), idx.shape[1])
            j = idx.ravel()
            dv = self.pos[j] - self.pos[i]
            d = np.linalg.norm(dv, axis=1) + 1e-9
            overlap = (self.r[i] + self.r[j]) - d
            act = overlap > 0
            if not act.any():
                return
            i, j, dv, d, overlap = i[act], j[act], dv[act], d[act], overlap[act]
            n_hat = dv / d[:, None]
            wi, wj = w[i], w[j]
            share = wi / (wi + wj)
            corr = (p.stiffness * overlap * share)[:, None] * n_hat
            np.add.at(self.pos, i, -corr)

    # -- 4. coalescence ----------------------------------------------------

    def condense(self, dt):
        """Steam bubbles at a free surface lose volume as vapour condenses
        through the exposed cap, so a surviving bubble gets *smaller* with age.

        Condensation flux scales with cap area (~r^2) while the cap volume
        scales as r^3, so dr/dt is constant -- until the Laplace overpressure
        of a small cap balances the driving force and the shrinking stalls at
        r_eq.  Old foam therefore converges on one fine size, which is why the
        far-side raft is both finer and far more monodisperse (sigma 0.29) than
        the freshly nucleated near side (sigma 0.48).
        """
        p = self.p
        stall = 1.0 - (p.r_eq / np.maximum(self.r, 1e-9)) ** 2
        self.r = np.maximum(self.r - p.condense_rate * np.clip(stall, 0, 1) * dt,
                            1e-6)

    def coalesce(self, dist, idx, dt):
        """Merge deeply overlapping neighbours, conserving gas volume."""
        p = self.p
        i = np.repeat(np.arange(self.n), idx.shape[1])
        j = idx.ravel()
        keep = i < j                                   # each pair once
        i, j = i[keep], j[keep]
        if len(i) == 0:
            return
        d = np.linalg.norm(self.pos[j] - self.pos[i], axis=1)
        rsum = self.r[i] + self.r[j]
        # The packing pass keeps bubbles at contact, so coalescence must be
        # driven by *touching*, not by overlap: two bubbles share a thin film
        # and it eventually ruptures.
        touching = d < rsum * (1.0 + p.coalesce_gap)
        i, j = i[touching], j[touching]
        if len(i) == 0:
            return
        # Drainage: the longer a film has existed the thinner it is, so the
        # rupture rate climbs with the age of the younger partner.  Foam swept
        # far from the hot spot is old, and therefore coarse -- this single
        # term produces the observed near/far size gradient.
        age = np.minimum(self.age[i], self.age[j])
        rate = p.coalesce_rate * self.vigour * (1.0 + p.coalesce_age_gain * age)
        prob = 1.0 - np.exp(-rate * dt)
        hit = self.rng.random(len(i)) < prob
        i, j = i[hit], j[hit]
        if len(i) == 0:
            return
        # A bubble may only take part in one merge per step, and merges may not
        # chain (a+b and b+c in the same step).  Three vectorised passes.
        perm = self.rng.permutation(len(i))
        i, j = i[perm], j[perm]
        _, first = np.unique(i, return_index=True)
        i, j = i[np.sort(first)], j[np.sort(first)]
        _, first = np.unique(j, return_index=True)
        i, j = i[np.sort(first)], j[np.sort(first)]
        free = ~(np.isin(i, j) | np.isin(j, i))
        i, j = i[free], j[free]
        if len(i) == 0:
            return
        vi, vj = self.r[i] ** 3, self.r[j] ** 3
        f = (vi / (vi + vj))[:, None]
        self.pos[i] = self.pos[i] * f + self.pos[j] * (1 - f)
        self.r[i] = np.cbrt(vi + vj)
        self.age[i] = np.minimum(self.age[i], self.age[j])
        self.generation[i] = np.maximum(self.generation[i],
                                        self.generation[j]) + 1
        alive = np.ones(self.n, dtype=bool)
        alive[j] = False
        self._keep(alive)

    # -- 5. bursting -------------------------------------------------------

    def burst(self, dt):
        p = self.p
        # Directly over the burner the surface is continuously torn open by
        # rising vapour, so no persistent raft can build there.  Away from it
        # the fine foam survives and accumulates.  This -- not advection -- is
        # what keeps the near half coarse and the far half fine at the slow
        # surface speeds actually measured in the video.
        d2 = ((self.pos - np.array(p.hot_spot)) ** 2).sum(1)
        churn = 1.0 + p.burst_churn * np.exp(-d2 / (2.0 * p.hot_sigma ** 2))
        h = (p.burst_rate * self.vigour
             * (self.r / p.burst_r_ref) ** p.burst_r_exp
             * (1.0 + churn * self.age / p.drain_time))
        prob = 1.0 - np.exp(-h * dt)
        dead = ((self.rng.random(self.n) < prob)
                | (self.r > p.r_max) | (self.r < p.r_min))
        self.stats['popped'] = int(dead.sum())
        self._keep(~dead)

    # -- driver ------------------------------------------------------------

    def step(self, dt):
        self.nucleate(dt)
        if self.n < 3:
            self.t += dt
            return
        self.advect(dt)
        dist, idx = self._neighbours()
        self.relax(dist, idx)
        self.condense(dt)
        self.coalesce(dist, idx, dt)
        self.age += dt
        self.burst(dt)
        self.t += dt

    def wobble(self):
        """Rayleigh mode-2 capillary oscillation, f = sqrt(8 sigma / rho r^3)/2pi,
        about an axis that differs per bubble: jammed cells are never round.

        Returns a stack of 2x2 deformation matrices, area-preserving to first
        order, to be composed with the camera Jacobian.
        """
        p = self.p
        w = np.sqrt(8.0 * p.surface_tension /
                    (p.density * np.maximum(self.r, 1e-5) ** 3))
        # viscous damping: a bubble rings just after it surfaces and then
        # settles, so the foam shimmers rather than flickering
        amp = (p.wobble_amp * np.clip(self.r / 0.002, 0.35, 1.7)
               * np.exp(-self.age / p.wobble_damp))
        e = amp * np.sin(w * self.t + self.phase)
        c, s = np.cos(self.tilt), np.sin(self.tilt)
        # D = R(tilt) diag(1+e, 1-e) R(-tilt)
        D = np.empty((self.n, 2, 2))
        D[:, 0, 0] = 1.0 + e * (c * c - s * s)
        D[:, 1, 1] = 1.0 - e * (c * c - s * s)
        D[:, 0, 1] = D[:, 1, 0] = 2.0 * e * c * s
        return D


# ---------------------------------------------------------------------------
# Software renderer.
# ---------------------------------------------------------------------------

class FoamRenderer:
    """Software rasteriser.

    The whole appearance of a bubble depends only on its position inside the
    unit disc, so the shading is baked once into a sprite lookup table and each
    bubble is then a gather + alpha blend.  That keeps ~5000 bubbles per frame
    affordable in pure numpy.
    """

    SPR = 224          # sprite resolution across [-SPAN, SPAN]
    SPAN = 1.12

    def __init__(self, p: Params, cam: Camera, rng: np.random.Generator):
        self.p, self.cam, self.rng = p, cam, rng
        self.val, self.alpha = self._sprite()
        self.bg = self._background()

    # -- the measured bubble appearance, baked once ------------------------

    def _sprite(self):
        """Bake the measured radial profile once.

        From the video: intensity climbs monotonically from a dark core to a
        bright rim peaking at u ~ 0.93, with a tight specular catch-light on
        the lit side and a dark Plateau border where neighbouring cells meet.
        """
        p = self.p
        t = np.linspace(-self.SPAN, self.SPAN, self.SPR)
        ux, uy = np.meshgrid(t, t)
        u = np.sqrt(ux * ux + uy * uy)
        lx, ly = p.light_dir

        # dark core brightening outward, then the bright meniscus rim
        val = p.core_level + (p.liquid_level - p.core_level) * u ** 2.2
        val += (p.rim_level - p.core_level) * np.exp(
            -((u - p.rim_u) ** 2) / (2.0 * p.rim_width ** 2))

        # outside the rim the film curves away into the dark border shared
        # with the neighbouring cell
        outer = np.clip((u - 0.99) / 0.13, 0.0, 1.0)
        val = val * (1 - outer) + p.border_level * outer

        # tight catch-light sitting on the rim, on the lit side
        sx, sy = ux - lx * 0.88, uy - ly * 0.88
        val += (p.spec_level - p.liquid_level) * 0.070 * np.exp(
            -(sx * sx + sy * sy) / (2.0 * 0.090 ** 2))

        # weak light bouncing back through the far wall of the film
        ox, oy = ux + lx * 0.55, uy + ly * 0.55
        val += 5.0 * np.exp(-(ox * ox + oy * oy) / (2.0 * 0.30 ** 2))

        alpha = np.clip((self.SPAN - u) / 0.07, 0.0, 1.0)
        return val.astype(np.float32), alpha.astype(np.float32)

    # -- static scene ------------------------------------------------------

    def _background(self):
        """Dark pan, liquid, rim highlight and the warm burner glow below."""
        p = self.p
        h, w = p.img_h, p.img_w
        yy, xx = np.mgrid[0:h, 0:w].astype(np.float32)
        img = np.full((h, w, 3), 24.0, np.float32)          # hob / shadow

        th = np.linspace(0, 2 * np.pi, 720)
        ring = np.stack([p.pan_radius * np.cos(th),
                         p.pan_radius * np.sin(th)], 1)
        px, py = self.cam.project(ring)
        outer = _fill_polygon(px, py, h, w)
        px2, py2 = self.cam.project(ring * 0.955)
        inner = _fill_polygon(px2, py2, h, w)

        img[outer] = 46.0                                   # pan wall
        img[inner] = p.liquid_level * 0.92                  # wet liquid surface
        self.liquid_mask = inner

        glint = np.clip((yy - h * 0.55) / (h * 0.45), 0, 1)
        rim = outer & ~inner
        img[rim] = (96 + 128 * glint)[rim][:, None]

        glow = np.exp(-((yy - h * 0.985) ** 2 / (2 * (h * 0.040) ** 2)
                        + (xx - w * 0.52) ** 2 / (2 * (w * 0.40) ** 2)))
        img[..., 2] += 128 * glow                           # R
        img[..., 1] += 26 * glow
        img[..., 0] += 20 * glow

        img *= (0.88 + 0.20 * (yy / h))[..., None]
        return img

    # -- per-frame ---------------------------------------------------------

    def _ripples(self, t):
        """Bright wobbling highlights on the open liquid between the cells --
        the surface is never flat during a boil."""
        p = self.p
        if not hasattr(self, "_grid"):
            yy, xx = np.mgrid[0:p.img_h, 0:p.img_w].astype(np.float32)
            self._grid = (xx / p.img_w, yy / p.img_h)
        X, Y = self._grid
        w = (np.sin(21.0 * X + 13.0 * Y + 5.1 * t)
             + np.sin(-14.0 * X + 26.0 * Y + 3.7 * t + 1.3)
             + 0.7 * np.sin(37.0 * X - 9.0 * Y + 8.4 * t + 2.6))
        return p.ripple * np.clip(w, 0.0, None) ** 2 * 0.33

    def render(self, foam: BoilingFoam):
        p, cam = self.p, self.cam
        img = self.bg.copy()
        rip = self._ripples(foam.t)
        img[self.liquid_mask] += rip[self.liquid_mask][:, None]
        if foam.n == 0:
            return self._finish(img)

        px, py, J = cam.jacobian(foam.pos)
        D = foam.wobble()
        J = J @ D                       # world circle -> deformed cell -> image

        # A unit circle in the world maps to the ellipse {J w : |w| <= r}, so
        # inv(J)/r carries image offsets back to the unit disc.  Perspective
        # magnification and the measured b/a squash both fall out of this --
        # no ad-hoc scaling anywhere.
        det = J[:, 0, 0] * J[:, 1, 1] - J[:, 0, 1] * J[:, 1, 0]
        det = np.where(np.abs(det) < 1e-12, 1e-12, det)
        inv = np.empty_like(J)
        inv[:, 0, 0] = J[:, 1, 1] / det
        inv[:, 1, 1] = J[:, 0, 0] / det
        inv[:, 0, 1] = -J[:, 0, 1] / det
        inv[:, 1, 0] = -J[:, 1, 0] / det
        rr = (foam.r * p.draw_scale)[:, None]
        inv /= rr[:, :, None]

        rx = ry = foam.r * p.draw_scale
        half_x = np.abs(J[:, 0, 0]) * rx + np.abs(J[:, 0, 1]) * ry
        half_y = np.abs(J[:, 1, 0]) * rx + np.abs(J[:, 1, 1]) * ry
        half_x = half_x * self.SPAN + 1.0
        half_y = half_y * self.SPAN + 1.0

        # big bubbles sit higher on the surface and catch more light
        gain = (0.95 + 0.08 * np.clip(foam.r / 0.0030, 0.0, 1.7)) * foam.shade

        # Foam cells are coplanar, so a front-to-back depth sort would give a
        # false fish-scale look.  Real foam reads the other way round: the big
        # cells tile the surface and the small ones nestle into the gaps
        # between them, so draw largest first, smallest last.
        order = np.argsort(foam.r)[::-1]
        warm = np.array([0.97, 0.99, 1.02], np.float32)       # B, G, R tint
        k2u = (self.SPR - 1) / (2.0 * self.SPAN)
        val, alph = self.val, self.alpha
        S = self.SPR

        for k in order:
            if half_x[k] < 0.8 or half_y[k] < 0.8:
                continue
            x0 = max(int(px[k] - half_x[k]), 0)
            x1 = min(int(px[k] + half_x[k]) + 1, p.img_w)
            y0 = max(int(py[k] - half_y[k]), 0)
            y1 = min(int(py[k] + half_y[k]) + 1, p.img_h)
            if x1 <= x0 or y1 <= y0:
                continue

            m = inv[k]
            gx = np.arange(x0, x1, dtype=np.float32) - px[k]
            gy = np.arange(y0, y1, dtype=np.float32) - py[k]
            # separable: u = M . (dx, dy)
            ux = m[0, 0] * gx[None, :] + m[0, 1] * gy[:, None]
            uy = m[1, 0] * gx[None, :] + m[1, 1] * gy[:, None]

            ix = ((ux + self.SPAN) * k2u).astype(np.int32)
            iy = ((uy + self.SPAN) * k2u).astype(np.int32)
            ok = (ix >= 0) & (ix < S) & (iy >= 0) & (iy < S)
            if not ok.any():
                continue
            np.clip(ix, 0, S - 1, out=ix)
            np.clip(iy, 0, S - 1, out=iy)

            a = alph[iy, ix] * ok
            v = val[iy, ix] * gain[k]

            patch = img[y0:y1, x0:x1]
            a3 = a[..., None]
            img[y0:y1, x0:x1] = patch + a3 * (v[..., None] * warm - patch)

        return self._finish(img)

    def _finish(self, img):
        p = self.p
        if p.softness > 0:
            from scipy.ndimage import gaussian_filter
            img = gaussian_filter(img, (p.softness, p.softness, 0))
        img = img + self.rng.normal(0.0, p.grain, img.shape)
        h, w = p.img_h, p.img_w
        yy, xx = np.mgrid[0:h, 0:w]
        vig = 1.0 - 0.20 * (((xx - w / 2) / (w * 0.78)) ** 2
                            + ((yy - h / 2) / (h * 0.95)) ** 2)
        img *= vig[..., None]
        return np.clip(img, 0, 255).astype(np.uint8)[..., ::-1]   # BGR -> RGB


def _fill_polygon(px, py, h, w):
    """Scanline fill of a convex-ish closed polygon -> boolean mask."""
    mask = np.zeros((h, w), bool)
    n = len(px)
    ys = np.arange(h)
    for i in range(n):
        j = (i + 1) % n
        y0, y1 = py[i], py[j]
        if y0 == y1:
            continue
        lo, hi = (y0, y1) if y0 < y1 else (y1, y0)
        sel = (ys >= lo) & (ys < hi)
        if not sel.any():
            continue
        t = (ys[sel] - y0) / (y1 - y0)
        xs = px[i] + t * (px[j] - px[i])
        for y, x in zip(ys[sel], xs):
            mask[y, int(np.clip(x, 0, w - 1))] ^= True
    return np.logical_xor.accumulate(mask, axis=1)


# ---------------------------------------------------------------------------
# Driver
# ---------------------------------------------------------------------------

def build(seed=0, vigour=1.0, **overrides):
    p = Params(**overrides)
    rng = np.random.default_rng(seed)
    cam = Camera(p)
    foam = BoilingFoam(p, rng, vigour)
    ren = FoamRenderer(p, cam, rng)
    return p, foam, ren


def spin_up(foam, seconds, dt):
    """Run without rendering until the population reaches steady state."""
    for _ in range(int(seconds / dt)):
        foam.step(dt)


def main():
    ap = argparse.ArgumentParser(description=__doc__.split("\n")[1])
    ap.add_argument("--seconds", type=float, default=10.0)
    ap.add_argument("--fps", type=int, default=30)
    ap.add_argument("--seed", type=int, default=3)
    ap.add_argument("--vigour", type=float, default=1.0,
                    help="boil intensity: 0.5 simmer, 1.0 as filmed, 2.0 furious")
    ap.add_argument("--save", metavar="FILE", default=None,
                    help="write an .mp4 or .gif instead of opening a window")
    ap.add_argument("--live", action="store_true",
                    help="force an interactive window")
    ap.add_argument("--scale", type=float, default=1.0,
                    help="render resolution multiplier")
    args = ap.parse_args()

    import matplotlib
    if args.save and not args.live:
        matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    from matplotlib import animation

    over = {}
    if args.scale != 1.0:
        over["img_w"] = int(478 * args.scale)
        over["img_h"] = int(850 * args.scale)
    p, foam, ren = build(seed=args.seed, vigour=args.vigour, **over)

    dt = 1.0 / (args.fps * p.substeps)
    print("spinning up the foam ...")
    spin_up(foam, 2.0, dt)
    print(f"steady state: {foam.n} bubbles on the surface")

    fig, ax = plt.subplots(figsize=(p.img_w / 100, p.img_h / 100), dpi=100)
    fig.subplots_adjust(0, 0, 1, 1)
    ax.set_axis_off()
    im = ax.imshow(ren.render(foam), interpolation="bilinear", animated=True)
    hud = ax.text(0.025, 0.985, "", transform=ax.transAxes, va="top",
                  color="#e8e8e8", fontsize=7.5, family="monospace",
                  bbox=dict(fc="#000000", alpha=0.42, pad=3.5, lw=0))

    nframes = int(args.seconds * args.fps)

    def update(_):
        for _ in range(p.substeps):
            foam.step(dt)
        im.set_array(ren.render(foam))
        rr = foam.r * 1000.0
        hud.set_text(
            f"t {foam.t:6.2f}s   bubbles {foam.n:5d}\n"
            f"r  med {np.median(rr):.2f} mm  p95 {np.percentile(rr, 95):.2f} mm\n"
            f"sigma(ln r) {np.std(np.log(np.maximum(rr, 1e-6))):.2f}   "
            f"popped/frame {foam.stats.get('popped', 0) * p.substeps}")
        return im, hud

    anim = animation.FuncAnimation(fig, update, frames=nframes,
                                   interval=1000 / args.fps, blit=True)

    if args.save:
        print(f"rendering {nframes} frames -> {args.save}")
        if args.save.lower().endswith(".gif"):
            anim.save(args.save, writer="pillow", fps=args.fps)
        else:
            anim.save(args.save, fps=args.fps, dpi=100,
                      extra_args=["-pix_fmt", "yuv420p", "-crf", "20"])
        print("done")
    else:
        plt.show()


if __name__ == "__main__":
    main()
