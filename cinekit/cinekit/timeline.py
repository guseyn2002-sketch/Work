# -*- coding: utf-8 -*-
"""Beat-locked timeline primitives.

Cuts that land on the beat are most of what separates an edit that feels
professional from one that does not, so every duration here is expressed in
beats and the grid is the unit of composition.
"""
import math

# ----------------------------------------------------------------- easing
def linear(t):      return t
def ease_in(t):     return t * t * t
def ease_out(t):    return 1 - (1 - t) ** 3
def ease_io(t):     return 3 * t * t - 2 * t * t * t
def ease_out_expo(t):
    return 1.0 if t >= 1 else 1 - 2 ** (-10 * t)
def ease_in_expo(t):
    return 0.0 if t <= 0 else 2 ** (10 * (t - 1))
def ease_back(t, s=1.70158):
    return 1 + (s + 1) * (t - 1) ** 3 + s * (t - 1) ** 2
def ease_out_elastic(t, p=0.36):
    if t <= 0: return 0.0
    if t >= 1: return 1.0
    return 2 ** (-9 * t) * math.sin((t - p / 4) * (2 * math.pi) / p) + 1
def ease_out_bounce(t):
    n, d = 7.5625, 2.75
    if t < 1 / d:       return n * t * t
    if t < 2 / d:       t -= 1.5 / d;  return n * t * t + 0.75
    if t < 2.5 / d:     t -= 2.25 / d; return n * t * t + 0.9375
    t -= 2.625 / d;     return n * t * t + 0.984375

def spring(t, stiffness=180.0, damping=18.0, mass=1.0):
    """Critically-ish damped spring, normalised to settle at 1.0."""
    if t <= 0: return 0.0
    w0 = math.sqrt(stiffness / mass)
    zeta = damping / (2 * math.sqrt(stiffness * mass))
    if zeta < 1:
        wd = w0 * math.sqrt(1 - zeta * zeta)
        return 1 - math.exp(-zeta * w0 * t) * (math.cos(wd * t) + zeta * w0 / wd * math.sin(wd * t))
    return 1 - math.exp(-w0 * t) * (1 + w0 * t)

EASE = {
    "linear": linear, "in": ease_in, "out": ease_out, "io": ease_io,
    "expo": ease_out_expo, "in_expo": ease_in_expo, "back": ease_back,
    "elastic": ease_out_elastic, "bounce": ease_out_bounce,
}

def ease(name, t):
    return EASE.get(name, ease_io)(max(0.0, min(1.0, t)))

def clamp(v, a=0.0, b=1.0):
    return max(a, min(b, v))

def band(t, t0, t1):
    """Normalised progress of t through [t0, t1], clamped."""
    return clamp((t - t0) / max(t1 - t0, 1e-9))

def mix(a, b, u):
    return a + (b - a) * u


# ----------------------------------------------------------------- grid
class Grid:
    """A musical grid. Everything in an edit is placed against this."""

    def __init__(self, bpm=100.0, fps=30):
        self.bpm, self.fps = float(bpm), int(fps)

    @property
    def beat(self):  return 60.0 / self.bpm
    @property
    def bar(self):   return self.beat * 4

    def b(self, beats):        return round(beats * self.beat, 6)
    def bars(self, n):         return round(n * self.bar, 6)
    def frame(self, t):        return int(round(t * self.fps))
    def time(self, frame):     return frame / self.fps

    def snap(self, t, division=1.0):
        """Snap a time to the nearest 1/division of a beat."""
        step = self.beat / division
        return round(t / step) * step

    def pulse(self, t, amount=0.014, width=0.085, start=0.0):
        """Scale factor that kicks on each beat and decays — the picture breathing."""
        if t < start:
            return 1.0
        ph = (t - start) % self.beat
        return 1.0 + amount * math.exp(-(ph / width) ** 2)


# ----------------------------------------------------------------- shots
class Shot:
    """One cut: where it sits on the timeline and what the camera does."""

    def __init__(self, name, t0, t1, src=(0.0, 0.0), zoom=(1.0, 1.0),
                 center=((0.5, 0.5), (0.5, 0.5)), roll=(0.0, 0.0),
                 easing="io", mirror=False, hold=None, shake=0.0,
                 blur=0.0, track=None, meta=None):
        self.name = name
        self.t0, self.t1 = float(t0), float(t1)
        self.s0, self.s1 = src
        self.z0, self.z1 = zoom
        self.c0, self.c1 = center
        self.roll = roll
        self.easing = easing
        self.mirror = mirror
        self.hold = hold
        self.shake = shake
        self.blur = blur
        self.track = track          # None | "face" | "body" — auto-reframe target
        self.meta = meta or {}

    @property
    def dur(self):   return self.t1 - self.t0
    @property
    def speed(self):
        return abs(self.s1 - self.s0) / self.dur if self.dur else 0.0

    def contains(self, t):
        return self.t0 <= t < self.t1

    def state(self, t):
        u = band(t, self.t0, self.t1)
        e = ease(self.easing, u)
        return dict(
            u=u, e=e, name=self.name,
            src_t=self.hold if self.hold is not None else mix(self.s0, self.s1, u),
            zoom=mix(self.z0, self.z1, e),
            cx=mix(self.c0[0], self.c1[0], e),
            cy=mix(self.c0[1], self.c1[1], e),
            roll=mix(self.roll[0], self.roll[1], e),
            mirror=self.mirror, shake=self.shake, blur=self.blur,
            track=self.track, meta=self.meta,
        )

    def __repr__(self):
        return (f"<Shot {self.name} {self.t0:.2f}-{self.t1:.2f} "
                f"src {self.s0:.2f}->{self.s1:.2f} @{self.speed:.2f}x>")


class Timeline:
    def __init__(self, grid):
        self.grid = grid
        self.shots = []

    def add(self, shot):
        self.shots.append(shot)
        return shot

    def cut(self, name, beats, **kw):
        """Append a shot `beats` long, starting where the previous one ended."""
        t0 = self.shots[-1].t1 if self.shots else 0.0
        s = Shot(name, t0, t0 + self.grid.b(beats), **kw)
        return self.add(s)

    @property
    def duration(self):
        return self.shots[-1].t1 if self.shots else 0.0

    @property
    def frames(self):
        return int(round(self.duration * self.grid.fps))

    @property
    def cuts(self):
        return [s.t0 for s in self.shots if s.t0 > 0]

    def at(self, t):
        for s in self.shots:
            if s.contains(t):
                return s
        return self.shots[-1] if self.shots else None

    def summary(self):
        out = [f"{self.grid.bpm:g} BPM · beat {self.grid.beat:.3f}s · "
               f"{self.duration:.2f}s · {self.frames} frames"]
        for s in self.shots:
            out.append(f"  {s.name:<12s} {s.t0:6.2f}-{s.t1:6.2f} "
                       f"({s.dur:4.2f}s)  src {s.s0:5.2f}->{s.s1:5.2f}  "
                       f"{s.speed:4.2f}x  z {s.z0:.2f}->{s.z1:.2f}"
                       + (f"  track={s.track}" if s.track else ""))
        return "\n".join(out)
