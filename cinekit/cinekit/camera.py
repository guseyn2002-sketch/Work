# -*- coding: utf-8 -*-
"""A virtual camera over the source frame — crop, punch, roll, handheld.

Shooting one long take and cutting several framings out of it is how a single
clip becomes an edit. On 4K source a 1080p delivery has about 2x of clean
headroom, so a wide, a medium and a close can all be real.
"""
import math
import numpy as np
from PIL import Image

from .timeline import band, ease


def _noise(seed, n, octaves=3):
    rng = np.random.default_rng(seed)
    out = np.zeros(n)
    for o in range(octaves):
        step = max(2, int(24 / (o + 1)))
        k = n // step + 2
        pts = rng.normal(0, 1, k)
        out += np.interp(np.linspace(0, k - 1, n), np.arange(k), pts) / (o + 1)
    return out / (np.abs(out).max() or 1)


class Camera:
    def __init__(self, store, out_w=1080, out_h=1920, seed=7, resample=Image.BICUBIC):
        self.store = store
        self.W, self.H = out_w, out_h
        self.resample = resample
        self._nx = _noise(seed, 20000)
        self._ny = _noise(seed + 12, 20000)
        self._nr = _noise(seed + 36, 20000)

    # ------------------------------------------------------------------
    def geometry(self, st, n, track=None):
        """Resolve the camera transform for one frame.

        Returned separately from the pixels so the identical transform can be
        applied to a subject mask, which is what lets graphics sit *behind*
        a person.
        """
        z, cx, cy = st["zoom"], st["cx"], st["cy"]
        roll = st["roll"]

        if track is not None and st.get("track"):
            tx, ty = track["at"](st["src_t"])
            aim = st["meta"].get("aim", (0.5, 0.42))
            hold = st["meta"].get("track_strength", 1.0)
            cw = ch = 1.0 / z
            cx += (tx - (aim[0] - 0.5) * cw - cx) * hold
            cy += (ty - (aim[1] - 0.5) * ch - cy) * hold

        amp = st["shake"]
        roll += self._nr[n % len(self._nr)] * amp * 0.18
        dx = self._nx[n % len(self._nx)] * amp
        dy = self._ny[n % len(self._ny)] * amp

        th = math.radians(abs(roll))
        if th > 1e-4:
            pad = max((self.W * math.cos(th) + self.H * math.sin(th)) / self.W,
                      (self.H * math.cos(th) + self.W * math.sin(th)) / self.H) * 1.004
        else:
            pad = 1.0
        z = max(z, pad)

        SW, SH = self.store.width, self.store.height
        cw, chh = SW / z * pad, SH / z * pad
        ppx = self.W / (SW / z)
        ccx = cx * SW + dx / ppx
        ccy = cy * SH + dy / ppx
        left = max(0.0, min(SW - cw, ccx - cw / 2))
        top = max(0.0, min(SH - chh, ccy - chh / 2))

        return dict(left=left, top=top, cw=cw, ch=chh, pad=pad, roll=roll,
                    zoom=z, mirror=st["mirror"], need=self.W / (SW / z),
                    src=(SW, SH))

    def apply_geometry(self, im, geo, resample=None):
        """Apply a resolved transform to any image laid out in source space."""
        resample = resample or self.resample
        sw, sh = im.size
        k = sw / geo["src"][0]
        box = (max(0.0, geo["left"] * k), max(0.0, geo["top"] * k),
               min(float(sw), (geo["left"] + geo["cw"]) * k),
               min(float(sh), (geo["top"] + geo["ch"]) * k))
        pad = geo["pad"]
        ow, oh = int(round(self.W * pad)), int(round(self.H * pad))
        out = im.resize((ow, oh), resample, box=box)
        if geo["mirror"]:
            out = out.transpose(Image.FLIP_LEFT_RIGHT)
        if abs(geo["roll"]) > 0.05:
            out = out.rotate(geo["roll"], resample=resample, expand=False)
        if pad != 1.0:
            l, t = (ow - self.W) // 2, (oh - self.H) // 2
            out = out.crop((l, t, l + self.W, t + self.H))
        if out.size != (self.W, self.H):
            out = out.resize((self.W, self.H), resample)
        return out

    def frame(self, shot_state, n, track=None, retime="flow", geo=None):
        """Render one output frame from a Shot.state() dict."""
        st = shot_state
        geo = geo or self.geometry(st, n, track)
        im = self.store.sample(st["src_t"], geo["need"], mode=retime)
        out = self.apply_geometry(im, geo)
        if st["blur"] > 0:
            out = directional_blur(out, st["blur"] * math.sin(math.pi * st["u"]) ** 0.7)
        return out

    def mask_frame(self, mask, geo):
        """Put a source-space subject mask into output space (float 0..1)."""
        if mask is None:
            return None
        m = np.asarray(mask, np.float32)
        im = Image.fromarray(np.clip(m * 255, 0, 255).astype(np.uint8))
        if im.size != geo["src"]:
            im = im.resize(geo["src"], Image.BILINEAR)
        return np.asarray(self.apply_geometry(im, geo, Image.BILINEAR),
                          np.float32)[..., None] / 255.0


def directional_blur(im, strength, axis=1, steps=7):
    """Horizontal smear for whip pans."""
    r = strength * 11
    if r < 0.4:
        return im
    arr = np.asarray(im).astype(np.float32)
    acc = np.zeros_like(arr)
    for s in range(steps):
        off = int(round((s - (steps - 1) / 2) * r / 2))
        acc += np.roll(arr, off, axis=axis)
    return Image.fromarray(np.clip(acc / steps, 0, 255).astype(np.uint8))


def suggest_shots(track, store, min_len=0.6):
    """Propose framings from the tracked subject.

    Cheap but useful: report where the subject is over time and how much
    punch-in the source can carry, so a timeline can be laid out against
    facts rather than guesses.
    """
    found = [r for r in track if r["found"]]
    if not found:
        return []
    xs = [r["face"][0] for r in found]
    ys = [r["face"][1] for r in found]
    spans = [r["scale"] for r in found]
    head = float(np.median(spans))
    max_z = min(store.width / 1080.0, store.height / 1920.0)
    return dict(
        present=(found[0]["t"], found[-1]["t"]),
        face_x=(min(xs), max(xs)), face_y=(min(ys), max(ys)),
        drift=max(xs) - min(xs), shoulder_span=head,
        clean_zoom_max=round(max_z, 2),
        suggestion=("subject drifts across frame — punch in and track"
                    if max(xs) - min(xs) > 0.15 else "subject stable — static framings fine"),
    )
