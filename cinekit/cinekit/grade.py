# -*- coding: utf-8 -*-
"""Colour: film-style grades as generated 3D LUTs, plus the optical layer.

Building the look as a .cube means the exact same grade can be applied in
ffmpeg, handed to a colourist, or loaded in Resolve.
"""
import os, math
import numpy as np
from PIL import Image, ImageFilter

from . import LUTS


# --------------------------------------------------------------- curves
def _spline(xs, ys, v):
    return np.interp(v, xs, ys)


def _tone(v, lift=0.02, gamma=1.0, gain=1.0, contrast=1.0, pivot=0.435):
    v = v * gain + lift * (1 - v)
    v = np.clip(v, 0, 1) ** (1.0 / max(gamma, 1e-3))
    return np.clip((v - pivot) * contrast + pivot, 0, 1)


LOOKS = {
    # lifted matte blacks, warm highlights — the default "shot on film" register
    "kodak": dict(lift=(0.022, 0.019, 0.032), gamma=(1.00, 1.00, 0.99),
                  gain=(1.010, 1.000, 0.986), contrast=1.06, sat=1.10,
                  split=((0.006, 0.000, 0.026), (0.014, 0.004, -0.010))),
    # cool shadows, clean neutral mids: tech / product register
    "apple":  dict(lift=(0.010, 0.012, 0.020), gamma=(1.0, 1.0, 1.0),
                   gain=(1.0, 1.0, 1.005), contrast=1.10, sat=1.04,
                   split=((0.000, 0.004, 0.020), (0.006, 0.002, 0.000))),
    # deep, desaturated, warm skin: fashion film
    "couture": dict(lift=(0.016, 0.013, 0.018), gamma=(0.98, 0.98, 0.97),
                    gain=(1.005, 0.995, 0.978), contrast=1.13, sat=0.80,
                    split=((0.004, 0.000, 0.014), (0.020, 0.008, -0.008))),
    # bright, creamy, soft: beauty
    "beauty": dict(lift=(0.030, 0.028, 0.030), gamma=(1.03, 1.02, 1.01),
                   gain=(1.010, 1.002, 0.996), contrast=1.02, sat=1.06,
                   split=((0.006, 0.002, 0.012), (0.016, 0.008, 0.000))),
    # hard contrast, steel highlights: automotive
    "auto":   dict(lift=(0.008, 0.009, 0.016), gamma=(0.97, 0.97, 0.97),
                   gain=(0.998, 1.000, 1.010), contrast=1.18, sat=0.86,
                   split=((0.000, 0.006, 0.026), (0.004, 0.002, 0.006))),
    # saturated, warm, slightly crushed: y2k / pinterest
    "y2k":    dict(lift=(0.026, 0.022, 0.038), gamma=(1.01, 1.01, 1.00),
                   gain=(1.014, 1.002, 0.984), contrast=1.05, sat=1.16,
                   split=((0.008, 0.000, 0.030), (0.018, 0.006, -0.012))),
}


def build_lut(look="kodak", size=33, path=None):
    """Generate a .cube 3D LUT for one of the looks."""
    p = LOOKS[look]
    g = np.linspace(0, 1, size)
    b, gg, r = np.meshgrid(g, g, g, indexing="ij")   # .cube order: r fastest
    rgb = np.stack([r, gg, b], -1).astype(np.float32)

    out = np.empty_like(rgb)
    for c in range(3):
        out[..., c] = _tone(rgb[..., c], p["lift"][c], p["gamma"][c],
                            p["gain"][c], p["contrast"])
    luma = out @ np.array([0.2126, 0.7152, 0.0722], np.float32)
    out = np.clip(luma[..., None] + (out - luma[..., None]) * p["sat"], 0, 1)

    sh, hi = np.array(p["split"][0]), np.array(p["split"][1])
    w_sh = np.clip(1 - luma * 1.9, 0, 1)[..., None]
    w_hi = np.clip((luma - 0.45) * 1.9, 0, 1)[..., None]
    out = np.clip(out + sh * w_sh + hi * w_hi, 0, 1)

    path = path or os.path.join(LUTS, f"{look}.cube")
    os.makedirs(os.path.dirname(path), exist_ok=True)
    with open(path, "w") as f:
        f.write(f"# cinekit look: {look}\nTITLE \"cinekit_{look}\"\n")
        f.write(f"LUT_3D_SIZE {size}\nDOMAIN_MIN 0 0 0\nDOMAIN_MAX 1 1 1\n")
        for v in out.reshape(-1, 3):
            f.write(f"{v[0]:.6f} {v[1]:.6f} {v[2]:.6f}\n")
    return path


def ffmpeg_chain(look="kodak", halation=0.13, sharpen=0.55, lut_path=None):
    """The grade as an ffmpeg filtergraph — LUT plus the optical layer."""
    lut = lut_path or os.path.join(LUTS, f"{look}.cube")
    if not os.path.exists(lut):
        build_lut(look, path=lut)
    esc = lut.replace("\\", "/").replace(":", "\\:")
    return (
        f"split=2[g][h];"
        f"[h]curves=all='0/0 0.88/0 0.95/0.32 1/1',gblur=sigma=22:steps=3,"
        f"colorchannelmixer=rr=1.0:gg=0.84:bb=0.72[hal];"
        f"[g]lut3d=file='{esc}':interp=tetrahedral[gg];"
        f"[gg][hal]blend=all_mode=screen:all_opacity={halation}[bl];"
        f"[bl]unsharp=5:5:{sharpen}:5:5:0.0"
    )


# --------------------------------------------------- optical layer (numpy)
class Optics:
    """Vignette, scanlines, grain, aberration, leaks — precomputed per size."""

    def __init__(self, w, h, seed=42, grain_tiles=9):
        self.W, self.H = w, h
        yy, xx = np.mgrid[0:h, 0:w].astype(np.float32)
        self.xx, self.yy = xx, yy
        nx = (xx - w / 2) / (w / 2)
        ny = (yy - h / 2) / (h / 2)
        rad = np.sqrt(nx ** 2 + (ny * 0.62) ** 2)
        self.vignette = np.clip(1.0 - 0.30 * np.clip(rad - 0.42, 0, None) ** 1.6 * 2.6,
                                0.55, 1.0)[..., None]
        self.scan = (1.0 - 0.028 * (np.sin(yy[:, :1] * math.pi / 2.0) * .5 + .5)
                     ).astype(np.float32)[..., None]
        rng = np.random.default_rng(seed)
        self.grain = []
        for _ in range(grain_tiles):
            g = rng.normal(0, 1, (h // 2, w // 2))
            g = np.array(Image.fromarray(((g * 40) + 128).clip(0, 255).astype(np.uint8))
                         .resize((w, h), Image.BILINEAR), np.float32)[..., None] - 128.0
            self.grain.append(g)
        self._leak = None

    def leak(self, cx=1.02, cy=0.18, tint=(255, 152, 78)):
        if self._leak is None:
            g = np.clip(1.25 - np.sqrt(((self.xx - self.W * cx) / (self.W * .92)) ** 2 +
                                       ((self.yy - self.H * cy) / (self.H * .55)) ** 2), 0, 1) ** 2.1
            self._leak = g[..., None] * np.array(tint, np.float32)[None, None, :]
        return self._leak

    def aberration(self, arr, k=1.0019):
        im = Image.fromarray(arr.clip(0, 255).astype(np.uint8))
        r, g, b = im.split()
        rw, rh = int(self.W * k), int(self.H * k)
        r = r.resize((rw, rh), Image.BICUBIC).crop(
            ((rw - self.W) // 2, (rh - self.H) // 2,
             (rw - self.W) // 2 + self.W, (rh - self.H) // 2 + self.H))
        k2 = 2 - k
        bw, bh = int(self.W * k2), int(self.H * k2)
        bb = Image.new("L", (self.W, self.H))
        bb.paste(b.resize((bw, bh), Image.BICUBIC), ((self.W - bw) // 2, (self.H - bh) // 2))
        return np.asarray(Image.merge("RGB", (r, g, bb)), np.float32)

    def apply(self, arr, n, grain=0.068, vignette=True, scan=True,
              aberration=1.0019, leak=0.0):
        if aberration:
            arr = self.aberration(arr, aberration)
        if vignette:
            arr = arr * self.vignette
        if scan:
            arr = arr * self.scan
        if leak:
            arr = arr + self.leak() * leak
        if grain:
            arr = arr + self.grain[n % len(self.grain)] * grain
        return arr


def bloom(arr, threshold=0.80, sigma=26, amount=0.16, tint=(1.0, 0.86, 0.74)):
    """Screen a blurred highlight pass back over the frame."""
    im = Image.fromarray(np.clip(arr, 0, 255).astype(np.uint8))
    a = np.asarray(im, np.float32) / 255.0
    l = a @ np.array([0.2126, 0.7152, 0.0722], np.float32)
    hi = np.clip((l - threshold) / max(1 - threshold, 1e-3), 0, 1)[..., None] * a
    hi = np.asarray(Image.fromarray((hi * 255).astype(np.uint8))
                    .filter(ImageFilter.GaussianBlur(sigma)), np.float32) / 255.0
    hi *= np.array(tint, np.float32)[None, None, :]
    out = 1 - (1 - a) * (1 - hi * amount)
    return np.clip(out * 255.0, 0, 255)
