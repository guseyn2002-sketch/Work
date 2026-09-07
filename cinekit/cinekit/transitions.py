# -*- coding: utf-8 -*-
"""Cut treatments. Applied to the picture array, before graphics."""
import math
import numpy as np
from PIL import Image, ImageFilter

from .timeline import band


def flash(arr, t, cuts, strength=0.16, hold=0.10, strong=()):
    """A short additive lift on the cut — reads as an in-camera exposure jump."""
    for c in cuts:
        dt = t - c
        if -0.02 <= dt < hold:
            s = strength * (2.0 if any(abs(c - x) < 1e-6 for x in strong) else 1.0)
            arr = arr + 255.0 * (1 - dt / hold) ** 2 * s
    return arr


def leak(arr, optics, t, cuts, base=0.040, breathe=0.030, kick=0.085, decay=0.55):
    """Warm light spill that swells after each cut."""
    amt = base + breathe * math.sin(t * 0.9)
    for c in cuts:
        if 0 <= t - c < decay:
            amt += kick * (1 - (t - c) / decay) ** 1.6
    return arr + optics.leak() * amt


def whip(arr, u, strength=1.0, steps=9):
    """Horizontal smear peaking mid-transition."""
    r = math.sin(math.pi * max(0.0, min(1.0, u))) ** 0.7 * strength * 22
    if r < 0.5:
        return arr
    acc = np.zeros_like(arr)
    for s in range(steps):
        acc += np.roll(arr, int(round((s - (steps - 1) / 2) * r / 2)), axis=1)
    return acc / steps


def glitch(arr, u, amount=1.0, seed=0):
    """RGB tearing in horizontal bands — digital, not filmic."""
    if amount <= 0:
        return arr
    rng = np.random.default_rng(int(u * 9973) + seed)
    out = arr.copy()
    h = arr.shape[0]
    for _ in range(int(6 * amount)):
        y0 = rng.integers(0, h - 8)
        hh = int(rng.integers(6, max(7, int(h * 0.05))))
        sh = int(rng.integers(-40, 40) * amount)
        out[y0:y0 + hh] = np.roll(out[y0:y0 + hh], sh, axis=1)
        ch = int(rng.integers(0, 3))
        out[y0:y0 + hh, :, ch] = np.roll(out[y0:y0 + hh, :, ch], sh * 2, axis=1)
    return out


def burn(arr, u, tint=(255, 170, 80)):
    """Film-burn bloom: a hot blotch that swells and clears."""
    a = math.sin(math.pi * max(0.0, min(1.0, u))) ** 1.4
    if a < 0.01:
        return arr
    h, w = arr.shape[:2]
    yy, xx = np.mgrid[0:h, 0:w].astype(np.float32)
    cx, cy = w * (0.30 + 0.4 * u), h * 0.42
    r = np.sqrt(((xx - cx) / (w * 0.55)) ** 2 + ((yy - cy) / (h * 0.42)) ** 2)
    g = np.clip(1.25 - r, 0, 1) ** 2.0
    return arr + g[..., None] * np.array(tint, np.float32)[None, None] * a * 0.9


def dip(arr, u, colour=(0, 0, 0)):
    """Dip to a colour and back — u is 0..1 across the whole transition."""
    a = math.sin(math.pi * u)
    c = np.array(colour, np.float32)[None, None]
    return arr * (1 - a) + c * a


def zoom_blur(arr, u, strength=1.0, steps=8):
    """Radial rush — the punch into a reveal."""
    a = math.sin(math.pi * max(0.0, min(1.0, u))) ** 0.8 * strength
    if a < 0.02:
        return arr
    h, w = arr.shape[:2]
    im = Image.fromarray(arr.clip(0, 255).astype(np.uint8))
    acc = np.zeros_like(arr, np.float32)
    for s in range(steps):
        k = 1.0 + a * 0.06 * (s + 1) / steps
        nw, nh = int(w * k), int(h * k)
        z = im.resize((nw, nh), Image.BILINEAR).crop(
            ((nw - w) // 2, (nh - h) // 2, (nw - w) // 2 + w, (nh - h) // 2 + h))
        acc += np.asarray(z, np.float32)
    return acc / steps


def push(a, b, u, axis="x", sign=1):
    """One frame shoves the other out — u in 0..1."""
    h, w = a.shape[:2]
    out = np.zeros_like(a)
    if axis == "x":
        off = int(w * u) * sign
        out[:] = np.roll(a, off, axis=1)
        bb = np.roll(b, off - w * sign, axis=1)
    else:
        off = int(h * u) * sign
        out[:] = np.roll(a, off, axis=0)
        bb = np.roll(b, off - h * sign, axis=0)
    mask = np.zeros((h, w, 1), np.float32)
    if axis == "x":
        cut = int(w * u)
        if sign > 0: mask[:, :cut] = 1
        else:        mask[:, w - cut:] = 1
    else:
        cut = int(h * u)
        if sign > 0: mask[:cut] = 1
        else:        mask[h - cut:] = 1
    return out * (1 - mask) + bb * mask
