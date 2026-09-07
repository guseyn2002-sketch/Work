# -*- coding: utf-8 -*-
"""Look presets.

Each preset names a colour grade, a type pairing, an optical intensity and a
sound register. They are starting points that keep a piece internally
consistent, not impersonations of anyone's identity.
"""
from .shapes import PALETTE

PRESETS = {
    "tech": dict(
        note="Neutral grotesk, quiet motion, cool clean grade. Product and tech register.",
        grade="apple", display=("Inter", 900), body=("Inter", 400),
        subtitle_style="quiet", tracking=0.02,
        optics=dict(grain=0.028, aberration=1.0009, vignette=True, leak=0.012, halation=0.08),
        accents=[PALETTE["white"], PALETTE["cream"]],
        audio=dict(bpm=88, palette=("pad_warm", "piano", "glock"), drums=False,
                   reverb=0.34, master=-14.0),
        motion=dict(ease="expo", stagger=3, pulse=0.006),
    ),
    "couture": dict(
        note="High-contrast serif in wide caps, hairline rules, muted film grade. Fashion film.",
        grade="couture", display=("PlayfairDisplay", 900), body=("Jost", 300),
        subtitle_style="quiet", tracking=0.16,
        optics=dict(grain=0.075, aberration=1.0022, vignette=True, leak=0.05, halation=0.16),
        accents=[PALETTE["champagne"], PALETTE["gold"]],
        audio=dict(bpm=74, palette=("strings", "cello", "harp"), drums=False,
                   reverb=0.42, master=-15.0),
        motion=dict(ease="out", stagger=6, pulse=0.0),
    ),
    "beauty": dict(
        note="Soft serif with a rounded sans, creamy bright grade, gentle bloom.",
        grade="beauty", display=("Fraunces", 900), body=("DMSans", 500),
        subtitle_style="pop", tracking=0.01,
        optics=dict(grain=0.038, aberration=1.0012, vignette=True, leak=0.045, halation=0.19),
        accents=[PALETTE["cream"], PALETTE["pink"]],
        audio=dict(bpm=92, palette=("epiano", "pad_halo", "celesta"), drums=True,
                   reverb=0.38, master=-14.0),
        motion=dict(ease="back", stagger=4, pulse=0.008),
    ),
    "auto": dict(
        note="Condensed industrial caps, hard contrast, steel highlights, low grain.",
        grade="auto", display=("Archivo900", 900), body=("BarlowCondensed", 600),
        subtitle_style="quiet", tracking=0.10,
        optics=dict(grain=0.030, aberration=1.0016, vignette=True, leak=0.02, halation=0.11),
        accents=[PALETTE["white"], PALETTE["cyan"]],
        audio=dict(bpm=124, palette=("synth_bass", "saw_lead", "timpani"), drums=True,
                   reverb=0.22, master=-13.0),
        motion=dict(ease="expo", stagger=2, pulse=0.016),
    ),
    "y2k": dict(
        note="Chrome display type, per-word colour, heavy stickers, warm saturated grade.",
        grade="y2k", display=("Unbounded", 900), body=("Montserrat", 900),
        subtitle_style="pop", tracking=0.0,
        optics=dict(grain=0.068, aberration=1.0019, vignette=True, leak=0.055, halation=0.13),
        accents=[PALETTE["pink"], PALETTE["cyan"], PALETTE["yellow"], PALETTE["lime"]],
        audio=dict(bpm=100, palette=("bell", "pad_warm", "epiano"), drums=True,
                   reverb=0.34, master=-14.0),
        motion=dict(ease="elastic", stagger=4, pulse=0.014),
    ),
    "editorial": dict(
        note="Serif headline over a grotesk deck, restrained motion, kodak grade.",
        grade="kodak", display=("BodoniModa", 900), body=("SpaceGrotesk", 500),
        subtitle_style="quiet", tracking=0.06,
        optics=dict(grain=0.062, aberration=1.0015, vignette=True, leak=0.04, halation=0.14),
        accents=[PALETTE["cream"], PALETTE["gold"]],
        audio=dict(bpm=82, palette=("piano", "strings", "flute"), drums=False,
                   reverb=0.36, master=-14.5),
        motion=dict(ease="out", stagger=5, pulse=0.004),
    ),
}


def get(name):
    if name not in PRESETS:
        raise KeyError(f"unknown preset {name!r}; have {sorted(PRESETS)}")
    return PRESETS[name]


def names():
    return sorted(PRESETS)
