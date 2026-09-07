"""cinekit — a code-first pipeline for brand-grade short-form video.

Modules
-------
timeline    beat grid, shots, easing
source      frame store, sub-frame sampling, optical-flow retiming
track       pose / face tracking and person masks (MediaPipe)
camera      virtual camera: crop, zoom, roll, handheld, auto-reframe
grade       film looks, LUTs, halation, grain, vignette, aberration
typo        chrome type, kinetic subtitles, brand type presets
shapes      vector stickers and frames
transitions whip, flash, leak, glitch, burn
web         headless-Chromium motion graphics
scene3d     Blender scene rendering
audio       synthesis, soundfont scoring, mastering
voice       narration with per-word timings
brands      look presets (apple / luxury / beauty / auto / y2k)
"""
__version__ = "0.1.0"

import os
ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
ASSETS = os.path.join(ROOT, "assets")
FONTS = os.path.join(ASSETS, "fonts")
MODELS = os.path.join(ASSETS, "models")
LUTS = os.path.join(ASSETS, "luts")
