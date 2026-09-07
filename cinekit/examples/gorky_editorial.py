# -*- coding: utf-8 -*-
"""Demo: one 9.7 s phone take -> a 20 s editorial piece.

Shows the things a hand-rolled overlay cannot do: a crop that tracks the
subject, type that passes behind her, slow motion warped along the optical
flow, a 3D metal title, and a scored soundtrack with narration.

    python3 examples/gorky_editorial.py <source.mov> [outdir]
"""
import os, sys, json, time
import numpy as np
from PIL import Image, ImageDraw

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from cinekit import FONTS
from cinekit.timeline import Grid, Timeline, band, ease
from cinekit.source import FrameStore
from cinekit.camera import Camera
from cinekit import track as tracking
from cinekit.grade import Optics
from cinekit.render import Compositor, sequence_layer, hud_layer
from cinekit import typo, shapes, audio as A, voice, scene3d, brands
from cinekit.shapes import PALETTE, rot_scale, paste_c

SRC = sys.argv[1] if len(sys.argv) > 1 else "input.mov"
OUT = sys.argv[2] if len(sys.argv) > 2 else "out"
WORK = os.path.join(OUT, "work")
os.makedirs(WORK, exist_ok=True)
W, H, FPS = 1080, 1920, 30
PRESET = brands.get("editorial")


def log(m):
    print(f"[{time.strftime('%H:%M:%S')}] {m}", flush=True)


# ----------------------------------------------------------------- picture
log("decoding source")
store = FrameStore(SRC, WORK, fps=FPS)
log(f"  {store}")

log("tracking subject")
tp = os.path.join(WORK, "track.npz")
pose = tracking.analyse(store, step=1, downscale=540, mask=True)
path = tracking.smooth(pose, key="face", window=11)
found = [r for r in pose if r["found"]]
log(f"  subject present in {len(found)}/{store.n} frames, "
    f"x drifts {min(r['face'][0] for r in found):.2f}->{max(r['face'][0] for r in found):.2f}")

g = Grid(bpm=PRESET["audio"]["bpm"], fps=FPS)   # 82 BPM -> beat 0.732 s
tl = Timeline(g)
# beats           name        src range      zoom            centre
tl.cut("open",   4, src=(0.10, 0.95), zoom=(1.34, 1.16), center=((.40, .46), (.44, .48)),
       easing="out", shake=0.9)
tl.cut("hero",   5, src=(1.05, 2.20), zoom=(1.30, 1.62), center=((.40, .44), (.40, .44)),
       easing="io", shake=1.1, track="face",
       meta=dict(aim=(0.40, 0.34), track_strength=0.92))
tl.cut("close",  4, src=(2.20, 2.85), zoom=(1.80, 2.05), center=((.30, .40), (.28, .40)),
       easing="io", shake=1.3, track="face",
       meta=dict(aim=(0.44, 0.38), track_strength=1.0))
tl.cut("turn",   3, src=(3.05, 4.30), zoom=(1.15, 1.05), center=((.50, .50), (.50, .50)),
       easing="io", shake=1.8, blur=0.85)
tl.cut("river",  5, src=(4.45, 6.20), zoom=(1.18, 1.00), center=((.50, .50), (.50, .50)),
       easing="io", shake=0.8)
tl.cut("bank",   4, src=(6.30, 7.60), zoom=(1.86, 1.70), center=((.48, .43), (.42, .42)),
       easing="io", shake=1.2)
tl.cut("wake",   4, src=(8.40, 9.60), zoom=(1.26, 1.10), center=((.50, .46), (.50, .49)),
       easing="io", shake=0.8)
log("timeline:\n" + tl.summary())

cam = Camera(store, W, H)
optics = Optics(W, H)

# ------------------------------------------------------------------- audio
log("scoring")
bpm = g.bpm
beats = tl.duration / g.beat

def chord(root, kinds, start, length, vel=72):
    return [(start, length, root + k, vel) for k in kinds]

PROG = [(45, [0, 7, 12, 16, 19]), (50, [0, 7, 12, 15, 19]),
        (52, [0, 7, 11, 14, 19]), (48, [0, 7, 12, 16, 21])]
strings, piano, harp = [], [], []
bar = 0
while bar * 4 < beats + 4:
    root, kinds = PROG[bar % 4]
    strings += chord(root, kinds, bar * 4, 4.0, 62)
    piano += chord(root + 12, kinds[:3], bar * 4, 1.6, 54)
    for j, k in enumerate(kinds):
        harp.append((bar * 4 + j * 0.5, 1.4, root + 24 + k, 44))
    bar += 1

bed_len = int((tl.duration + 2.2) * A.SR)
def render_part(notes, program, gain, fx):
    x = A.play_midi(notes, program=A.GM[program], bpm=bpm)
    x = A.apply_board(x, fx)
    out = np.zeros(bed_len, np.float32)
    A.place(out, x, 0.0, gain)
    return out

music = (render_part(strings, "strings", 0.52, A.board(reverb=0.42, lp_hz=9000, comp=(-20, 3)))
         + render_part(piano, "piano", 0.34, A.board(reverb=0.34, delay=0.16, lp_hz=13000))
         + render_part(harp, "harp", 0.20, A.board(reverb=0.46, lp_hz=11000)))

sfx = np.zeros(bed_len, np.float32)
for i, c in enumerate(tl.cuts):
    A.place(sfx, A.whoosh(0.55, seed=i, bright=0.85), c - 0.26, 0.14)
A.place(sfx, A.riser(2.0, 260, 5200), 0.10, 0.16)
A.place(sfx, A.impact(), tl.cuts[0], 0.22)
A.place(sfx, A.sub_drop(2.0), tl.duration - 3.4, 0.16)
music += sfx + A.crackle(bed_len) * 0.6

LINES = [
    dict(t=1.05, text="Москва.",                     slot=(0.0, 2.9)),
    dict(t=3.20, text="Парк Горького, сентябрь.",    slot=(2.9, 6.6)),
    dict(t=6.90, text="Солнце уходит в реку.",       slot=(6.6, 9.5)),
    dict(t=11.10, text="Город становится тёплым.",   slot=(10.6, 14.2)),
    dict(t=14.60, text="Здесь время идёт иначе.",    slot=(14.2, 17.5)),
    dict(t=17.80, text="Я запомню тебя такой.",      slot=(17.5, tl.duration)),
]
log("narrating")
vo, timing = voice.narrate(LINES, voice="anna", tempo=0.93,
                           total=tl.duration + 2.2, workdir=os.path.join(WORK, "vo"))
voice.save_timing(timing, os.path.join(WORK, "timing.json"))
for tline in timing:
    over = "  OVERRUN" if tline["end"] > tline["slot"][1] + 0.05 else ""
    log(f"  {tline['t']:5.2f}-{tline['end']:5.2f} {tline['text']}{over}")

bed = np.stack([music, music], 1)
mixed = A.duck_under(bed, vo, depth=0.46) + np.stack([vo, vo], 1) * 1.0
tail = np.clip((tl.duration + 0.6 - np.arange(len(mixed)) / A.SR) / 1.0, 0, 1)[:, None]
audio_path = A.master(mixed * tail, target_lufs=PRESET["audio"]["master"],
                      out_path=os.path.join(WORK, "mix.wav"))
log(f"  {audio_path}")

# ---------------------------------------------------------------- 3D title
log("rendering 3D title (Blender)")
# Blender runs on CPU here and EEVEE goes through software EGL, so a 1080p
# frame costs ~25 s. A bloomed metal title carries almost no high-frequency
# detail, so rendering at half size and letting the compositor scale it back
# is visually free and four times quicker.
TITLE_SCALE = 2
title_frames = list(range(0, 46))
t3d = scene3d.chrome_title(
    "МОСКВА", font_path=os.path.join(FONTS, "PlayfairDisplay900.ttf"),
    width=W // TITLE_SCALE, height=H // TITLE_SCALE, frames=title_frames,
    outdir=os.path.join(WORK, "title3d"),
    engine="BLENDER_EEVEE", samples=48, orbit=11.0, fill=0.70,
    colour=(0.90, 0.88, 0.80))
log(f"  {len(t3d)} frames at {W // TITLE_SCALE}x{H // TITLE_SCALE}")

# ---------------------------------------------------------------- graphics
subs = typo.Subtitles(timing, w=W, h=H, y=int(H * 0.735),
                      font_name="PlayfairDisplay600", size=64,
                      style=PRESET["subtitle_style"])

BIG = typo.font("PlayfairDisplay900", 178)
_behind = typo.solid("ГОРЬКОГО", BIG, PALETTE["champagne"], stroke=0,
                     shadow=(0, 0, 0, 120), shadow_off=(0, 6), tracking=10)
_kicker = typo.solid("MOSCOW · GORKY PARK", typo.font("Jost400", 30),
                     PALETTE["cream"], tracking=13)


def behind_title(canvas, t, n):
    """The word passes behind her — the shot that proves the mask."""
    a = min(band(t, 5.20, 5.90), 1 - band(t, 8.20, 8.90))
    if a <= 0.02:
        return
    u = band(t, 5.20, 8.90)
    paste_c(canvas, rot_scale(_behind, 0, 1.0, a),
            W / 2 + (0.5 - u) * 130, H * 0.40)


def kicker(canvas, t, n):
    a = min(band(t, 9.60, 10.30), 1 - band(t, 13.4, 14.1))
    if a <= 0.02:
        return
    paste_c(canvas, rot_scale(_kicker, 0, 1.0, a * 0.9), W / 2, H * 0.245)
    d = ImageDraw.Draw(canvas)
    ry = int(H * 0.268)
    half = int(190 * ease("out", band(t, 9.70, 10.60)))
    d.line([(W / 2 - half, ry), (W / 2 + half, ry)],
           fill=PALETTE["cream"] + (int(190 * a),), width=2)


_end = typo.luxury_card("Парк Горького", "москва · закат", w=W, h=H, y=int(H * 0.44),
                        serif="PlayfairDisplay900", sans="Jost400")


def end_card(canvas, t, n):
    a = min(band(t, tl.duration - 4.0, tl.duration - 3.3),
            1 - band(t, tl.duration - 0.55, tl.duration - 0.05))
    if a <= 0.02:
        return
    canvas.alpha_composite(rot_scale(_end, 0, 1.0, a))


comp = Compositor(tl, cam, look=PRESET["grade"], track=path, pose=pose,
                  optics=optics, retime="flow", pulse=PRESET["motion"]["pulse"],
                  optics_kw=dict(grain=PRESET["optics"]["grain"],
                                 aberration=PRESET["optics"]["aberration"],
                                 leak=PRESET["optics"]["leak"]))
comp.back(behind_title, "title-behind-subject")
comp.front(sequence_layer(t3d, start_frame=2, w=W, h=H, fade=(0.30, 0.45), fps=FPS,
                          scale=TITLE_SCALE),
           "3d-title")
comp.front(kicker, "kicker")
comp.front(subs.draw, "subtitles")
comp.front(end_card, "end-card")
comp.front(hud_layer(W, H, "REC", accent=PALETTE["cream"], font_name="JetBrainsMono400",
                     size=24, show=(0.5, tl.duration - 4.2), fps=FPS), "hud")

if "--preview" in sys.argv:
    marks = [8, 40, 80, 130, 170, 200, 235, 265, 300, 340, 380, 420,
             460, 500, 540, 575, 600]
    log("preview -> " + comp.preview(marks, os.path.join(OUT, "preview.jpg"), cols=6))
else:
    log(f"rendering {tl.frames} frames")
    out = comp.render(os.path.join(OUT, "gorky_editorial.mp4"), audio=audio_path,
                      crf=19, preset="slow")
    log("done -> " + out)
