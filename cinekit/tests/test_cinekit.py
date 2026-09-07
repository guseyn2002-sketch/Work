# -*- coding: utf-8 -*-
"""Smoke tests. No network, no source footage, a few seconds to run.

    python3 tests/test_cinekit.py
"""
import os, sys, math, tempfile, traceback
import numpy as np
from PIL import Image

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

RESULTS = []


def check(name):
    def deco(fn):
        def run():
            try:
                fn()
                RESULTS.append((name, None))
            except Exception as e:
                RESULTS.append((name, f"{type(e).__name__}: {e}"))
        run.__name__ = fn.__name__
        return run
    return deco


@check("timeline: beats, cuts and easing")
def t_timeline():
    from cinekit.timeline import Grid, Timeline, ease, spring
    g = Grid(bpm=120, fps=30)
    assert abs(g.beat - 0.5) < 1e-9
    assert g.frame(g.b(4)) == 60
    tl = Timeline(g)
    tl.cut("a", 4, src=(0, 1), zoom=(1.0, 1.2))
    tl.cut("b", 4, src=(1, 2))
    assert abs(tl.duration - 4.0) < 1e-9 and tl.cuts == [2.0]
    assert tl.at(0.1).name == "a" and tl.at(2.1).name == "b"
    st = tl.at(0.0).state(2.0 - 1e-6)
    assert 1.19 < st["zoom"] <= 1.2, st["zoom"]
    for n in ("linear", "in", "out", "io", "expo", "back", "elastic", "bounce"):
        assert abs(ease(n, 0.0)) < 1e-6 and abs(ease(n, 1.0) - 1.0) < 1e-6, n
    assert 0.0 <= spring(0.0) < 1e-9 and abs(spring(3.0) - 1.0) < 0.02
    # the beat pulse must peak on the beat and relax between beats
    assert g.pulse(0.0, 0.02) > g.pulse(0.25, 0.02)


@check("grade: LUT round-trip and identity")
def t_grade():
    from cinekit.grade import build_lut, load_cube, apply_lut, LOOKS
    with tempfile.TemporaryDirectory() as d:
        p = build_lut("kodak", size=17, path=os.path.join(d, "k.cube"))
        lut = load_cube(p)
        assert lut.shape == (17, 17, 17, 3)
        # an identity LUT must leave pixels alone
        n = 17
        g = np.linspace(0, 1, n)
        b, gg, r = np.meshgrid(g, g, g, indexing="ij")
        ident = np.stack([r, gg, b], -1).astype(np.float32)
        img = (np.random.default_rng(0).random((24, 32, 3)) * 255).astype(np.float32)
        out = apply_lut(img, ident)
        assert np.abs(out - img).max() < 1.5, np.abs(out - img).max()
        # every named look must build
        for k in LOOKS:
            build_lut(k, size=9, path=os.path.join(d, f"{k}.cube"))


@check("grade: optics preserve shape and respond to grain")
def t_optics():
    from cinekit.grade import Optics, bloom
    o = Optics(64, 96, grain_tiles=2)
    a = np.full((96, 64, 3), 120.0, np.float32)
    flat = o.apply(a.copy(), 0, grain=0.0, leak=0.0)
    noisy = o.apply(a.copy(), 0, grain=0.5, leak=0.0)
    assert flat.shape == a.shape
    assert noisy.std() > flat.std()
    assert flat[48, 32].mean() > flat[2, 2].mean()          # vignette darkens edges
    assert bloom(a).shape == a.shape


@check("shapes and typography render non-empty alpha")
def t_gfx():
    from cinekit import shapes, typo
    for f in (shapes.sparkle(20), shapes.star5(20), shapes.heart(20),
              shapes.butterfly(24), shapes.ring(20, dash=6)):
        assert f.getchannel("A").getbbox() is not None
    assert shapes.chrome_ramp((32, 32)).size == (32, 32)
    assert shapes.gold_ramp((32, 32)).size == (32, 32)
    fnt = typo.font(typo.pick(tag="display", weight=900), 40)
    for lay in (typo.solid("Тест", fnt, (255, 255, 255), stroke=4),
                typo.metal("Тест", fnt, ramp="gold", stroke=3),
                typo.metal("Тест", fnt, ramp="chrome", stroke=3)):
        assert lay.getchannel("A").getbbox() is not None
    assert typo.solid("АБВ", fnt, (255, 255, 255), tracking=12).width > \
           typo.solid("АБВ", fnt, (255, 255, 255), tracking=0).width
    assert typo.have(typo.pick(tag="luxury", cyrillic=True))


@check("typography: missing weight names the available ones")
def t_font_guard():
    from cinekit import typo
    try:
        typo.font("PlayfairDisplay137", 20)
    except FileNotFoundError as e:
        assert "PlayfairDisplay" in str(e), str(e)
    else:
        raise AssertionError("expected FileNotFoundError")


@check("subtitles: per-word layout and karaoke timing")
def t_subs():
    from cinekit import typo
    timing = [dict(idx=0, t=0.0, end=1.6, style="pop", slot=(0.0, 3.0),
                   text="Один два три четыре пять",
                   words=[dict(w=w, t0=i * 0.3, t1=i * 0.3 + 0.28)
                          for i, w in enumerate("Один два три четыре пять".split())])]
    s = typo.Subtitles(timing, w=540, h=960, size=44, max_width=460)
    assert len(s.blocks[0]["placed"]) == 5
    ys = sorted({round(p["y"]) for p in s.blocks[0]["placed"]})
    assert len(ys) >= 2, "long line should wrap onto more than one row"
    canvas = Image.new("RGBA", (540, 960), (0, 0, 0, 0))
    s.draw(canvas, 0.05)
    early = canvas.getchannel("A").getbbox()
    canvas2 = Image.new("RGBA", (540, 960), (0, 0, 0, 0))
    s.draw(canvas2, 1.4)
    late = np.asarray(canvas2)[..., 3].sum()
    assert early is not None and late > np.asarray(canvas)[..., 3].sum()


@check("transitions keep shape and stay in range")
def t_transitions():
    from cinekit import transitions as tr
    a = np.random.default_rng(1).random((48, 32, 3)).astype(np.float32) * 255
    b = np.zeros_like(a)
    for out in (tr.whip(a, .5), tr.glitch(a, .5), tr.burn(a, .5), tr.dip(a, .5),
                tr.zoom_blur(a, .5), tr.push(a, b, .5)):
        assert out.shape == a.shape and np.isfinite(out).all()
    assert tr.dip(a, 0.5).mean() < a.mean()                 # dip to black darkens
    assert tr.flash(a.copy(), 1.0, [1.0], strength=0.5).mean() > a.mean()


@check("audio: design elements and mastering")
def t_audio():
    from cinekit import audio as A
    for f in (A.whoosh(0.2), A.riser(0.3), A.impact(0.3), A.sparkle(0.3),
              A.shutter(), A.sub_drop(0.3)):
        assert len(f) > 100 and np.isfinite(f).all() and np.abs(f).max() > 0
    kit = A.drum_kit()
    assert set(kit) == {"kick", "snare", "hat", "hat_open", "clap"}
    x = np.sin(2 * np.pi * 220 * A.t_axis(A.SR // 2)).astype(np.float32)
    m = A.master(x)
    assert m.ndim == 2 and np.abs(m).max() <= 1.0
    d = A.duck_under(np.stack([x, x], 1), np.abs(x))
    assert d.shape == (len(x), 2) and d.max() <= np.abs(x).max() + 1e-6
    sc = A.sidechain(A.SR, [0.0, 0.5], depth=0.5)
    assert abs(sc[0] - 0.5) < 1e-6, f"duck floor should be 1-depth, got {sc[0]:.3f}"
    assert sc[int(0.45 * A.SR)] > 0.9


@check("brands: presets are complete")
def t_brands():
    from cinekit import brands
    from cinekit.grade import LOOKS
    for n in brands.names():
        p = brands.get(n)
        for k in ("grade", "display", "body", "optics", "audio", "motion", "note"):
            assert k in p, f"{n} missing {k}"
        assert p["grade"] in LOOKS, f"{n}: unknown grade {p['grade']}"
        assert p["subtitle_style"] in ("pop", "quiet")


@check("camera: geometry is reusable and mask follows the crop")
def t_camera():
    from cinekit.timeline import Grid, Timeline
    from cinekit.camera import Camera

    class FakeStore:
        width, height, fps, n = 400, 800, 30, 30
        def sample(self, t, scale=1.0, mode="flow"):
            im = Image.new("RGB", (400, 800), (30, 30, 30))
            im.paste((240, 40, 40), (150, 300, 250, 500))    # a marker block
            return im

    tl = Timeline(Grid(100, 30))
    tl.cut("a", 4, src=(0, 1), zoom=(2.0, 2.0), center=((0.5, 0.5), (0.5, 0.5)))
    cam = Camera(FakeStore(), 100, 200)
    st = tl.shots[0].state(0.0)
    geo = cam.geometry(st, 0)
    frame = cam.frame(st, 0, geo=geo)
    assert frame.size == (100, 200)
    # a mask covering exactly the marker must land on the marker in output space
    m = np.zeros((800, 400), np.float32)
    m[300:500, 150:250] = 1.0
    out_mask = cam.mask_frame(m, geo)
    assert out_mask.shape == (200, 100, 1)
    arr = np.asarray(frame, np.float32)
    red = arr[..., 0] > 150
    assert (out_mask[..., 0] > 0.5).sum() > 0
    overlap = (red & (out_mask[..., 0] > 0.5)).sum() / max(1, red.sum())
    assert overlap > 0.85, f"mask and picture disagree ({overlap:.2f})"


@check("compositor: back layers land behind the subject")
def t_behind():
    from cinekit.timeline import Grid, Timeline
    from cinekit.camera import Camera
    from cinekit.render import Compositor
    from PIL import ImageDraw

    class FakeStore:
        width, height, fps, n = 200, 400, 30, 30
        def sample(self, t, scale=1.0, mode="flow"):
            im = Image.new("RGB", (200, 400), (20, 20, 20))
            im.paste((0, 240, 0), (60, 120, 140, 280))       # the "subject"
            return im

    mask = np.zeros((400, 200), np.float32)
    mask[120:280, 60:140] = 1.0
    pose = [dict(i=i + 1, t=i / 30, found=True, face=(0.5, 0.4), body=(0.5, 0.6),
                 scale=0.3, mask=mask) for i in range(30)]

    tl = Timeline(Grid(100, 30))
    tl.cut("a", 4, src=(0, 0.5), zoom=(1.0, 1.0))
    cam = Camera(FakeStore(), 200, 400)

    def band_layer(canvas, t, n):
        ImageDraw.Draw(canvas).rectangle([0, 180, 200, 220], fill=(255, 0, 255, 255))

    c = Compositor(tl, cam, look=None, pose=pose,
                   optics_kw=dict(grain=0.0, leak=0.0, vignette=False,
                                  scan=False, aberration=None))
    c.back(band_layer)
    f = c.frame(0).astype(int)
    # inside the subject the magenta band must be hidden; outside it must show
    inside = f[200, 100]
    outside = f[200, 10]
    assert inside[1] > 150 and inside[0] < 120, f"subject was covered: {inside}"
    assert outside[0] > 180 and outside[2] > 180, f"band missing outside: {outside}"

    # with the same layer in front, the subject must be covered
    c2 = Compositor(tl, cam, look=None, pose=pose,
                    optics_kw=dict(grain=0.0, leak=0.0, vignette=False,
                                   scan=False, aberration=None))
    c2.front(band_layer)
    assert c2.frame(0)[200, 100][0] > 180, "front layer should cover the subject"


@check("track: smoothing rejects outliers and fills gaps")
def t_smooth():
    from cinekit import track as T
    rec = []
    for i in range(40):
        x = 0.30 + i * 0.004
        found = True
        if i == 20:
            x = 0.95              # a detector spike
        if i in (25, 26):
            found = False         # a dropout
        rec.append(dict(i=i + 1, t=i / 30, found=found,
                        face=(x, 0.40) if found else None,
                        body=(x, 0.6) if found else None, scale=0.3, mask=None))
    s = T.smooth(rec, key="face", window=7)
    xs = s["x"]
    assert np.isfinite(xs).all()
    assert xs.max() < 0.6, f"outlier survived: {xs.max():.2f}"
    step = np.abs(np.diff(xs)).max()
    assert step < 0.05, f"path is not smooth: {step:.3f}"
    ax, ay = s["at"](0.0)
    assert 0.25 < ax < 0.45 and 0.3 < ay < 0.5


@check("source: probe reads rotation and geometry")
def t_probe():
    from cinekit.source import probe
    import subprocess, tempfile
    with tempfile.TemporaryDirectory() as d:
        p = os.path.join(d, "t.mp4")
        subprocess.run(["ffmpeg", "-v", "error", "-f", "lavfi",
                        "-i", "testsrc=size=64x128:rate=30:duration=1",
                        "-c:v", "libx264", "-pix_fmt", "yuv420p", "-y", p], check=True)
        info = probe(p)
        assert info["width"] == 64 and info["height"] == 128
        assert abs(info["fps"] - 30) < 0.01
        assert 0.9 < info["duration"] < 1.2


def main():
    tests = [v for k, v in sorted(globals().items()) if k.startswith("t_")]
    for t in tests:
        t()
    width = max(len(n) for n, _ in RESULTS)
    failed = 0
    for name, err in RESULTS:
        if err:
            failed += 1
            print(f"FAIL  {name:<{width}}  {err}")
        else:
            print(f"ok    {name}")
    print(f"\n{len(RESULTS) - failed}/{len(RESULTS)} passed")
    return 1 if failed else 0


if __name__ == "__main__":
    sys.exit(main())
