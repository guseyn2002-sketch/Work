# -*- coding: utf-8 -*-
"""Typography: the layer that most reads as "who made this".

Two registers are covered. Loud — chrome fills, thick keylines, per-word
colour, elastic pops. Quiet — wide letterspacing, hairlines, slow fades, which
is what the luxury houses actually do.
"""
import os, json, math
import numpy as np
from PIL import Image, ImageDraw, ImageFont, ImageFilter

from . import FONTS
from .shapes import chrome_ramp, gold_ramp, PALETTE, rot_scale, paste_c
from .timeline import band, ease

_fc = {}
_catalog = None


def catalog():
    global _catalog
    if _catalog is None:
        p = os.path.join(FONTS, "catalog.json")
        _catalog = json.load(open(p)) if os.path.exists(p) else []
    return _catalog


def font(name, size):
    """Load by file stem (e.g. 'PlayfairDisplay900')."""
    k = (name, int(size))
    if k not in _fc:
        path = name if os.path.isabs(name) else os.path.join(FONTS, name + ".ttf")
        _fc[k] = ImageFont.truetype(path, int(size))
    return _fc[k]


def pick(tag=None, weight=None, cyrillic=None, family=None):
    """Choose a face from the catalogue by register."""
    rows = catalog()
    if family:
        rows = [r for r in rows if r["family"].lower() == family.lower()]
    if tag:
        rows = [r for r in rows if tag in r["tags"]]
    if cyrillic is not None:
        rows = [r for r in rows if r["cyrillic"] == cyrillic]
    if weight:
        exact = [r for r in rows if r["weight"] == str(weight)]
        rows = exact or sorted(rows, key=lambda r: abs(int(r["weight"]) - int(weight)))
    if not rows:
        raise LookupError(f"no font for tag={tag} weight={weight} cyrillic={cyrillic}")
    return os.path.splitext(rows[0]["file"])[0]


# ------------------------------------------------------------------ atoms
def _bbox(txt, fnt, stroke=0, tracking=0):
    d = ImageDraw.Draw(Image.new("L", (8, 8)))
    if tracking:
        w = sum(d.textbbox((0, 0), c, font=fnt, stroke_width=stroke)[2] for c in txt)
        w += tracking * max(0, len(txt) - 1)
        h = d.textbbox((0, 0), txt or "X", font=fnt, stroke_width=stroke)[3]
        return (0, 0, int(w), int(h))
    return d.textbbox((0, 0), txt, font=fnt, stroke_width=stroke)


def _draw_tracked(d, xy, txt, fnt, tracking, **kw):
    x, y = xy
    for ch in txt:
        d.text((x, y), ch, font=fnt, **kw)
        x += d.textbbox((0, 0), ch, font=fnt,
                        stroke_width=kw.get("stroke_width", 0))[2] + tracking


def text_mask(txt, fnt, stroke=0, tracking=0):
    box = _bbox(txt, fnt, stroke, tracking)
    w = box[2] - box[0] + 2 * stroke + 8
    h = box[3] - box[1] + 2 * stroke + 8
    m = Image.new("L", (max(w, 1), max(h, 1)), 0)
    d = ImageDraw.Draw(m)
    pos = (-box[0] + stroke + 4, -box[1] + stroke + 4)
    if tracking:
        _draw_tracked(d, pos, txt, fnt, tracking, fill=255,
                      stroke_width=stroke, stroke_fill=255)
    else:
        d.text(pos, txt, font=fnt, fill=255, stroke_width=stroke, stroke_fill=255)
    return m


def solid(txt, fnt, fill, stroke=0, stroke_fill=(0, 0, 0), shadow=None,
          shadow_off=(0, 0), glow=None, glow_r=14, tracking=0):
    """Filled type with keyline, drop shadow and optional colour bloom."""
    box = _bbox(txt, fnt, stroke, tracking)
    pad = int(glow_r * 2 + max(abs(shadow_off[0]), abs(shadow_off[1])) + 10)
    W = box[2] - box[0] + 2 * stroke + 2 * pad
    H = box[3] - box[1] + 2 * stroke + 2 * pad
    layer = Image.new("RGBA", (W, H), (0, 0, 0, 0))
    ox, oy = -box[0] + stroke + pad, -box[1] + stroke + pad

    def put(target, off, colour, sw):
        d = ImageDraw.Draw(target)
        pos = (ox + off[0], oy + off[1])
        if tracking:
            _draw_tracked(d, pos, txt, fnt, tracking, fill=colour,
                          stroke_width=sw, stroke_fill=colour)
        else:
            d.text(pos, txt, font=fnt, fill=colour, stroke_width=sw, stroke_fill=colour)

    if glow:
        g = Image.new("RGBA", (W, H), (0, 0, 0, 0))
        put(g, (0, 0), glow + (255,), stroke + 3)
        g = g.filter(ImageFilter.GaussianBlur(glow_r))
        layer = Image.alpha_composite(Image.alpha_composite(layer, g), g)
    if shadow:
        s = Image.new("RGBA", (W, H), (0, 0, 0, 0))
        put(s, shadow_off, shadow, stroke)
        layer = Image.alpha_composite(layer, s.filter(ImageFilter.GaussianBlur(2.0)))

    t = Image.new("RGBA", (W, H), (0, 0, 0, 0))
    d = ImageDraw.Draw(t)
    col = fill + (255,) if len(fill) == 3 else fill
    if tracking:
        _draw_tracked(d, (ox, oy), txt, fnt, tracking, fill=col,
                      stroke_width=stroke, stroke_fill=stroke_fill)
    else:
        d.text((ox, oy), txt, font=fnt, fill=col, stroke_width=stroke,
               stroke_fill=stroke_fill)
    return Image.alpha_composite(layer, t)


def metal(txt, fnt, ramp="chrome", stroke=0, stroke_fill=(20, 20, 40), outer=None,
          glow=(140, 200, 255), glow_r=22, phase=0.0, tracking=0):
    """Chrome or gold display type with keyline and bloom."""
    m = text_mask(txt, fnt, stroke, tracking)
    W, H = m.size
    pad = int(glow_r * 2 + 12)
    canvas = Image.new("RGBA", (W + 2 * pad, H + 2 * pad), (0, 0, 0, 0))

    if glow:
        g = Image.new("RGBA", canvas.size, (0, 0, 0, 0))
        g.paste(glow + (255,), (pad, pad), m.filter(ImageFilter.MaxFilter(5)))
        g = g.filter(ImageFilter.GaussianBlur(glow_r))
        canvas = Image.alpha_composite(Image.alpha_composite(canvas, g), g)
    if outer:
        om = text_mask(txt, fnt, stroke + outer[1], tracking)
        o = Image.new("RGBA", canvas.size, (0, 0, 0, 0))
        o.paste(outer[0] + (255,), (pad - (om.size[0] - W) // 2,
                                    pad - (om.size[1] - H) // 2), om)
        canvas = Image.alpha_composite(canvas, o)
    if stroke:
        s = Image.new("RGBA", canvas.size, (0, 0, 0, 0))
        s.paste(stroke_fill + (255,), (pad, pad), text_mask(txt, fnt, stroke, tracking))
        canvas = Image.alpha_composite(canvas, s)

    core = text_mask(txt, fnt, 0, tracking)
    grad = (gold_ramp if ramp == "gold" else chrome_ramp)(core.size, phase).convert("RGBA")
    grad.putalpha(core)
    out = canvas.copy()
    out.alpha_composite(grad, (pad + (W - core.size[0]) // 2,
                               pad + (H - core.size[1]) // 2))
    return out


# --------------------------------------------------------------- subtitles
class Subtitles:
    """Per-word karaoke captions: each word gets a colour and pops on its cue."""

    def __init__(self, lines, w=1080, h=1920, y=None, font_name=None, size=88,
                 max_width=None, palette=None, style="pop", line_height=None):
        self.W, self.H = w, h
        self.y = y if y is not None else int(h * 0.67)
        self.size = size
        self.max_width = max_width or int(w * 0.83)
        self.line_height = line_height or int(size * 1.32)
        self.style = style
        self.font_name = font_name or pick(tag="display", weight=900, cyrillic=True)
        self.palette = palette or [PALETTE[k] for k in
                                   ("yellow", "cyan", "pink", "lime", "lilac",
                                    "orange", "mint", "hot")]
        self.blocks = [self._layout(i, l) for i, l in enumerate(lines)]

    def _layout(self, idx, line):
        fnt = font(self.font_name, self.size)
        d = ImageDraw.Draw(Image.new("L", (8, 8)))
        words = line["words"]
        widths = [d.textbbox((0, 0), w["w"], font=fnt, stroke_width=10)[2] for w in words]
        space = int(self.size * 0.30)

        rows, cur, curw = [], [], 0
        for i, _ in enumerate(words):
            add = widths[i] + (space if cur else 0)
            if cur and curw + add > self.max_width:
                rows.append(cur); cur, curw = [], 0; add = widths[i]
            cur.append(i); curw += add
        if cur:
            rows.append(cur)

        total = len(rows) * self.line_height
        placed = []
        for r, row in enumerate(rows):
            roww = sum(widths[i] for i in row) + space * (len(row) - 1)
            x = self.W / 2 - roww / 2
            y = self.y - total / 2 + r * self.line_height + self.line_height / 2
            for i in row:
                col = self.palette[(idx * 3 + i) % len(self.palette)]
                if self.style == "quiet":
                    lay = solid(words[i]["w"].upper(), fnt, PALETTE["cream"], stroke=0,
                                shadow=(0, 0, 0, 150), shadow_off=(0, 3), tracking=6)
                else:
                    lay = solid(words[i]["w"], fnt, col, stroke=int(self.size * 0.125),
                                stroke_fill=PALETTE["ink"], shadow=(0, 0, 0, 165),
                                shadow_off=(4, 7), glow=col, glow_r=int(self.size * 0.19))
                placed.append(dict(x=x + widths[i] / 2, y=y, lay=lay,
                                   t0=words[i]["t0"], t1=words[i]["t1"],
                                   tilt=(-3.2 if i % 2 == 0 else 3.0)))
                x += widths[i] + space
        first = words[0]["t0"]
        last = words[-1]["t1"]
        return dict(placed=placed, show0=first - 0.20,
                    show1=min(line.get("slot", (0, 1e9))[1] - 0.06, last + 0.95))

    def draw(self, canvas, t):
        for blk in self.blocks:
            if not (blk["show0"] <= t <= blk["show1"] + 0.30):
                continue
            out_u = band(t, blk["show1"], blk["show1"] + 0.26)
            for p in blk["placed"]:
                if t < p["t0"] - 0.14:
                    continue
                u = band(t, p["t0"] - 0.14, p["t0"] + 0.16)
                if self.style == "quiet":
                    sc, a = 1.0, ease("out", u)
                    tilt = 0.0
                else:
                    sc = 0.35 + 0.65 * ease("elastic", u)
                    if p["t0"] <= t <= p["t1"] + 0.10:
                        k = band(t, p["t0"], p["t1"] + 0.10)
                        sc *= 1.0 + 0.16 * math.sin(math.pi * k) ** 0.6
                    a = 1.0
                    tilt = p["tilt"] * (1.0 - 0.6 * u)
                if out_u > 0:
                    a *= 1 - out_u
                    sc *= 1 - 0.22 * out_u
                if a <= 0.02:
                    continue
                paste_c(canvas, rot_scale(p["lay"], tilt, sc * 0.95, a),
                        p["x"], p["y"] - 30 * (1 - u))


# ------------------------------------------------------------- title cards
def luxury_card(title, sub=None, w=1080, h=1920, y=None, serif=None, sans=None,
                colour=None, rule_w=360):
    """Wide-tracked caps over a hairline: the fashion-house lockup."""
    y = y if y is not None else int(h * 0.46)
    colour = colour or PALETTE["cream"]
    serif = serif or pick(tag="luxury", cyrillic=True)
    sans = sans or pick(tag="luxury", cyrillic=True)
    lay = Image.new("RGBA", (w, h), (0, 0, 0, 0))
    t = solid(title.upper(), font(serif, int(h * 0.052)), colour,
              shadow=(0, 0, 0, 120), shadow_off=(0, 3), tracking=int(h * 0.011))
    paste_c(lay, t, w / 2, y)
    d = ImageDraw.Draw(lay)
    ry = y + int(h * 0.045)
    d.line([(w / 2 - rule_w / 2, ry), (w / 2 + rule_w / 2, ry)], fill=colour + (210,), width=2)
    if sub:
        s = solid(sub.upper(), font(sans, int(h * 0.017)), colour,
                  shadow=(0, 0, 0, 110), shadow_off=(0, 2), tracking=int(h * 0.008))
        paste_c(lay, s, w / 2, ry + int(h * 0.032))
    return lay
