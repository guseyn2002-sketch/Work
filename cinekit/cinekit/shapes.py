# -*- coding: utf-8 -*-
"""Vector stickers and frames, drawn rather than sourced, so they scale."""
import math
import numpy as np
from PIL import Image, ImageDraw, ImageFilter

PALETTE = {
    "pink": (255, 74, 178), "hot": (255, 40, 138), "cyan": (60, 226, 255),
    "blue": (90, 150, 255), "lime": (176, 255, 74), "yellow": (255, 214, 59),
    "orange": (255, 138, 61), "lilac": (186, 137, 255), "mint": (140, 255, 214),
    "white": (255, 255, 255), "cream": (255, 246, 232), "ink": (24, 12, 40),
    "gold": (212, 175, 55), "champagne": (240, 227, 199), "noir": (16, 16, 18),
}


def chrome_ramp(size, phase=0.0):
    """Y2K liquid chrome: sky -> white -> steel -> rose."""
    w, h = size
    stops = [(0.00, (168, 206, 255)), (0.18, (232, 245, 255)), (0.34, (255, 255, 255)),
             (0.47, (206, 222, 245)), (0.58, (120, 150, 205)), (0.68, (236, 214, 255)),
             (0.80, (255, 190, 232)), (0.90, (255, 236, 246)), (1.00, (176, 196, 232))]
    ys = np.linspace(0, 1, h)
    ramp = np.zeros((h, 3), np.float32)
    for i in range(len(stops) - 1):
        (p0, c0), (p1, c1) = stops[i], stops[i + 1]
        m = (ys >= p0) & (ys <= p1)
        if not m.any():
            continue
        u = ((ys[m] - p0) / max(p1 - p0, 1e-6))[:, None]
        ramp[m] = np.array(c0)[None] * (1 - u) + np.array(c1)[None] * u
    img = np.repeat(ramp[:, None, :], w, axis=1)
    xx, yy = np.meshgrid(np.linspace(0, 1, w), np.linspace(0, 1, h))
    sheen = np.clip(1 - np.abs((xx * .6 + yy * .4) - (0.35 + phase)) * 3.2, 0, 1) ** 2
    return Image.fromarray(np.clip(img + sheen[..., None] * 55, 0, 255).astype(np.uint8))


def gold_ramp(size, phase=0.0):
    """Brushed gold for luxury titling."""
    w, h = size
    stops = [(0.0, (120, 88, 26)), (0.22, (214, 176, 92)), (0.38, (255, 240, 200)),
             (0.52, (196, 154, 62)), (0.70, (255, 232, 168)), (0.86, (162, 118, 40)),
             (1.0, (226, 192, 112))]
    ys = np.linspace(0, 1, h)
    ramp = np.zeros((h, 3), np.float32)
    for i in range(len(stops) - 1):
        (p0, c0), (p1, c1) = stops[i], stops[i + 1]
        m = (ys >= p0) & (ys <= p1)
        if not m.any():
            continue
        u = ((ys[m] - p0) / max(p1 - p0, 1e-6))[:, None]
        ramp[m] = np.array(c0)[None] * (1 - u) + np.array(c1)[None] * u
    img = np.repeat(ramp[:, None, :], w, axis=1)
    xx, yy = np.meshgrid(np.linspace(0, 1, w), np.linspace(0, 1, h))
    sheen = np.clip(1 - np.abs((xx * .7 + yy * .3) - (0.4 + phase)) * 4.0, 0, 1) ** 2
    return Image.fromarray(np.clip(img + sheen[..., None] * 60, 0, 255).astype(np.uint8))


def sparkle(size, color=(255, 255, 255), waist=0.20, glow=True):
    S = int(size)
    im = Image.new("RGBA", (S * 2, S * 2), (0, 0, 0, 0))
    d = ImageDraw.Draw(im)
    c, r, w = S, S * .94, S * waist
    d.polygon([(c, c - r), (c + w * .62, c - w * .62), (c + r, c), (c + w * .62, c + w * .62),
               (c, c + r), (c - w * .62, c + w * .62), (c - r, c), (c - w * .62, c - w * .62)],
              fill=color + (255,))
    if not glow:
        return im
    g = im.filter(ImageFilter.GaussianBlur(S * .22))
    out = Image.alpha_composite(Image.alpha_composite(
        Image.new("RGBA", im.size, (0, 0, 0, 0)), g), g)
    return Image.alpha_composite(out, im)


def star5(size, color=(255, 214, 59), outline=(255, 255, 255)):
    S = int(size)
    im = Image.new("RGBA", (S * 2, S * 2), (0, 0, 0, 0))
    d = ImageDraw.Draw(im)
    pts = []
    for i in range(10):
        a = -math.pi / 2 + i * math.pi / 5
        rad = S * .92 if i % 2 == 0 else S * .40
        pts.append((S + rad * math.cos(a), S + rad * math.sin(a)))
    d.polygon(pts, fill=color + (255,), outline=outline + (255,), width=max(2, S // 12))
    return im


def heart(size, color=(255, 74, 178), outline=(255, 255, 255)):
    S = int(size)
    im = Image.new("RGBA", (S * 2, S * 2), (0, 0, 0, 0))
    pts = []
    for i in range(160):
        t = i / 159 * 2 * math.pi
        x = 16 * math.sin(t) ** 3
        y = -(13 * math.cos(t) - 5 * math.cos(2 * t) - 2 * math.cos(3 * t) - math.cos(4 * t))
        pts.append((S + x * S / 18.5, S + y * S / 18.5))
    ImageDraw.Draw(im).polygon(pts, fill=color + (255,), outline=outline + (255,),
                               width=max(2, S // 11))
    return im


def butterfly(size, c1=(255, 110, 200), c2=(120, 210, 255)):
    S = int(size)
    im = Image.new("RGBA", (S * 2, S * 2), (0, 0, 0, 0))
    d = ImageDraw.Draw(im)
    cx = cy = S

    def wing(scale, dy, col, tilt):
        pts = []
        for i in range(80):
            a = i / 79 * math.pi
            rr = S * scale * (.45 + .55 * math.sin(a) ** .75)
            pts.append((cx + math.cos(a - tilt) * rr,
                        cy - math.sin(a - tilt) * rr * .86 + dy))
        pts.append((cx, cy + dy * .4))
        d.polygon(pts, fill=col + (235,))
        d.polygon([(2 * cx - x, y) for x, y in pts], fill=col + (235,))

    wing(.92, -S * .10, c1, .30)
    wing(.62, S * .34, c2, -.42)
    d.ellipse([cx - S * .055, cy - S * .46, cx + S * .055, cy + S * .52], fill=(60, 40, 80, 255))
    for s in (-1, 1):
        d.line([(cx, cy - S * .42), (cx + s * S * .30, cy - S * .80)],
               fill=(60, 40, 80, 255), width=max(2, S // 22))
    return Image.alpha_composite(im.filter(ImageFilter.GaussianBlur(S * .10)), im)


def ring(size, color=(255, 255, 255), width=4, dash=None):
    S = int(size)
    im = Image.new("RGBA", (S * 2, S * 2), (0, 0, 0, 0))
    d = ImageDraw.Draw(im)
    box = [width, width, S * 2 - width, S * 2 - width]
    if dash:
        for i in range(dash):
            a0 = i * 360 / dash
            d.arc(box, a0, a0 + 360 / dash * .55, fill=color + (255,), width=width)
    else:
        d.ellipse(box, outline=color + (255,), width=width)
    return im


def rule(length, thickness=3, color=(255, 255, 255)):
    """A hairline — the quiet luxury separator."""
    im = Image.new("RGBA", (int(length), max(1, int(thickness))), color + (255,))
    return im


def frame_corners(w, h, inset=48, arm=110, width=5, color=(255, 255, 255)):
    """Four corner brackets: viewfinder / editorial register."""
    im = Image.new("RGBA", (w, h), (0, 0, 0, 0))
    d = ImageDraw.Draw(im)
    c = color + (255,)
    for sx, x0 in ((1, inset), (-1, w - inset)):
        for sy, y0 in ((1, inset), (-1, h - inset)):
            d.line([(x0, y0), (x0 + sx * arm, y0)], fill=c, width=width)
            d.line([(x0, y0), (x0, y0 + sy * arm)], fill=c, width=width)
    return im


def rot_scale(im, deg=0.0, scale=1.0, alpha=1.0):
    if im is None or scale <= 0.01 or alpha <= 0.004:
        return None
    o = im.resize((max(1, int(im.width * scale)), max(1, int(im.height * scale))),
                  Image.BICUBIC)
    if abs(deg) > 0.2:
        o = o.rotate(deg, resample=Image.BICUBIC, expand=True)
    if alpha < 0.999:
        o.putalpha(o.getchannel("A").point(lambda v: int(v * max(0.0, min(1.0, alpha)))))
    return o


def paste_c(dst, src, cx, cy):
    """Alpha-composite src centred at (cx, cy)."""
    if src is None:
        return
    dst.alpha_composite(src, (int(cx - src.width / 2), int(cy - src.height / 2)))
