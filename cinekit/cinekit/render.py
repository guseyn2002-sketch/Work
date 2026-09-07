# -*- coding: utf-8 -*-
"""The pipeline driver: picture, grade, optics, graphics, output.

The layer model is the reason this file exists. Graphics tagged `back` are
drawn over the graded picture and then the tracked subject is restored on top
of them, so type passes *behind* a person. That single move is most of the
distance between an overlay and a composite.
"""
import os, math, subprocess, sys
import numpy as np
from PIL import Image

from .grade import apply_lut, look_path, Optics, bloom
from .timeline import band


class Layer:
    def __init__(self, fn, z="front", name=""):
        self.fn, self.z, self.name = fn, z, name


class Compositor:
    """Assemble frames from a Timeline and a Camera.

    look             name of a grade preset, or None to leave colour alone
    track            smoothed subject path (track.smooth output) for reframing
    pose             raw track records (track.analyse output) for the mask
    """

    def __init__(self, timeline, camera, look=None, track=None, pose=None,
                 optics=None, retime="flow", grade_strength=1.0,
                 pulse=0.0, pulse_start=None, optics_kw=None):
        self.tl, self.cam = timeline, camera
        self.grid = timeline.grid
        self.W, self.H = camera.W, camera.H
        self.look = look_path(look) if look else None
        self.grade_strength = grade_strength
        self.track, self.pose = track, pose
        self.optics = optics or Optics(self.W, self.H)
        self.optics_kw = optics_kw or {}
        self.retime = retime
        self.pulse = pulse
        self.pulse_start = self.tl.cuts[0] if pulse_start is None and self.tl.cuts else 0.0
        self.layers = []
        self._mask_cache = {}

    # ------------------------------------------------------------------
    def add(self, fn, z="front", name=""):
        self.layers.append(Layer(fn, z, name))
        return self

    def back(self, fn, name=""):
        return self.add(fn, "back", name)

    def front(self, fn, name=""):
        return self.add(fn, "front", name)

    # ------------------------------------------------------------------
    def _mask(self, st, geo):
        """Subject mask for this frame, in output space."""
        if not self.pose:
            return None
        key = round(st["src_t"], 3)
        m = self._mask_cache.get(key)
        if m is None:
            i = int(round(st["src_t"] * self.grid.fps))
            rec = min(self.pose, key=lambda r: abs(r["i"] - 1 - i))
            m = rec.get("mask")
            if m is None:
                return None
            if len(self._mask_cache) > 12:
                self._mask_cache.clear()
            self._mask_cache[key] = m
        return self.cam.mask_frame(m, geo)

    def _pulse(self, arr, t):
        if self.pulse <= 0 or t < self.pulse_start:
            return arr
        k = self.grid.pulse(t, self.pulse, start=self.pulse_start)
        if k - 1.0 < 0.0022:
            return arr
        nw, nh = int(self.W * k), int(self.H * k)
        im = Image.fromarray(arr.clip(0, 255).astype(np.uint8)).resize((nw, nh), Image.BICUBIC)
        l, tp = (nw - self.W) // 2, (nh - self.H) // 2
        return np.asarray(im.crop((l, tp, l + self.W, tp + self.H)), np.float32)

    @staticmethod
    def _over(arr, canvas):
        ov = np.asarray(canvas, np.float32)
        a = ov[..., 3:4] / 255.0
        return arr * (1 - a) + ov[..., :3] * a

    # ------------------------------------------------------------------
    def frame(self, n):
        t = n / self.grid.fps
        shot = self.tl.at(t)
        st = shot.state(t)
        geo = self.cam.geometry(st, n, self.track)
        pic = self.cam.frame(st, n, track=self.track, retime=self.retime, geo=geo)

        arr = np.asarray(pic, np.float32)
        arr = self._pulse(arr, t)
        if self.look:
            arr = apply_lut(arr, self.look, self.grade_strength)
        arr = self.optics.apply(arr, n, **self.optics_kw)

        back = [l for l in self.layers if l.z == "back"]
        if back:
            clean = arr.copy()
            canvas = Image.new("RGBA", (self.W, self.H), (0, 0, 0, 0))
            for l in back:
                l.fn(canvas, t, n)
            arr = self._over(arr, canvas)
            m = self._mask(st, geo)
            if m is not None:
                arr = arr * (1 - m) + clean * m      # subject back on top

        front = [l for l in self.layers if l.z == "front"]
        if front:
            canvas = Image.new("RGBA", (self.W, self.H), (0, 0, 0, 0))
            for l in front:
                l.fn(canvas, t, n)
            arr = self._over(arr, canvas)

        return np.clip(arr, 0, 255).astype(np.uint8)

    # ------------------------------------------------------------------
    def preview(self, marks, path, cols=6, tile=180):
        th = tile * self.H // self.W
        rows = (len(marks) + cols - 1) // cols
        sheet = Image.new("RGB", (cols * tile, rows * th), (10, 10, 14))
        for i, m in enumerate(marks):
            f = Image.fromarray(self.frame(min(m, self.tl.frames - 1)))
            sheet.paste(f.resize((tile, th), Image.LANCZOS),
                        ((i % cols) * tile, (i // cols) * th))
        os.makedirs(os.path.dirname(os.path.abspath(path)) or ".", exist_ok=True)
        sheet.save(path, quality=91)
        return path

    def render(self, out_path, audio=None, crf=18, preset="medium",
               audio_bitrate="256k", progress=True, frames=None):
        frames = frames if frames is not None else range(self.tl.frames)
        frames = list(frames)
        os.makedirs(os.path.dirname(os.path.abspath(out_path)) or ".", exist_ok=True)
        cmd = ["ffmpeg", "-v", "error", "-y",
               "-f", "rawvideo", "-pix_fmt", "rgb24",
               "-s", f"{self.W}x{self.H}", "-r", str(self.grid.fps), "-i", "-"]
        if audio:
            cmd += ["-i", audio, "-map", "0:v:0", "-map", "1:a:0", "-shortest",
                    "-c:a", "aac", "-b:a", audio_bitrate, "-ar", "48000", "-ac", "2"]
        cmd += ["-c:v", "libx264", "-preset", preset, "-crf", str(crf),
                "-profile:v", "high", "-pix_fmt", "yuv420p",
                "-x264-params", "keyint=60:min-keyint=30:ref=4:bframes=3",
                "-movflags", "+faststart", out_path]
        p = subprocess.Popen(cmd, stdin=subprocess.PIPE)
        for k, n in enumerate(frames):
            p.stdin.write(self.frame(n).tobytes())
            if progress and k % 30 == 0:
                pct = 100.0 * k / max(1, len(frames))
                print(f"  render {k}/{len(frames)} ({pct:.0f}%)", flush=True)
        p.stdin.close()
        rc = p.wait()
        if rc != 0:
            raise RuntimeError(f"ffmpeg exited {rc}")
        return out_path


# ------------------------------------------------------------- layer makers
def sequence_layer(paths, start_frame, w, h, fade=(0.0, 0.0), fps=30,
                   position="center", scale=1.0, opacity=1.0):
    """Composite a rendered PNG sequence (3D or web) as a layer."""
    cache = {}

    def fn(canvas, t, n):
        i = n - start_frame
        if i < 0 or i >= len(paths):
            return
        im = cache.get(i)
        if im is None:
            im = Image.open(paths[i]).convert("RGBA")
            if scale != 1.0:
                im = im.resize((int(im.width * scale), int(im.height * scale)), Image.LANCZOS)
            if len(cache) > 90:
                cache.clear()
            cache[i] = im
        a = opacity
        if fade[0]:
            a *= band(i / fps, 0, fade[0])
        if fade[1]:
            a *= 1 - band(i / fps, len(paths) / fps - fade[1], len(paths) / fps)
        if a <= 0.01:
            return
        if a < 0.999:
            im = im.copy()
            im.putalpha(im.getchannel("A").point(lambda v: int(v * a)))
        if position == "center":
            canvas.alpha_composite(im, ((w - im.width) // 2, (h - im.height) // 2))
        else:
            canvas.alpha_composite(im, position)
    return fn


def hud_layer(w, h, text_left="REC", accent=(60, 226, 255), font_name=None,
              size=27, show=(0.0, 1e9), timecode=True, fps=30):
    """Camera-style overlay: blinking dot, label, running timecode."""
    from .typo import solid, font, pick
    from .shapes import PALETTE, rot_scale
    from PIL import ImageDraw
    fname = font_name or pick(tag="mono")
    cache = {}

    def fn(canvas, t, n):
        a = min(band(t, show[0], show[0] + 0.4), 1 - band(t, show[1] - 0.35, show[1]))
        if a <= 0.02:
            return
        d = ImageDraw.Draw(canvas)
        if (t * 1.6) % 1.0 < 0.62:
            d.ellipse([64, 96, 92, 124], fill=(255, 60, 90, int(235 * a)))
        if "lbl" not in cache:
            cache["lbl"] = solid(text_left, font(fname, size), (255, 255, 255),
                                 stroke=5, stroke_fill=PALETTE["ink"])
        canvas.alpha_composite(rot_scale(cache["lbl"], 0, 1.0, a), (96, 78))
        if timecode:
            tc = f"{int(t)//60:02d}:{int(t)%60:02d}:{int((t%1)*fps):02d}"
            lay = solid(tc, font(fname, size), accent, stroke=5, stroke_fill=PALETTE["ink"])
            canvas.alpha_composite(rot_scale(lay, 0, 1.0, a), (w - lay.width - 46, 78))
    return fn
