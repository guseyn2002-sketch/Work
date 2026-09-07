# -*- coding: utf-8 -*-
"""Source footage: decode once, then sample at arbitrary sub-frame positions.

Slow motion is the give-away in amateur edits. Blending two neighbours ghosts;
this module warps along the optical flow instead, which is what a hardware
retimer does.
"""
import os, math, json, subprocess, shutil
import numpy as np
from PIL import Image

try:
    import cv2
except ImportError:                                   # optional at import time
    cv2 = None


def probe(path):
    """Container facts we actually need."""
    out = subprocess.run(
        ["ffprobe", "-v", "error", "-select_streams", "v:0",
         "-show_entries", "stream=width,height,r_frame_rate,nb_frames,codec_name",
         "-show_entries", "stream_side_data=rotation",
         "-show_entries", "format=duration",
         "-of", "json", path], capture_output=True, text=True).stdout
    d = json.loads(out or "{}")
    st = (d.get("streams") or [{}])[0]
    num, den = (st.get("r_frame_rate") or "30/1").split("/")
    rot = 0
    for sd in st.get("side_data_list", []) or []:
        if "rotation" in sd:
            rot = int(sd["rotation"])
    return dict(width=st.get("width"), height=st.get("height"),
                fps=float(num) / float(den or 1), codec=st.get("codec_name"),
                rotation=rot,
                duration=float((d.get("format") or {}).get("duration") or 0.0))


class FrameStore:
    """Decoded frames on disk plus a small in-memory cache.

    Keeping the store at the *source* resolution is deliberate: a 4K original
    gives roughly 2x of punch-in headroom at a 1080p delivery, which is where
    extra shots come from when there is only one take.
    """

    def __init__(self, video, workdir, width=None, height=None, fps=30,
                 quality=2, rebuild=False):
        self.video, self.workdir, self.fps = video, workdir, fps
        self.dir = os.path.join(workdir, "frames")
        self.info = probe(video)
        # a rotated phone clip reports landscape dimensions; swap them
        w, h = self.info["width"], self.info["height"]
        if abs(self.info["rotation"]) in (90, 270):
            w, h = h, w
        self.width = width or w
        self.height = height or h
        if rebuild and os.path.isdir(self.dir):
            shutil.rmtree(self.dir)
        if not os.path.isdir(self.dir) or not os.listdir(self.dir):
            os.makedirs(self.dir, exist_ok=True)
            subprocess.run(
                ["ffmpeg", "-v", "error", "-i", video,
                 "-vf", f"scale={self.width}:{self.height}:flags=lanczos,fps={fps}",
                 "-q:v", str(quality), "-y", os.path.join(self.dir, "%05d.jpg")],
                check=True)
        self.files = sorted(f for f in os.listdir(self.dir) if f.endswith(".jpg"))
        self.n = len(self.files)
        self.duration = self.n / fps
        self._cache, self._flow = {}, {}

    # ------------------------------------------------------------ access
    def path(self, i):
        return os.path.join(self.dir, self.files[max(0, min(self.n - 1, i - 1))])

    def load(self, i, scale=1.0):
        """Frame `i` (1-based) decoded no larger than needed.

        JPEG draft mode decodes at 1/2, 1/4 or 1/8 for free, which is most of
        the speed of the whole pipeline when the shot is not punched in.
        """
        red = 1
        for r in (8, 4, 2):
            if scale <= 1.0 / r:
                red = r
                break
        key = (i, red)
        hit = self._cache.get(key)
        if hit is not None:
            return hit
        im = Image.open(self.path(i))
        if red > 1:
            im.draft("RGB", (self.width // red, self.height // red))
        im = im.convert("RGB")
        if len(self._cache) > 24:
            self._cache.clear()
        self._cache[key] = im
        return im

    # ------------------------------------------------------------ retime
    def sample(self, t, scale=1.0, mode="flow"):
        """Sample at time `t` seconds; `mode` is 'flow', 'blend' or 'nearest'."""
        f = t * self.fps
        i0 = int(math.floor(f)) + 1
        frac = f - math.floor(f)
        if frac < 0.015 or mode == "nearest" or i0 >= self.n:
            return self.load(i0, scale)
        a = self.load(i0, scale)
        b = self.load(i0 + 1, scale)
        if b.size != a.size:
            b = b.resize(a.size, Image.BICUBIC)
        if mode == "blend" or cv2 is None:
            return Image.blend(a, b, frac)
        return self._flow_warp(a, b, frac, i0, scale)

    def _flow_pair(self, a, b, key):
        hit = self._flow.get(key)
        if hit is not None:
            return hit
        ga = cv2.cvtColor(np.asarray(a), cv2.COLOR_RGB2GRAY)
        gb = cv2.cvtColor(np.asarray(b), cv2.COLOR_RGB2GRAY)
        dis = cv2.DISOpticalFlow_create(cv2.DISOPTICAL_FLOW_PRESET_MEDIUM)
        fwd = dis.calc(ga, gb, None)
        bwd = dis.calc(gb, ga, None)
        if len(self._flow) > 8:
            self._flow.clear()
        self._flow[key] = (fwd, bwd)
        return fwd, bwd

    def _flow_warp(self, a, b, u, i0, scale):
        na, nb = np.asarray(a), np.asarray(b)
        h, w = na.shape[:2]
        fwd, bwd = self._flow_pair(a, b, (i0, w, h))
        gy, gx = np.mgrid[0:h, 0:w].astype(np.float32)
        wa = cv2.remap(na, gx + fwd[..., 0] * u, gy + fwd[..., 1] * u,
                       cv2.INTER_LINEAR, borderMode=cv2.BORDER_REPLICATE)
        wb = cv2.remap(nb, gx + bwd[..., 0] * (1 - u), gy + bwd[..., 1] * (1 - u),
                       cv2.INTER_LINEAR, borderMode=cv2.BORDER_REPLICATE)
        out = (wa.astype(np.float32) * (1 - u) + wb.astype(np.float32) * u)
        return Image.fromarray(np.clip(out, 0, 255).astype(np.uint8))

    def __repr__(self):
        return (f"<FrameStore {self.n} frames {self.width}x{self.height} "
                f"@{self.fps}fps ({self.duration:.2f}s) codec={self.info['codec']}>")
