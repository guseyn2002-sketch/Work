# -*- coding: utf-8 -*-
"""Subject tracking: where the person is, and which pixels are them.

Two things fall out of this that are hard to fake by hand — a crop that stays
composed on the subject through a whole move, and type that passes *behind*
them.
"""
import os, json, math
import numpy as np

from . import MODELS

_POSE = None
_FACE = None


def _pose():
    global _POSE
    if _POSE is None:
        import mediapipe as mp
        from mediapipe.tasks import python as mpp
        from mediapipe.tasks.python import vision
        path = os.path.join(MODELS, "pose_landmarker.task")
        if not os.path.exists(path):
            raise FileNotFoundError(
                f"{path} missing — run scripts/fetch_assets.sh")
        _POSE = vision.PoseLandmarker.create_from_options(
            vision.PoseLandmarkerOptions(
                base_options=mpp.BaseOptions(model_asset_buffer=open(path, "rb").read()),
                output_segmentation_masks=True,
                min_pose_detection_confidence=0.4,
                min_tracking_confidence=0.4))
    return _POSE


# landmark indices we care about
NOSE, L_EYE, R_EYE = 0, 2, 5
L_SHOULDER, R_SHOULDER, L_HIP, R_HIP = 11, 12, 23, 24


def analyse(store, step=1, downscale=540, mask=True, verbose=False):
    """Run the tracker over a FrameStore.

    Returns a list, one entry per analysed frame:
        {i, t, found, face:(x,y), body:(x,y), scale, mask (HxW float or None)}
    Coordinates are normalised to the frame.
    """
    import mediapipe as mp
    import cv2
    det = _pose()
    out = []
    h = int(downscale * store.height / store.width)
    for i in range(1, store.n + 1, step):
        im = store.load(i, scale=downscale / store.width)
        arr = np.asarray(im.resize((downscale, h)))
        res = det.detect(mp.Image(image_format=mp.ImageFormat.SRGB, data=arr))
        rec = dict(i=i, t=(i - 1) / store.fps, found=False,
                   face=None, body=None, scale=None, mask=None)
        if res.pose_landmarks:
            lm = res.pose_landmarks[0]
            eyes = [lm[L_EYE], lm[R_EYE]]
            fx = float(np.mean([lm[NOSE].x] + [e.x for e in eyes]))
            fy = float(np.mean([lm[NOSE].y] + [e.y for e in eyes]))
            sh = [lm[L_SHOULDER], lm[R_SHOULDER]]
            hp = [lm[L_HIP], lm[R_HIP]]
            bx = float(np.mean([p.x for p in sh + hp]))
            by = float(np.mean([p.y for p in sh + hp]))
            span = float(abs(lm[L_SHOULDER].x - lm[R_SHOULDER].x)) or 0.1
            rec.update(found=True, face=(fx, fy), body=(bx, by), scale=span,
                       vis=float(lm[NOSE].visibility))
            if mask and res.segmentation_masks:
                rec["mask"] = res.segmentation_masks[0].numpy_view().copy()
        out.append(rec)
        if verbose and i % 30 == 1:
            print(f"  track {i}/{store.n}", flush=True)
    return out


def smooth(track, key="face", window=9, max_jump=0.18):
    """Reject outliers, fill gaps, then low-pass — a usable camera path.

    Detectors flicker; a camera driven straight off raw detections judders.
    """
    pts = [(r["t"], r[key]) for r in track]
    ts = np.array([t for t, _ in pts])
    xs = np.array([p[0] if p else np.nan for _, p in pts])
    ys = np.array([p[1] if p else np.nan for _, p in pts])

    def clean(v):
        v = v.copy()
        good = ~np.isnan(v)
        if good.sum() < 2:
            return np.full_like(v, 0.5)
        # drop points that jump further than a subject plausibly moves
        med = np.nanmedian(v)
        v[np.abs(v - med) > max_jump * 4] = np.nan
        idx = np.arange(len(v))
        good = ~np.isnan(v)
        v = np.interp(idx, idx[good], v[good])
        for _ in range(2):                     # repeated box filter ~= gaussian
            k = np.ones(window) / window
            v = np.convolve(np.pad(v, window // 2, mode="edge"), k, mode="valid")[:len(v)]
        return v

    sx, sy = clean(xs), clean(ys)
    return dict(t=ts, x=sx, y=sy,
                at=lambda tt: (float(np.interp(tt, ts, sx)), float(np.interp(tt, ts, sy))))


def mask_at(track, t, fps, size=None, feather=3.0):
    """Person mask nearest time `t`, optionally resized, as float 0..1."""
    import cv2
    i = int(round(t * fps))
    best = min(track, key=lambda r: abs(r["i"] - 1 - i))
    m = best.get("mask")
    if m is None:
        return None
    m = np.asarray(m, np.float32)
    if size and (m.shape[1], m.shape[0]) != size:
        m = cv2.resize(m, size, interpolation=cv2.INTER_LINEAR)
    if feather > 0:
        k = int(feather) * 2 + 1
        m = cv2.GaussianBlur(m, (k, k), feather)
    return np.clip(m, 0, 1)
