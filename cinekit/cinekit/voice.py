# -*- coding: utf-8 -*-
"""Narration with per-word timings.

The timings are the point: they let captions land on the syllable rather than
being eyeballed. RHVoice gives Russian and a dozen other languages offline.
"""
import os, re, json, shutil, subprocess, tempfile
import numpy as np
from scipy import signal as sg
from scipy.io import wavfile

SR = 48000
VOWELS = "аеёиоуыэюяaeiouyAEIOUYАЕЁИОУЫЭЮЯ"

RU_VOICES = ["anna", "elena", "irina", "arina", "tatiana", "victoria",
             "aleksandr", "aleksandr-hq", "artemiy", "mikhail", "pavel",
             "evgeniy-rus", "vitaliy", "yuriy"]


def available():
    return shutil.which("RHVoice-test") is not None


def voices():
    d = "/usr/share/RHVoice/voices"
    return sorted(os.listdir(d)) if os.path.isdir(d) else []


def _synth(text, voice, rate, pitch, volume, out_wav):
    tmp = tempfile.mkdtemp()
    txt = os.path.join(tmp, "in.txt")
    open(txt, "w", encoding="utf-8").write(text)
    subprocess.run(["RHVoice-test", "-p", voice, "-q", "3",
                    "-r", str(rate), "-t", str(pitch), "-v", str(volume),
                    "-R", str(SR), "-i", txt, "-o", out_wav],
                   check=True, capture_output=True)
    shutil.rmtree(tmp, ignore_errors=True)
    sr, x = wavfile.read(out_wav)
    x = x.astype(np.float32) / 32768.0
    return x.mean(1) if x.ndim > 1 else x


def _trim(x, thr_db=-46, pad=0.04):
    w = int(0.01 * SR)
    e = np.convolve(np.abs(x), np.ones(w) / w, mode="same")
    idx = np.where(e > 10 ** (thr_db / 20) * (e.max() or 1))[0]
    if not len(idx):
        return x
    return x[max(0, idx[0] - int(pad * SR)): min(len(x), idx[-1] + int(pad * SR))]


def _word_spans(x, words, snap=0.10):
    """Split a phrase into per-word spans.

    A proportional estimate from syllable weight, snapped to the nearest real
    dip in energy. Pure gap-finding picks stop consonants inside words;
    pure proportion ignores how the voice actually phrased it.
    """
    n_w = len(words)
    dur = len(x) / SR
    if n_w == 1:
        return [(0.0, dur)]

    def weight(w):
        return max(1.0, sum(c in VOWELS for c in w)) + 0.25 * len(w) / 3.0 + 0.35

    wts = np.array([weight(w) for w in words], float)
    wts /= wts.sum()
    targets, acc = [], 0.0
    for wt in wts[:-1]:
        acc += wt
        targets.append(acc * dur)

    w = int(0.015 * SR)
    e = np.convolve(np.abs(x), np.ones(w) / w, mode="same")
    e = e / (e.max() + 1e-9)
    quiet = e < 0.11
    runs, i = [], 0
    while i < len(quiet):
        if quiet[i]:
            j = i
            while j < len(quiet) and quiet[j]:
                j += 1
            if (j - i) > int(0.018 * SR):
                runs.append((i + j) / 2 / SR)
            i = j
        else:
            i += 1

    cuts, prev = [], 0.06
    for k, tg in enumerate(targets):
        cands = [c for c in runs if abs(c - tg) < snap and c > prev + 0.06
                 and c < dur - 0.06]
        c = min(cands, key=lambda c: abs(c - tg)) if cands else max(tg, prev + 0.06)
        c = min(c, dur - 0.06 * (n_w - k - 1) - 0.06)
        cuts.append(c); prev = c

    spans, prev = [], 0.0
    for c in cuts:
        spans.append((prev, c)); prev = c
    spans.append((prev, dur))
    return spans


def _process(x, presence=0.35, body=0.18, comp=(0.16, 3.4), reverb=0.16):
    b, a = sg.butter(2, 90 / (SR / 2), "high"); x = sg.lfilter(b, a, x)
    b, a = sg.iirpeak(3200 / (SR / 2), 1.6); x = x + presence * sg.lfilter(b, a, x)
    b, a = sg.iirpeak(220 / (SR / 2), 1.2); x = x + body * sg.lfilter(b, a, x)
    b, a = sg.butter(4, 13000 / (SR / 2), "low"); x = sg.lfilter(b, a, x)
    thr, ratio = comp
    w = int(0.012 * SR)
    env = np.convolve(np.abs(x), np.ones(w) / w, mode="same")
    g = np.ones_like(env)
    over = env > thr
    g[over] = (thr + (env[over] - thr) / ratio) / (env[over] + 1e-9)
    x = np.tanh(x * g * 1.5 * 1.05) * 0.92
    if reverb:
        rng = np.random.default_rng(11)
        ln = int(0.85 * SR)
        ir = rng.normal(0, 1, ln) * np.exp(-np.arange(ln) / (0.85 * SR / 3.4))
        b, a = sg.butter(4, 3800 / (SR / 2), "low"); ir = sg.lfilter(b, a, ir)
        ir[:int(0.012 * SR)] = 0
        ir /= (np.abs(ir).sum() / 14)
        x = (1 - reverb) * x + reverb * sg.fftconvolve(x, ir)[:len(x)]
    return x


def narrate(lines, voice="anna", rate=50, pitch=50, volume=100, tempo=0.93,
            total=None, workdir=None, process=True):
    """Render narration lines onto one bed and return (audio, timing).

    lines: [{t: seconds, text: str, ...}, ...]
    RHVoice clamps rate below 50, so slowing down is done with a time-stretch.
    """
    if not available():
        raise RuntimeError("RHVoice-test not found; apt install rhvoice rhvoice-russian")
    work = workdir or tempfile.mkdtemp()
    os.makedirs(work, exist_ok=True)
    total = total or (max(l["t"] for l in lines) + 8.0)
    bed = np.zeros(int(total * SR), np.float32)
    timing = []

    for i, ln in enumerate(lines):
        raw_p = os.path.join(work, f"line{i:02d}.wav")
        x = _synth(ln["text"], voice, rate, pitch, volume, raw_p)
        if tempo and abs(tempo - 1.0) > 1e-3:
            slow = raw_p.replace(".wav", "_t.wav")
            subprocess.run(["ffmpeg", "-v", "error", "-y", "-i", raw_p,
                            "-af", f"atempo={tempo}", slow],
                           check=True, capture_output=True)
            sr, xx = wavfile.read(slow)
            x = (xx.astype(np.float32) / 32768.0)
            x = x.mean(1) if x.ndim > 1 else x
        x = _trim(x)
        words = ln["text"].split()
        spans = _word_spans(x, words)
        out = _process(x) if process else x
        t0 = ln["t"]
        s = int(t0 * SR)
        m = min(len(out), len(bed) - s)
        if m > 0:
            bed[s:s + m] += out[:m]
        timing.append(dict(
            idx=i, t=t0, end=t0 + len(x) / SR, text=ln["text"],
            style=ln.get("style", "pop"), slot=ln.get("slot", (t0 - .2, t0 + len(x) / SR + 1.)),
            words=[dict(w=w, t0=round(t0 + a, 3), t1=round(t0 + b, 3))
                   for w, (a, b) in zip(words, spans)]))
    return bed, timing


def save_timing(timing, path):
    json.dump(timing, open(path, "w"), ensure_ascii=False, indent=1)
    return path
