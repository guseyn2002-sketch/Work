# -*- coding: utf-8 -*-
"""Score and sound design: synthesis, real instruments, mastering.

Two sources of sound. Synthesis for anything designed — risers, impacts,
whooshes, sub drops. A General MIDI soundfont for anything played, because a
sampled piano or string section reads as production value in a way a sine
stack does not.
"""
import math, os, shutil, subprocess, tempfile
import numpy as np
from scipy import signal as sg
from scipy.io import wavfile

SR = 48000
SF2_CANDIDATES = [
    "/usr/share/sounds/sf2/FluidR3_GM.sf2",
    "/usr/share/soundfonts/FluidR3_GM.sf2",
    "/usr/share/sounds/sf2/default-GM.sf2",
]

# a few useful General MIDI programs
GM = dict(piano=0, epiano=4, harp=46, strings=48, ensemble=49, choir=52, voice=53,
          trumpet=56, horn=60, oboe=68, flute=73, pad_warm=89, pad_halo=94,
          bell=14, celesta=8, marimba=12, glock=9, nylon_guitar=24, cello=42,
          contrabass=43, timpani=47, synth_bass=38, saw_lead=81)


# ------------------------------------------------------------------ helpers
def t_axis(n):
    return np.arange(n) / SR


def midi_hz(m):
    return 440.0 * 2 ** ((m - 69) / 12.0)


def place(buf, x, t0, gain=1.0):
    i = int(t0 * SR)
    if i >= len(buf) or i < -len(x):
        return
    if i < 0:
        x, i = x[-i:], 0
    m = min(len(x), len(buf) - i)
    if m > 0:
        buf[i:i + m] += x[:m] * gain


def adsr(n, a=0.01, d=0.1, s=0.7, r=0.2):
    e = np.zeros(n)
    ai, di, ri = int(a * SR), int(d * SR), int(r * SR)
    si = max(0, n - ai - di - ri)
    i = 0
    if ai: e[i:i + ai] = np.linspace(0, 1, ai); i += ai
    if di: e[i:i + di] = np.linspace(1, s, di); i += di
    if si: e[i:i + si] = s; i += si
    if ri: e[i:i + ri] = np.linspace(s, 0, min(ri, max(0, n - i)))
    return e


def lp(x, fc, order=4):
    b, a = sg.butter(order, max(40.0, min(fc, SR / 2 * .95)) / (SR / 2), "low")
    return sg.lfilter(b, a, x, axis=0)


def hp(x, fc, order=2):
    b, a = sg.butter(order, max(20.0, min(fc, SR / 2 * .95)) / (SR / 2), "high")
    return sg.lfilter(b, a, x, axis=0)


def bp(x, f1, f2, order=4):
    b, a = sg.butter(order, [max(20., f1) / (SR / 2), min(f2, SR / 2 * .95) / (SR / 2)], "band")
    return sg.lfilter(b, a, x, axis=0)


# ------------------------------------------------------------------ design
def note(f, dur, kind="saw", detune=0.0, a=.01, d=.1, s=.6, r=.2):
    n = int(dur * SR); t = t_axis(n)
    if kind == "saw":
        y = sg.sawtooth(2 * np.pi * f * t)
        if detune:
            y = .55 * y + .45 * sg.sawtooth(2 * np.pi * f * (1 + detune) * t)
    elif kind == "square":
        y = sg.square(2 * np.pi * f * t, duty=.42)
    elif kind == "tri":
        y = sg.sawtooth(2 * np.pi * f * t, width=.5)
    elif kind == "bell":
        y = np.sin(2 * np.pi * f * t + np.sin(2 * np.pi * f * 3.51 * t) * np.exp(-t * 7) * 4.2)
    else:
        y = np.sin(2 * np.pi * f * t)
    return y * adsr(n, a, d, s, r)


def whoosh(dur=.62, reverse=False, bright=1.0, seed=0):
    n = int(dur * SR); u = t_axis(n) / dur
    rng = np.random.default_rng(seed)
    nz = rng.normal(0, 1, n); out = np.zeros(n)
    for i in range(0, n, 1024):
        j = min(n, i + 1024)
        p = 1 - i / n if reverse else i / n
        fc = (420 + 5200 * math.sin(math.pi * p) ** 1.4) * bright
        out[i:j] = bp(nz[max(0, i - 1024):j], fc * .45, min(fc * 2.4, 19000))[-(j - i):]
    return out * np.sin(np.pi * u) ** 1.5


def riser(dur=2.2, f0=280, f1=6800, seed=1):
    n = int(dur * SR); u = t_axis(n) / dur
    rng = np.random.default_rng(seed)
    nz = rng.normal(0, 1, n); out = np.zeros(n)
    for i in range(0, n, 2048):
        j = min(n, i + 2048)
        fc = f0 * (f1 / f0) ** (i / n)
        out[i:j] = bp(nz[max(0, i - 2048):j], fc * .55, min(fc * 1.5, 20000))[-(j - i):]
    return out * (u ** 2.4) * np.hanning(n) ** .25


def impact(dur=1.6, f0=92, seed=2):
    n = int(dur * SR); t = t_axis(n)
    rng = np.random.default_rng(seed)
    f = f0 * np.exp(-t * 8) + 32
    y = np.sin(2 * np.pi * np.cumsum(f) / SR) * np.exp(-t * 3.0)
    return y + bp(rng.normal(0, 1, n), 200, 3000) * np.exp(-t * 16) * .35


def sub_drop(dur=2.4, f0=110, f1=28):
    n = int(dur * SR); t = t_axis(n)
    f = f0 * (f1 / f0) ** (t / dur)
    return np.sin(2 * np.pi * np.cumsum(f) / SR) * np.exp(-t * 1.4)


def sparkle(dur=1.0, n_part=13, base=1750, seed=3):
    rng = np.random.default_rng(seed)
    n = int(dur * SR); out = np.zeros(n)
    for k in range(n_part):
        f = base * (1 + .62 * k) * (1 + rng.normal(0, .02))
        d = .10 + .42 * rng.random()
        place(out, note(f, d, "bell", a=.001, d=.05, s=.05, r=d * .7),
              rng.random() * dur * .5, .13 / (1 + k * .22))
    return out


def shutter(seed=4):
    rng = np.random.default_rng(seed)
    n = int(.16 * SR); y = np.zeros(n)
    a = rng.normal(0, 1, int(.02 * SR)) * np.exp(-t_axis(int(.02 * SR)) * 260)
    b = rng.normal(0, 1, int(.03 * SR)) * np.exp(-t_axis(int(.03 * SR)) * 150)
    y[:len(a)] += bp(a, 1800, 11000)
    o = int(.055 * SR); y[o:o + len(b)] += bp(b, 900, 7000) * .85
    return y


def crackle(n, density=.00035, seed=5):
    rng = np.random.default_rng(seed)
    y = rng.normal(0, 1, n) * .0022
    pops = rng.random(n) < density
    y[pops] += rng.normal(0, 1, pops.sum()) * .05
    return bp(y, 900, 9500)


def drum_kit(seed=6):
    rng = np.random.default_rng(seed)

    def kick():
        n = int(.34 * SR); t = t_axis(n)
        f = 118 * np.exp(-t * 34) + 44
        y = np.sin(2 * np.pi * np.cumsum(f) / SR) * np.exp(-t * 8.5)
        return lp(y + rng.normal(0, 1, n) * np.exp(-t * 190) * .28, 2600)

    def snare():
        n = int(.24 * SR); t = t_axis(n)
        return bp(rng.normal(0, 1, n) * np.exp(-t * 22), 900, 7200) + \
               np.sin(2 * np.pi * 195 * t) * np.exp(-t * 30) * .32

    def hat(open_=False):
        n = int((.20 if open_ else .055) * SR); t = t_axis(n)
        return hp(bp(rng.normal(0, 1, n) * np.exp(-t * (10 if open_ else 62)), 6200, 15000), 5200)

    def clap():
        n = int(.28 * SR); y = np.zeros(n)
        for off, g in ((0, .7), (.008, .9), (.017, 1.), (.027, .6)):
            i = int(off * SR)
            y[i:] += rng.normal(0, 1, n - i) * np.exp(-t_axis(n - i) * 30) * g
        return bp(y, 1100, 6000)

    return dict(kick=kick(), snare=snare(), hat=hat(), hat_open=hat(True), clap=clap())


# ------------------------------------------------------- real instruments
def soundfont():
    for p in SF2_CANDIDATES:
        if os.path.exists(p):
            return p
    return None


def play_midi(notes, program=0, bpm=100, sf2=None, gain=0.9, tempo_ticks=480):
    """Render note events with a soundfont.

    notes: iterable of (start_beats, dur_beats, midi_pitch, velocity).
    Returns a mono float array.
    """
    import mido
    from mido import Message, MidiFile, MidiTrack, MetaMessage, bpm2tempo
    sf = sf2 or soundfont()
    if not sf:
        raise RuntimeError("no soundfont found; apt install fluid-soundfont-gm")

    mf = MidiFile(ticks_per_beat=tempo_ticks)
    tr = MidiTrack(); mf.tracks.append(tr)
    tr.append(MetaMessage("set_tempo", tempo=bpm2tempo(bpm)))
    tr.append(Message("program_change", program=int(program), time=0))
    evs = []
    for start, dur, pitch, vel in notes:
        evs.append((start, 1, int(pitch), int(vel)))
        evs.append((start + dur, 0, int(pitch), 0))
    evs.sort(key=lambda e: (e[0], e[1]))
    last = 0.0
    for beat, on, pitch, vel in evs:
        dt = int(round((beat - last) * tempo_ticks))
        tr.append(Message("note_on" if on else "note_off", note=pitch,
                          velocity=vel, time=max(0, dt)))
        last = beat

    tmp = tempfile.mkdtemp()
    mid = os.path.join(tmp, "s.mid"); wav = os.path.join(tmp, "s.wav")
    mf.save(mid)
    subprocess.run(["fluidsynth", "-ni", "-F", wav, "-r", str(SR), "-g", str(gain),
                    sf, mid], capture_output=True, check=False)
    if not os.path.exists(wav):
        shutil.rmtree(tmp, ignore_errors=True)
        raise RuntimeError("fluidsynth produced no output")
    sr, x = wavfile.read(wav)
    shutil.rmtree(tmp, ignore_errors=True)
    x = x.astype(np.float32) / 32768.0
    return x.mean(1) if x.ndim > 1 else x


# ------------------------------------------------------------- processing
def board(reverb=0.0, room=0.7, delay=0.0, delay_time=0.28, comp=None,
          chorus=0.0, pitch=0.0, hp_hz=0.0, lp_hz=0.0, gain_db=0.0, limit_db=None):
    """Assemble a pedalboard chain; falls back to scipy if unavailable."""
    from pedalboard import (Pedalboard, Reverb, Delay, Compressor, Chorus,
                            PitchShift, HighpassFilter, LowpassFilter, Gain, Limiter)
    fx = []
    if hp_hz:   fx.append(HighpassFilter(hp_hz))
    if pitch:   fx.append(PitchShift(semitones=pitch))
    if comp:    fx.append(Compressor(threshold_db=comp[0], ratio=comp[1]))
    if chorus:  fx.append(Chorus(rate_hz=0.6, depth=0.3, mix=chorus))
    if delay:   fx.append(Delay(delay_seconds=delay_time, feedback=0.32, mix=delay))
    if reverb:  fx.append(Reverb(room_size=room, damping=0.4, wet_level=reverb,
                                 dry_level=1 - reverb * 0.4, width=1.0))
    if lp_hz:   fx.append(LowpassFilter(lp_hz))
    if gain_db: fx.append(Gain(gain_db))
    if limit_db is not None:
        fx.append(Limiter(threshold_db=limit_db))
    return Pedalboard(fx)


def apply_board(x, chain):
    mono = x.ndim == 1
    y = chain(x.astype(np.float32), SR)
    return y if not mono else np.asarray(y).reshape(-1)


def sidechain(n, cuts, depth=0.45, attack=0.02, release=0.26):
    """Duck envelope keyed to a list of times.

    The eased recovery is applied to the interpolation, not to the finished
    gain, so the floor really is `1 - depth`.
    """
    duck = np.ones(n)
    L = int(release * SR)
    ramp = 1.0 - depth * (1.0 - np.linspace(0.0, 1.0, L) ** 0.7)
    for c in cuts:
        i = int(c * SR)
        if 0 <= i < n:
            m = min(L, n - i)
            duck[i:i + m] = np.minimum(duck[i:i + m], ramp[:m])
    return duck


def duck_under(bed, key, depth=0.52, attack=0.35, release=0.9985):
    """Classic broadcast ducking: bed drops whenever `key` speaks."""
    env = np.abs(key if key.ndim == 1 else np.abs(key).max(1))
    w = int(.030 * SR)
    env = np.convolve(env, np.ones(w) / w, mode="same")
    env = env / (env.max() + 1e-9)
    k = np.clip(env * 3.4, 0, 1)
    acc, out = 0.0, np.zeros_like(k)
    for i in range(len(k)):
        acc = max(k[i] * attack + acc * (1 - attack), acc * release)
        out[i] = acc
    g = 1.0 - depth * out
    return bed * (g[:, None] if bed.ndim == 2 else g)


def master(x, target_lufs=-14.0, tp=-1.2, out_path=None):
    """Limit, then normalise loudness with ffmpeg's loudnorm."""
    x = np.tanh(x * 1.12) * 0.93
    peak = np.abs(x).max() or 1.0
    x = x / peak * 0.97
    if x.ndim == 1:
        x = np.stack([x, x], 1)
    if not out_path:
        return x
    tmp = out_path + ".raw.wav"
    wavfile.write(tmp, SR, (np.clip(x, -1, 1) * 32767).astype(np.int16))
    subprocess.run(["ffmpeg", "-v", "error", "-y", "-i", tmp,
                    "-af", f"loudnorm=I={target_lufs}:TP={tp}:LRA=9",
                    "-ar", str(SR), "-ac", "2", out_path], check=True)
    os.remove(tmp)
    return out_path


def write(path, x, sr=SR):
    if x.ndim == 1:
        x = np.stack([x, x], 1)
    wavfile.write(path, sr, (np.clip(x, -1, 1) * 32767).astype(np.int16))
    return path
