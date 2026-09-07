# cinekit

A code-first pipeline for short-form video: beat-locked editing, film colour,
subject tracking, typography, 3D and sound, driven from Python.

It exists because one 9.7-second phone take had to become a finished reel.
Everything here came out of that problem and was tested against that footage.

## What it does that an overlay cannot

- **Tracks the subject** and drives the crop from it, so a punch-in stays
  composed through a whole move instead of drifting.
- **Puts graphics behind people.** A person segmentation mask is warped by the
  same camera transform as the picture, so a word can pass behind a shoulder.
- **Retimes along the optical flow.** Slow motion is warped, not cross-faded,
  so it does not ghost.
- **Grades through real 3D LUTs** it generates itself — the same `.cube` files
  open in Resolve or feed `ffmpeg -vf lut3d`.
- **Renders 3D** through headless Blender, so a title can be extruded metal lit
  in a studio rather than a bevel filter.
- **Scores with real instruments** through a General MIDI soundfont, and
  narrates with per-word timings so captions land on the syllable.

## Install

```bash
scripts/install_stack.sh     # ffmpeg, blender, fluidsynth, rhvoice, python deps
scripts/fetch_assets.sh      # MediaPipe models (~42 MB) + generate the LUTs
```

Fonts are committed. Models are not — they are large binaries Google serves
directly, and they version independently of this code.

## A minimal edit

```python
from cinekit.timeline import Grid, Timeline
from cinekit.source import FrameStore
from cinekit.camera import Camera
from cinekit.render import Compositor
from cinekit import track as tracking

store = FrameStore("take.mov", "work", fps=30)      # decode once
pose  = tracking.analyse(store)                      # landmarks + masks
path  = tracking.smooth(pose, key="face")            # a usable camera path

g  = Grid(bpm=92, fps=30)
tl = Timeline(g)
tl.cut("wide",  4, src=(0.0, 1.6), zoom=(1.00, 1.12))
tl.cut("close", 4, src=(1.7, 2.6), zoom=(1.5, 2.0), track="face",
       meta=dict(aim=(0.42, 0.36)))

comp = Compositor(tl, Camera(store), look="couture", track=path, pose=pose)
comp.back(my_title_layer)     # drawn behind the subject
comp.front(my_caption_layer)  # drawn over everything
comp.render("out.mp4", audio="mix.wav")
```

`examples/gorky_editorial.py` is the full version: 3D title, narration,
scored soundtrack, tracked reframing and a word passing behind the subject.

## Modules

| Module | Responsibility |
|---|---|
| `timeline` | Beat grid, shots, easing (spring, elastic, bounce, expo) |
| `source` | Frame store, sub-frame sampling, optical-flow retiming |
| `track` | Pose landmarks, person masks, outlier rejection, smoothing |
| `camera` | Crop, punch, roll, handheld, auto-reframe; reusable geometry |
| `grade` | Six looks as 33³ LUTs, halation, grain, vignette, aberration, bloom |
| `typo` | Chrome and gold type, karaoke captions, luxury lockups, font picking |
| `shapes` | Vector stickers, metal ramps, corner frames |
| `transitions` | Whip, flash, leak, glitch, burn, dip, zoom blur, push |
| `web` | Headless Chromium stepped by `window.setFrame(n)` |
| `scene3d` | Blender headless: extruded type, primitives, studio lighting |
| `audio` | Designed SFX, soundfont instruments, pedalboard, ducking, mastering |
| `voice` | Narration with per-word timings |
| `brands` | Look presets binding grade, type, optics and sound |
| `render` | Layer compositing and encode |

## Looks

`kodak` · `apple` · `couture` · `beauty` · `auto` · `y2k`

Each is a grade plus a type pairing, an optical intensity and a sound register.
They are starting points that keep a piece internally consistent. They are not
impersonations of anyone's identity, and no licensed brand typeface ships here.

## Fonts

155 files across 64 families, all open licence (OFL / Apache), 36 with
Cyrillic. `assets/fonts/catalog.json` tags every face by register, so
`typo.pick(tag="luxury", cyrillic=True)` returns something appropriate.

Commercial typefaces used by brands are deliberately absent. If you have
licences for them, drop the files into `assets/fonts/` and rebuild the
catalogue — the picker reads whatever is there.

## Performance

Measured at 1080×1920 on four CPU cores:

| Stage | Cost |
|---|---|
| Optical-flow interpolation | 0.44 s per frame pair |
| Pose landmarks + mask | ~120 ms per frame |
| LUT application | ~350 ms per frame |
| Chromium motion graphics | ~540 ms per frame |
| Blender EEVEE title, 1080x1920 | ~25 s per frame |
| Blender EEVEE title, 540x960 | ~6 s per frame |

Track once and cache; the tracker is the only stage you never need to repeat.

## Limits worth knowing

- The pose mask is inferred at 256×256, so its edge is soft at 1080p. It holds
  up behind type; it will not hold up as a hard key against a busy background.
- `scene3d` renders on CPU and EEVEE goes through software EGL, so there is
  no fast path: budget ~25 s per 1080p frame. Render titles at half size and
  let `sequence_layer(scale=2)` bring them back — a bloomed metal title has
  almost no high-frequency detail to lose.
- RHVoice clamps its rate parameter at natural speed, so slower delivery is a
  time-stretch rather than a synthesis setting.
- Blender here is built without OpenImageDenoise, so Cycles denoising is off
  and low sample counts stay visibly noisy.
