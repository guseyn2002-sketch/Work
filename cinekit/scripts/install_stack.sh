#!/usr/bin/env bash
# Everything cinekit shells out to. Debian/Ubuntu.
set -euo pipefail
sudo apt-get update -qq
sudo apt-get install -y \
  ffmpeg blender imagemagick potrace \
  fluidsynth fluid-soundfont-gm \
  rhvoice rhvoice-russian rhvoice-english \
  libgles2 libegl1 libgl1
pip install numpy scipy pillow opencv-python-headless mediapipe \
  pedalboard librosa soundfile scikit-image fonttools mido playwright
python3 -m playwright install chromium || true
