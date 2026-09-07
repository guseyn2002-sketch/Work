#!/usr/bin/env bash
# Fetch the model files cinekit needs. They are not committed: ~42 MB of
# binaries that Google serves directly and that change independently of this code.
set -euo pipefail
DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)/assets/models"
mkdir -p "$DIR"
B=https://storage.googleapis.com/mediapipe-models
get(){ [ -s "$DIR/$2" ] || curl -sSL --max-time 120 -o "$DIR/$2" "$1"; printf '  %-28s %s\n' "$2" "$(du -h "$DIR/$2" | cut -f1)"; }
echo "fetching MediaPipe models -> $DIR"
get $B/pose_landmarker/pose_landmarker_lite/float16/1/pose_landmarker_lite.task pose_landmarker.task
get $B/face_detector/blaze_face_short_range/float16/1/blaze_face_short_range.tflite blaze_face.tflite
get $B/face_landmarker/face_landmarker/float16/1/face_landmarker.task face_landmarker.task
get $B/image_segmenter/selfie_segmenter/float16/1/selfie_segmenter.tflite selfie_segmenter.tflite
get $B/image_segmenter/selfie_multiclass_256x256/float32/1/selfie_multiclass_256x256.tflite selfie_multiclass.tflite
get $B/image_segmenter/deeplab_v3/float32/1/deeplab_v3.tflite deeplab_v3.tflite
get $B/object_detector/efficientdet_lite0/float32/1/efficientdet_lite0.tflite efficientdet_lite0.tflite
echo "building LUTs"
python3 -c "from cinekit.grade import build_lut, LOOKS; [build_lut(k) for k in LOOKS]; print('  %d LUTs' % len(LOOKS))"
