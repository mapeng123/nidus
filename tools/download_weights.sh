#!/bin/bash
# Download and reassemble Nidus model weights from the GitHub release.
# Usage: bash tools/download_weights.sh [output_dir]   (default: model/)
set -e
OUT=${1:-model}
BASE=https://github.com/mapeng123/nidus/releases/download/v1.0.0
NAME=pmdm_v3_oponly_step2000.pt
mkdir -p "$OUT" tmp_weights
for i in $(seq -w 0 13); do
  curl -L -o tmp_weights/$NAME.part$i $BASE/$NAME.part$i
done
cat tmp_weights/$NAME.part* > "$OUT/$NAME"
echo "1c6c09b56401936331ae91f674394fb3  $OUT/$NAME" | md5sum -c - && rm -rf tmp_weights
echo "Done: $OUT/$NAME"
