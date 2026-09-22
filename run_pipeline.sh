#!/usr/bin/env bash
# run_pipeline.sh — single entry point for the AutoStroke bake.
#
# Runs the three manual steps in order:
#   1. export_seeds.py   (inside Blender, bpy)
#   2. bake_position.py  (inside Blender, bpy)   -- same Blender session as (1)
#   3. bake_indirection.py (terminal python, no bpy)
#
# Usage:
#   ./run_pipeline.sh
#   AUTOSTROKE_DIR=/other/folder ./run_pipeline.sh
#
# Override the executables below if your Blender/python install moves.

set -euo pipefail

BASE_DIR="${AUTOSTROKE_DIR:-$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)}"
BLEND_FILE="${AUTOSTROKE_BLEND:-$BASE_DIR/PainterlyTexture.blend}"
BLENDER="${AUTOSTROKE_BLENDER:-$HOME/Library/Application Support/Steam/steamapps/common/Blender/Blender.app/Contents/MacOS/Blender}"
PYTHON3="${AUTOSTROKE_PYTHON:-/Applications/Xcode.app/Contents/Developer/usr/bin/python3}"

export AUTOSTROKE_DIR="$BASE_DIR"

for exe_desc in "Blender:$BLENDER" "python3:$PYTHON3"; do
    name="${exe_desc%%:*}"; path="${exe_desc#*:}"
    if [ ! -x "$path" ]; then
        echo "error: $name not found or not executable at: $path" >&2
        echo "  set AUTOSTROKE_BLENDER / AUTOSTROKE_PYTHON to override" >&2
        exit 1
    fi
done
if [ ! -f "$BLEND_FILE" ]; then
    echo "error: blend file not found: $BLEND_FILE" >&2
    exit 1
fi

echo "== AutoStroke pipeline =="
echo "data dir : $BASE_DIR"
echo "blend    : $BLEND_FILE"
echo

echo "-- stage 1+2: export_seeds.py, bake_position.py (Blender) --"
"$BLENDER" -b "$BLEND_FILE" \
    --python "$BASE_DIR/export_seeds.py" \
    --python "$BASE_DIR/bake_position.py"
echo

echo "-- stage 3: bake_indirection.py (terminal python) --"
"$PYTHON3" "$BASE_DIR/bake_indirection.py"
echo

echo "== done =="
