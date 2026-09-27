#!/usr/bin/env bash
# encode_media.sh -- turn the showcase stills and one raw screen recording into every
# format the README and the portfolio page use. Needs ffmpeg.
#
#   tools/encode_media.sh [recording.mov] [panel_screenshot.png]
#
# Always (from docs/media/$HERO_before.png + $HERO_after.png, made by render_showcase.py;
# HERO defaults to suzanne):
#   docs/media/hero.jpg               before | after side by side, for the README
#   docs/media/social_preview.jpg     1280x640, for GitHub's repo social preview
#   docs/media/portfolio/images/AutoStroke/AutoStroke_painterly_off.png, _on.png
#   docs/media/portfolio/images/AutoStroke/AutoStroke_card.jpg      960x540 home-page card
# With a recording:
#   docs/media/live_preview.gif       800 px wide, 12 fps, for the README
#   docs/media/portfolio/videos/AutoStroke/AutoStroke_Showcase.mp4 + .jpg poster
# With a panel screenshot:
#   docs/media/panel.png  and  docs/media/portfolio/images/AutoStroke/AutoStroke_panel.png
#
# docs/media/portfolio/ mirrors the portfolio repo's assets/ layout and is git-ignored:
# copy its contents into JSS6088.github.io/assets/ rather than committing duplicates here.

set -euo pipefail

REPO="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
MEDIA="$REPO/docs/media"
PORT_IMG="$MEDIA/portfolio/images/AutoStroke"
PORT_VID="$MEDIA/portfolio/videos/AutoStroke"
REC="${1:-}"
PANEL="${2:-}"
FF=(ffmpeg -hide_banner -loglevel error -y)

command -v ffmpeg >/dev/null || { echo "error: ffmpeg not found" >&2; exit 1; }
HERO="${HERO:-suzanne}"
BEFORE="$MEDIA/${HERO}_before.png"
AFTER="$MEDIA/${HERO}_after.png"
for f in "${HERO}_before.png" "${HERO}_after.png"; do
    [ -f "$MEDIA/$f" ] || { echo "error: $MEDIA/$f missing -- run tools/render_showcase.py" >&2; exit 1; }
done
mkdir -p "$PORT_IMG" "$PORT_VID"

# Stills, from before/after frames of any (matching) size -- a render_showcase.py render
# or a viewport screenshot. The hero puts them side by side; the card and social preview
# fit the "after" frame and pad it with its own background grey (sampled from a corner).
BG="0x$(ffmpeg -hide_banner -loglevel error -i "$AFTER" -vf "crop=1:1:5:5" \
    -f rawvideo -pix_fmt rgb24 - | xxd -p)"
"${FF[@]}" -i "$BEFORE" -i "$AFTER" \
    -filter_complex "[0][1]hstack=inputs=2,scale='min(1800,iw)':-2" -q:v 2 "$MEDIA/hero.jpg"
"${FF[@]}" -i "$AFTER" \
    -vf "scale=1280:640:force_original_aspect_ratio=decrease,pad=1280:640:(ow-iw)/2:(oh-ih)/2:color=$BG" \
    -q:v 2 "$MEDIA/social_preview.jpg"
"${FF[@]}" -i "$AFTER" \
    -vf "scale=960:540:force_original_aspect_ratio=decrease,pad=960:540:(ow-iw)/2:(oh-ih)/2:color=$BG" \
    -q:v 2 "$PORT_IMG/AutoStroke_card.jpg"
cp "$BEFORE" "$PORT_IMG/AutoStroke_painterly_off.png"
cp "$AFTER"  "$PORT_IMG/AutoStroke_painterly_on.png"

if [ -n "$REC" ]; then
    # mp4: H.264, no audio, yuv420p for Safari, faststart so it plays while loading.
    "${FF[@]}" -i "$REC" -an -vf "scale='min(1600,iw)':-2,fps=30" -c:v libx264 -crf 23 \
        -preset slow -pix_fmt yuv420p -movflags +faststart "$PORT_VID/AutoStroke_Showcase.mp4"
    "${FF[@]}" -i "$PORT_VID/AutoStroke_Showcase.mp4" -frames:v 1 -q:v 3 \
        "$PORT_VID/AutoStroke_Showcase.jpg"
    # GIF: two-pass palette, so the strokes' greys don't band.
    "${FF[@]}" -i "$REC" -vf "fps=12,scale=800:-2:flags=lanczos,split[a][b];[a]palettegen=stats_mode=diff[p];[b][p]paletteuse=dither=bayer:bayer_scale=4" \
        "$MEDIA/live_preview.gif"
fi

if [ -n "$PANEL" ]; then
    cp "$PANEL" "$MEDIA/panel.png"
    cp "$PANEL" "$PORT_IMG/AutoStroke_panel.png"
fi

echo "wrote:"
find "$MEDIA" -type f \( -name '*.jpg' -o -name '*.png' -o -name '*.gif' -o -name '*.mp4' \) \
    -newer "$AFTER" -exec ls -lh {} \; | awk '{print "  " $5 "\t" $NF}'
