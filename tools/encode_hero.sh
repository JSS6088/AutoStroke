#!/usr/bin/env bash
# encode_hero.sh -- turn render_hero.py's frame folders into files an editor can use.
#
#   tools/encode_hero.sh <out_dir>
#
# For every <out_dir>/<model>/<aspect>/<shot>/ folder of PNG frames:
#   <shot>.mov           ProRes 4444 WITH alpha, for editing (Premiere, DaVinci, Final Cut,
#                        CapCut all read it) -- drop any background under it
#   <shot>_preview.mp4   H.264 on a neutral grey, for a quick look
# For models marked "hero" in report.json:
#   wipe.mov / wipe_preview.mp4   before -> after, a left-to-right wipe mid-turn. The
#                        "after" stream is trimmed to the same moment, so both halves are
#                        at the same angle -- the model keeps turning straight through it.
# And per model and aspect:
#   contact.jpg          first / middle / last frame of every shot, to eyeball before editing
#
# Needs ffmpeg with prores_ks and xfade (drawtext is NOT needed -- labels go in the editor).

set -euo pipefail

OUT="${1:?usage: tools/encode_hero.sh <out_dir>}"
FPS=30
WIPE_AT=0.3        # fraction of the turn at which the wipe starts
WIPE_LEN=0.3       # fraction of the turn the wipe takes
BG=0x2a2a2a
FF=(ffmpeg -hide_banner -loglevel error -y)

command -v ffmpeg >/dev/null || { echo "error: ffmpeg not found" >&2; exit 1; }

heroes=""
if [ -f "$OUT/report.json" ]; then
    heroes=$(python3 -c "import json,sys; r=json.load(open(sys.argv[1])); \
print(' '.join(k for k, v in r.items() if v.get('hero')))" "$OUT/report.json")
fi

prores() {   # <frames dir> <out.mov>
    "${FF[@]}" -framerate $FPS -i "$1/%04d.png" -c:v prores_ks -profile:v 4444 \
        -pix_fmt yuva444p10le -vendor apl0 "$2"
}

preview() {  # <frames dir | .mov> <out.mp4>: the shot over a neutral grey, H.264
    local in="$1" out="$2" args probe size
    if [ -d "$in" ]; then
        args=(-framerate $FPS -i "$in/%04d.png"); probe="$in/0001.png"
    else
        args=(-i "$in"); probe="$in"
    fi
    size=$(ffprobe -v error -select_streams v:0 -show_entries stream=width,height \
        -of csv=p=0:s=x "$probe")
    "${FF[@]}" "${args[@]}" -filter_complex \
        "color=c=$BG:s=$size:r=$FPS[bg];[bg][0:v]overlay=shortest=1,format=yuv420p" \
        -c:v libx264 -crf 20 -preset slow -movflags +faststart "$out"
}

for model_dir in "$OUT"/*/; do
    model=$(basename "$model_dir")
    for asp_dir in "$model_dir"*/; do
        [ -d "$asp_dir" ] || continue
        asp=$(basename "$asp_dir")
        rows=()
        for shot_dir in "$asp_dir"*/; do
            shot=$(basename "$shot_dir")
            [ -f "$shot_dir/0001.png" ] || continue
            prores "$shot_dir" "$asp_dir/$shot.mov"
            preview "${shot_dir%/}" "$asp_dir/${shot}_preview.mp4"
            n=$(ls "$shot_dir" | grep -c '\.png$')
            mid=$(printf "%04d" $(( (n + 1) / 2 )))
            last=$(printf "%04d" "$n")
            "${FF[@]}" -i "$shot_dir/0001.png" -i "$shot_dir/$mid.png" -i "$shot_dir/$last.png" \
                -filter_complex "[0][1][2]hstack=inputs=3,scale=1440:-2" "$asp_dir/.row_$shot.png"
            rows+=(-i "$asp_dir/.row_$shot.png")
        done
        if [ ${#rows[@]} -gt 0 ]; then
            k=$(( ${#rows[@]} / 2 ))
            if [ "$k" -gt 1 ]; then
                "${FF[@]}" "${rows[@]}" -filter_complex "vstack=inputs=$k" -q:v 3 "$asp_dir/contact.jpg"
            else
                "${FF[@]}" "${rows[@]}" -q:v 3 "$asp_dir/contact.jpg"
            fi
            rm -f "$asp_dir"/.row_*.png
        fi
        if [[ " $heroes " == *" $model "* ]] && [ -f "$asp_dir/before.mov" ] && [ -f "$asp_dir/after.mov" ]; then
            n=$(ls "$asp_dir/before" | grep -c '\.png$')
            at=$(python3 -c "print(round($n * $WIPE_AT / $FPS, 3))")
            len=$(python3 -c "print(round($n * $WIPE_LEN / $FPS, 3))")
            "${FF[@]}" -i "$asp_dir/before.mov" -i "$asp_dir/after.mov" -filter_complex \
                "[1:v]trim=start=$at,setpts=PTS-STARTPTS[a];[0:v][a]xfade=transition=wipeleft:duration=$len:offset=$at,format=yuva444p10le" \
                -c:v prores_ks -profile:v 4444 -pix_fmt yuva444p10le -vendor apl0 "$asp_dir/wipe.mov"
            preview "$asp_dir/wipe.mov" "$asp_dir/wipe_preview.mp4"
        fi
        echo "$model/$asp: $(ls "$asp_dir" | grep -E '\.(mov|mp4|jpg)$' | tr '\n' ' ')"
    done
done
