#!/usr/bin/env bash
# encode_hero.sh -- turn rendered frame folders into clips an editor can use.
#
#   tools/encode_hero.sh <out_dir> [--wipe Car,Die] [--prores] [--force]
#
# Reads either layout, per model folder:
#   <out_dir>/<model>/<shot>/####.png            files land in <model>/ as <model>_<shot>.*
#   <out_dir>/<model>/<aspect>/<shot>/####.png   (what render_scene.py and render_hero.py
#                                                write) files land in <aspect>/ as <shot>.*
#
# For every shot:
#   .mp4          H.264, over a neutral grey where the frames are transparent
#   .mov          ProRes 4444 WITH alpha, only with --prores (large: ~1 MB per frame)
# For models named in --wipe, or marked "hero" in <out_dir>/report.json:
#   wipe.mp4      before -> after, a wipe mid-turn (and wipe.mov with --prores). The "after"
#                 stream is trimmed to the same moment, so both halves are at the same
#                 angle -- the model keeps turning straight through it.
# And per model:
#   contact.jpg   first / middle / last frame of every shot, to eyeball before editing
#
# Only what is missing or older than its frames is encoded; --force redoes everything.
# Needs ffmpeg with xfade (and prores_ks for --prores). drawtext is NOT needed -- labels
# go in the editor.

set -euo pipefail

USAGE="usage: tools/encode_hero.sh <out_dir> [--wipe a,b] [--prores] [--force]"
OUT="${1:?$USAGE}"
shift
PRORES=0
FORCE=0
WIPES=""
while [ $# -gt 0 ]; do
    case "$1" in
        --prores) PRORES=1 ;;
        --force) FORCE=1 ;;
        --wipe) WIPES="${2:?$USAGE}"; shift ;;
        *) echo "$USAGE" >&2; exit 1 ;;
    esac
    shift
done

FPS=30
WIPE_AT=0.3        # fraction of the turn at which the wipe starts
WIPE_LEN=0.3       # fraction of the turn the wipe takes
BG=0x2a2a2a
FF=(ffmpeg -hide_banner -loglevel error -y)
X264=(-c:v libx264 -crf 20 -preset slow -movflags +faststart)
PRO=(-c:v prores_ks -profile:v 4444 -pix_fmt yuva444p10le -vendor apl0)

command -v ffmpeg >/dev/null || { echo "error: ffmpeg not found" >&2; exit 1; }
[ -d "$OUT" ] || { echo "error: no such folder: $OUT" >&2; exit 1; }

heroes=" ${WIPES//,/ } "
if [ -f "$OUT/report.json" ]; then
    heroes="$heroes$(python3 -c "import json,sys; r=json.load(open(sys.argv[1])); \
print(' '.join(k for k, v in r.items() if v.get('hero')))" "$OUT/report.json") "
fi

stale() {    # <output> <frames dir>...: true when the output has to be (re)made
    local out="$1" d
    shift
    if [ "$FORCE" = 1 ] || [ ! -f "$out" ]; then return 0; fi
    for d in "$@"; do
        if [ -n "$(find "$d" -maxdepth 1 -name '*.png' -newer "$out" -print -quit)" ]; then
            return 0
        fi
    done
    return 1
}

count() { ls "$1" | grep -c '\.png$'; }

on_grey() {  # <frames dir>: filter that lays the stream labelled [v] over the grey
    local size
    size=$(ffprobe -v error -select_streams v:0 -show_entries stream=width,height \
        -of csv=p=0:s=x "$1/0001.png")
    echo "color=c=$BG:s=$size:r=$FPS[bg];[bg][v]overlay=shortest=1,format=yuv420p"
}

wipe() {     # <before dir> <after dir> <output without extension>
    local n at len x
    n=$(count "$1")
    at=$(python3 -c "print(round($n * $WIPE_AT / $FPS, 3))")
    len=$(python3 -c "print(round($n * $WIPE_LEN / $FPS, 3))")
    x="[0:v]format=gbrap[b];[1:v]format=gbrap,trim=start=$at,setpts=PTS-STARTPTS[a]"
    x="$x;[b][a]xfade=transition=wipeleft:duration=$len:offset=$at"
    "${FF[@]}" -framerate $FPS -i "$1/%04d.png" -framerate $FPS -i "$2/%04d.png" \
        -filter_complex "$x[v];$(on_grey "$1")" "${X264[@]}" "$3.mp4"
    if [ "$PRORES" = 1 ]; then
        "${FF[@]}" -framerate $FPS -i "$1/%04d.png" -framerate $FPS -i "$2/%04d.png" \
            -filter_complex "$x,format=yuva444p10le" "${PRO[@]}" "$3.mov"
    fi
}

encode_group() {   # <label> <model> <dest dir> <file prefix> <shot dir>...
    local label="$1" model="$2" dest="$3" pre="$4"
    shift 4
    local made="" dir shot n mid last tmp before="" after=""
    local rows=()
    for dir in "$@"; do
        dir="${dir%/}"
        shot=$(basename "$dir")
        case "$(echo "$shot" | tr '[:upper:]' '[:lower:]')" in
            before) before="$dir" ;;
            after) after="$dir" ;;
        esac
        if stale "$dest/$pre$shot.mp4" "$dir"; then
            "${FF[@]}" -framerate $FPS -i "$dir/%04d.png" \
                -filter_complex "[0:v]null[v];$(on_grey "$dir")" "${X264[@]}" "$dest/$pre$shot.mp4"
            made="$made $pre$shot.mp4"
        fi
        if [ "$PRORES" = 1 ] && stale "$dest/$pre$shot.mov" "$dir"; then
            "${FF[@]}" -framerate $FPS -i "$dir/%04d.png" "${PRO[@]}" "$dest/$pre$shot.mov"
            made="$made $pre$shot.mov"
        fi
    done

    if stale "$dest/${pre}contact.jpg" "$@"; then
        tmp=$(mktemp -d)
        for dir in "$@"; do
            dir="${dir%/}"
            shot=$(basename "$dir")
            n=$(count "$dir")
            mid=$(printf "%04d" $(( (n + 1) / 2 )))
            last=$(printf "%04d" "$n")
            "${FF[@]}" -i "$dir/0001.png" -i "$dir/$mid.png" -i "$dir/$last.png" \
                -filter_complex "[0][1][2]hstack=inputs=3,scale=1440:-2" "$tmp/$shot.png"
            rows+=(-i "$tmp/$shot.png")
        done
        if [ $# -gt 1 ]; then
            "${FF[@]}" "${rows[@]}" -filter_complex "vstack=inputs=$#" -q:v 3 "$dest/${pre}contact.jpg"
        else
            "${FF[@]}" "${rows[@]}" -q:v 3 "$dest/${pre}contact.jpg"
        fi
        rm -rf "$tmp"
        made="$made ${pre}contact.jpg"
    fi

    if [[ "$heroes" == *" $model "* ]]; then
        if [ -z "$before" ] || [ -z "$after" ]; then
            echo "$label: no wipe -- it needs both a before and an after shot" >&2
        elif [ "$(count "$before")" != "$(count "$after")" ]; then
            echo "$label: no wipe -- before and after have different frame counts" >&2
        elif stale "$dest/${pre}wipe.mp4" "$before" "$after" || \
                { [ "$PRORES" = 1 ] && stale "$dest/${pre}wipe.mov" "$before" "$after"; }; then
            wipe "$before" "$after" "$dest/${pre}wipe"
            made="$made ${pre}wipe.mp4"
            if [ "$PRORES" = 1 ]; then made="$made ${pre}wipe.mov"; fi
        fi
    fi
    echo "$label:${made:- up to date}"
}

for model_dir in "$OUT"/*/; do
    [ -d "$model_dir" ] || continue
    model=$(basename "$model_dir")
    flat=()
    for d in "$model_dir"*/; do
        [ -d "$d" ] || continue
        if [ -f "$d/0001.png" ]; then
            flat+=("$d")
            continue
        fi
        shots=()
        for s in "$d"*/; do
            if [ -f "$s/0001.png" ]; then shots+=("$s"); fi
        done
        if [ ${#shots[@]} -gt 0 ]; then
            encode_group "$model/$(basename "$d")" "$model" "${d%/}" "" "${shots[@]}"
        fi
    done
    if [ ${#flat[@]} -gt 0 ]; then
        encode_group "$model" "$model" "${model_dir%/}" "${model}_" "${flat[@]}"
    fi
done
