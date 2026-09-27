# Showcase media

The images here are used by the top-level README. The same sources also feed the portfolio
page (`JSS6088.github.io/projects/autostroke/`). Record or render each asset once, then run
`tools/encode_media.sh` to produce every format both places need.

| File | Made by |
|---|---|
| `suzanne_before.png`, `suzanne_after.png` | Viewport screenshots of Suzanne, plain and baked. `tools/render_showcase.py` can make rendered versions instead. |
| `hero.jpg`, `social_preview.jpg` | `tools/encode_media.sh`, from the before/after pair (any matching size) |
| `live_preview.gif` | `tools/encode_media.sh <recording>`, from your screen recording |
| `panel.png` | `tools/encode_media.sh <recording> <screenshot>`, from your screenshot |

`encode_media.sh` also writes `portfolio/` (git-ignored), laid out like the portfolio
repo's `assets/`. Copy its contents there, then uncomment the page's `TODO(media)` blocks
and the `card:` line in `_data/projects.yml`.

## Rendered stills instead of screenshots

```bash
"$BLENDER" --factory-startup PainterlyTexture.blend --python tools/render_showcase.py \
    -- docs/media "Suzanne@-0.7,-1,-0.2" sun=2 world=0.7
tools/encode_media.sh
```

The script bakes into a temporary folder and never saves the `.blend`.
- The view is front, slightly left and below.
- The sun is doubled and the world light dimmed, so the render has a strong light-to-shadow
  range where the strokes read clearly.
- `HERO=<name> tools/encode_media.sh` encodes a different object's renders.

## Recording checklist

**Showcase clip.** One recording becomes both the README GIF and the portfolio's mp4:
- 3D viewport around 1600×900, sidebar open on the AutoStroke tab, Suzanne selected.
- 8–12 seconds:
  1. Live Preview ON.
  2. Drag **Stroke Count**.
  3. Drag **Stroke Size**.
  4. Drag **Global Rotation**.
  5. Press **Bake**, and hold on the finished material for a second.
- Keep the mouse movements slow. The GIF runs at 12 fps.
- Save it as a `.mov` or `.mp4`, then run `tools/encode_media.sh ~/Desktop/clip.mov`.

**Panel screenshot:**
- Sidebar visible, Live Preview ON, a bake report showing.
- Crop to just the panel (portrait). Pass it as the second argument.

**Size targets:**
- GIF ≤ 8 MB. If it's over, shorten the clip or drop to 10 fps in the script.
- Each still ≤ 1 MB.
