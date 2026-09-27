# Showcase media

The images here are used by the top-level README. The same sources also feed the portfolio
page (`JSS6088.github.io/projects/autostroke/`). Record or render each asset once, then run
`tools/encode_media.sh` to produce every format both places need.

| File | Made by |
|---|---|
| `body_before.png`, `body_after.png` | `tools/render_showcase.py`: a fresh bake of Body, then EEVEE renders |
| `hero.jpg`, `social_preview.jpg` | `tools/encode_media.sh`, cropped from the renders |
| `live_preview.gif` | `tools/encode_media.sh <recording>`, from your screen recording |
| `panel.png` | `tools/encode_media.sh <recording> <screenshot>`, from your screenshot |

`encode_media.sh` also writes `portfolio/` (git-ignored), laid out like the portfolio
repo's `assets/`. Copy its contents there, then uncomment the page's `TODO(media)` blocks
and the `card:` line in `_data/projects.yml`.

## Re-rendering the stills

```bash
"$BLENDER" --factory-startup PainterlyTexture.blend --python tools/render_showcase.py -- docs/media Body
tools/encode_media.sh
```

The script bakes into a temporary folder and never saves the `.blend`. If you change the
camera, re-tune the crop offsets at the top of `encode_media.sh`.

## Recording checklist

**Showcase clip.** One recording becomes both the README GIF and the portfolio's mp4:
- 3D viewport around 1600×900, sidebar open on the AutoStroke tab, Body selected.
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
