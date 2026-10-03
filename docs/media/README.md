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

## Hero video

A 25–30 s showcase for the portfolio page, the README and LinkedIn. It's built from raw
shots that the scripts render and you cut to music in your own editor.

**Shot list:**

| Time | Shot | Source |
|---|---|---|
| 0–3 s | Hook: the hero model turning, wiped from plain to painterly | `wipe.mp4` |
| 3–12 s | Montage, ~1.5 s per model on the beat: organic, hard surface, stylised | `after.mp4` per model |
| 12–17 s | One model, a different brush set on each beat | `brush_<set>.mp4` |
| 17–23 s | Live preview: viewport + panel, dragging Stroke Size and Rotation | your screen recording |
| 23–27 s | Fast bake: select several models, one Bake, the report's time on screen | your screen recording |
| 27–30 s | End card: name, "AutoStroke · painterly texturing for Blender", link | your editor |

Cut two versions from the same shots:
- **16:9, 12–15 s, looping, no end card:** the portfolio's `AutoStroke_Showcase.mp4` slot
  and the README GIF.
- **4:5, 25–30 s, with the end card:** LinkedIn.

**1. List your models.**
- Copy `tools/showcase/shots.example.json` to `tools/showcase/shots.json`, which is
  git-ignored, and point it at 5–8 models with UVs.
- Aim for a mix of organic, hard surface, and at least one with an albedo texture for
  colour shots.
- Relative paths resolve from the folder you run Blender in.

**2. Check, then render:**

```bash
"$BLENDER" --factory-startup --python tools/render_hero.py -- ~/HeroShots --check
"$BLENDER" --factory-startup --python tools/render_hero.py -- ~/HeroShots
```

- `--check` imports each model and validates it (file, UVs, albedo, brush sets) without
  rendering.
- A full run bakes each model with the real Bake (GPU) at 2K and renders 5 s looping
  turntables with a transparent background, in **16:9 (1920×1080) and 4:5 (1080×1350)**.
- The lights are fixed while the model turns, so strokes catch the light.
- `--only a,b`, `--aspect 16x9` and `--frames 30` are for quick tests.
- Nothing is saved to any `.blend`.
- Bake times per model go to `~/HeroShots/report.json`. Use these for the fast-bake shot.

**3. Encode:**

```bash
tools/encode_hero.sh ~/HeroShots
```

Per model and aspect this writes:

| File | Use |
|---|---|
| `<shot>.mp4` | H.264, for a quick look or a straight cut; transparent frames go over grey |
| `<shot>.mov` | ProRes 4444 with alpha, for editing over any background. Only with `--prores`: about 1 MB per frame |
| `wipe.mp4` | before → after, for models with `"hero": true` or named in `--wipe Car,Die` |
| `contact.jpg` | first, middle and last frame of every shot, to check before you edit |

- It only encodes what is missing or older than its frames. `--force` redoes everything.
- Frames can also sit one level up, in `<out>/<model>/<shot>/`. The files then land in the
  model's folder as `<model>_<shot>.mp4`, `<model>_wipe.mp4` and `<model>_contact.jpg`.

Each model's shots are `before` and `after`, plus `brush_<set>` if you listed extra sets.
"After" breaks the model's albedo into flat per-stroke colour, or shows the grey stroke
material if the model has no albedo.

### From a scene you've already baked and look-dev'd

If your models are already baked and dressed in a showcase `.blend` with your own lights,
render straight from it. Nothing is re-baked and no material changes. Open the file with
`tools/render_scene.py` and name one model per run; with no name it lists the models it
can render:

```bash
"$BLENDER" --factory-startup ~/Desktop/Showcase.blend --python tools/render_scene.py \
    -- ~/HeroShots Car --wipe --check
"$BLENDER" --factory-startup ~/Desktop/Showcase.blend --python tools/render_scene.py \
    -- ~/HeroShots Car --wipe
"$BLENDER" --factory-startup ~/Desktop/Showcase.blend --python tools/render_scene.py \
    -- ~/HeroShots DeadWood2
tools/encode_hero.sh ~/HeroShots
```

Frames are 16:9 (1920×1080). `--wipe` gets that model a before → after `wipe.mp4`; each
run adds its model to `report.json` rather than replacing it. If you move or rename the
folders afterwards, name the models on the encode instead:
`tools/encode_hero.sh ~/HeroShots --wipe Car,Die`.

**What stays as you set it:**
- your lights, world, render engine and colour management;
- your materials, in the `after` shots.

**What changes, in memory only (the `.blend` is never saved):**
- The named mesh renders on its own; the others are hidden from the render.
- The model turns about its own centre while your lights stay fixed.
- It's shot through your camera named `CAM_<model>` if the file has one
  (`tools/compose_cameras.py` places starting ones to compose). Otherwise a camera is fitted
  to the whole turn: a raised three-quarter view by default, or `--view -1,-1,0.2`.
- Textures whose files moved are relinked from `--search` (default `~/Downloads`).
- The pre-0.10 AutoStroke geometry-nodes modifier, if a model still has one, is switched
  off.

**`before` is each material without AutoStroke:**
- textures read through the indirection map go back to their own UVs;
- the per-stroke tone is neutral;
- the stroke normal is unplugged, and your own normal map goes back in if it's still in
  the tree.

Every node you made is kept, so the wipe shows exactly what AutoStroke adds.

Other options: `--frames 30` for a quick test, `--transparent`, `--no-before`.

**Screen recordings** (for the live-preview and fast-bake shots):
- Record the viewport and sidebar at 1920×1080. For the 4:5 cut, you'll crop around the
  model and the panel.
- Slow slider drags.
- For the bake shot, select several models, press Bake once, and hold on the report line
  showing the time.
