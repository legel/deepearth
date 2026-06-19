# Track B — Viewer UX: mesh-projected textures, playback speed, rain preview

**Status: DONE 2026-06-18.** All 4 tasks below implemented, plus several bugs
found during manual verification (see "Found during verification" at the
bottom). Two of those findings are simulation-physics issues, not viewer bugs
— written up in `plans/track_A_sim_physics.md` for whoever picks up Track A
next, since they need a `flood_sim.py` change + rerun, not a viewer fix.

Repo: `flood_hydrology`. Touches `viewer/static/js/*` and `viewer/server.py`
only. Loosely depends on Track A (richer data — e.g. infiltration frames —
makes item 4 below meaningful) but items 1–3 can start immediately against
the viewer's existing data.

## Why

From the meeting notes (paraphrased, several different phrasings of the same
two asks):

> "Possible to get the layer projected... Mesh layer on top of texture
> layers... Texture on the mesh — so paint on the mesh? Do put the texture —
> the mesh can still show up."
>
> "Press the button we should be able to see rain."
>
> "Simulate every second of rain — every second additional water put into
> the env, and we should be able to see the flow... should be able to see it
> slow down — keep 1 second update time but we can speed up... manipulate the
> speed of the water."

Two distinct asks: (1) textures should be **painted onto the terrain mesh**
geometry, not floating as separate flat planes above it; (2) playback needs a
**speed control** that's decoupled from the underlying 1-second simulation
cadence, plus an explicit way to **trigger/see rain** outside of full scenario
playback.

## Current state (confirmed by reading the JS — start server with
`python3 viewer/server.py`, browse to `http://localhost:5050/` to see it live
before changing anything)

- **Layers are separate flat planes, not mesh-painted textures**:
  - `viewer/static/js/terrain.js` builds the DEM as a `PlaneGeometry` with
    per-vertex height displacement, rendered as a semi-transparent wireframe.
  - `viewer/static/js/overlays.js` builds NAIP and SSURGO as **separate flat
    planes** positioned just above the terrain (y=2–3 in local units).
  - `viewer/static/js/floodLayer.js` is another separate flat plane (y=0.5)
    with a dynamically updated RGBA `DataTexture` for flood depth.
  - `viewer/static/js/main.js` builds the static water-surface plane.
  - This is exactly the "mesh vs. texture" disconnect from the notes — there
    is currently no single mesh with a texture map applied to its surface.
- **Layer selection already works**: `viewer/static/js/layerControls.js`
  renders a checkbox panel (wired up in `main.js`) that toggles each layer's
  `mesh.visible`. The "select different layer/texture" ask is largely already
  satisfied by toggling — what's missing is *combining* mesh geometry with a
  texture (rather than just stacking/toggling independent planes).
- **Rain is fully automatic, not button-triggered**: `rainParticles.js`
  defines a 2000-particle system; `simulationControls.js` line 259–260 calls
  `rainParticles.update(hydrograph.rain_mm_hr[currentFrame], dt)` only while a
  scenario hydrograph is loaded and playing. There is no standalone "show
  rain" control independent of a loaded scenario.
- **Playback is fixed-rate, no speed control**: in
  `viewer/static/js/simulationControls.js`:
  - line 24: `const FPS_PLAYBACK = 5;` — frames per second in animation playback
  - line 25: `const FRAME_MS = 1000 / FPS_PLAYBACK;`
  - Play/Pause/Reset buttons and a scrubber exist (search for those handlers
    in the same file), but `FPS_PLAYBACK` is a hardcoded constant, not exposed
    to the UI.

## Tasks

1. **Paint textures onto the mesh.** In `terrain.js`, change the DEM
   `PlaneGeometry`'s material from wireframe-only to a `THREE.MeshStandardMaterial`
   (or `MeshBasicMaterial` if lighting isn't set up) that accepts a `map`
   texture — reuse the same NAIP/SSURGO PNGs currently loaded by
   `overlays.js` as the texture source, applied directly to the terrain mesh's
   UV-mapped surface instead of (or in addition to — keep both as a toggle
   option) a separate floating plane. Keep the wireframe renderable as an
   optional overlay toggle in `layerControls.js` so the "mesh layer on top of
   texture" combination the notes ask for is achievable: textured mesh as the
   base, wireframe as an optional layer on top.
2. **Speed control.** Add a slider/dropdown to the simulation controls panel
   (wherever `simulationControls.js`'s UI is built/injected — check
   `index.html` for the `#layer-list`-style injection point) offering speed
   multipliers (e.g. 0.25x, 0.5x, 1x, 2x, 4x). Implement by scaling
   `FRAME_MS` at runtime (`FRAME_MS = 1000 / FPS_PLAYBACK / speedMultiplier`)
   rather than changing what each frame represents — the notes are explicit
   that the underlying "1 second update" data cadence should stay fixed; only
   playback speed changes.
3. **Rain preview button.** Add a standalone button that calls
   `rainParticles.update(intensityMmHr, dt)` on a render loop independent of
   `simulationControls`'s scenario-driven path, with a fixed or
   slider-controlled intensity, so rain can be visually sanity-checked without
   loading a full scenario — directly useful for verifying Track A's
   localized-rain work once it's wired up here too (e.g. a future task: only
   show rain particles within the same hilltop region used in Track A).
4. **(Stretch — do after Track A lands `cumulative_infiltration_mm` frames)**
   Add an "Infiltration" entry to `layerControls.js`, reusing the existing
   `floodLayer.js` depth-texture-from-array machinery (the `setFrame()`
   function and its `depthToRGBA` color mapping) pointed at the new
   infiltration array instead of depth.

## Files touched

`viewer/static/js/terrain.js`, `overlays.js`, `simulationControls.js`,
`rainParticles.js`, `layerControls.js`, possibly `viewer/templates/index.html`
for new UI controls.

## Verification

- `python3 viewer/server.py`, open `http://localhost:5050/`.
- Toggle the new mesh-texture rendering on/off, confirm NAIP/SSURGO render
  correctly draped on the terrain's actual elevation (not as a flat floating
  plane), and that the wireframe can still be shown on top.
- Move the speed slider during scenario playback, confirm visually that frame
  advancement rate changes (faster/slower) while total scenario duration in
  the time display stays correct.
- Click the rain preview button outside of scenario playback, confirm rain
  particles render at the chosen intensity with no scenario loaded.

## Found during verification (2026-06-18)

Beyond the 4 tasks above, manual verification turned up — and this session
fixed — several more issues:

1. **Flood Depth was never actually draped** — it built its own flat,
   undisplaced `PlaneGeometry` near `z_min`, so toggling it on always showed
   water as a flat sheet regardless of terrain shape. Fixed by giving
   `floodLayer.js` an optional `terrainGeometry` parameter (reusing the exact
   same shared geometry as the draped overlays); both Flood Depth and the new
   Infiltration layer now pass `terrain.geometry` from `main.js`.
2. **"Ponds suddenly appear" instead of a gradual fill** — `floodLayer.js`
   clamped anything under 1cm depth to fully transparent. Checked the actual
   exported data: under the default full-AOI rain mask, ~100% of the grid has
   *some* water within ~10–15 minutes, just sub-cm almost everywhere except
   in depressions. Lowered the floor to 0.1mm and added two new low-alpha
   colormap buckets (<1mm, 1mm–1cm) so the thin film fades in instead of
   popping at one hard threshold.
3. **Rain particles invisible** — `PARTICLE_SIZE` was 1.4 against a ~2000m
   scene; bumped to 6.0.
4. **Layer list grouping** — NAIP/SSURGO and their `(Draped)` variants
   weren't adjacent in the legend; reordered in `main.js`.
5. **Panels ate ~1/3 of viewport width** — `#controls-panel` and
   `#simulation-panel` sat side by side (220px + 240px). Wrapped both in a
   single `#left-stack` flex column (shared 240px width, internal scroll) in
   `index.html` / `styles.css`.

Two findings turned out to be **simulation-physics**, not viewer, issues —
not fixed here, written up as a Track A follow-up instead:

- Infiltration is spatially uniform (every cell identical at every
  timestep) under the current `--rain-mask full` + spatially-uniform Horton
  model, so the colored layer is real data but has zero spatial structure —
  it just looks like a flat tint. See `plans/track_A_sim_physics.md`.
- No scenario ever captures post-storm flood recession in its saved frames
  (the post-storm drainage loop in `flood_sim.py` doesn't call
  `frame_depths.append`). See `plans/track_A_sim_physics.md`.
