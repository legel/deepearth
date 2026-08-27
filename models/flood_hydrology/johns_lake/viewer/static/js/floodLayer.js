/**
 * floodLayer.js — Animated flood depth texture overlay
 *
 * Loads per-timestep depth frames from simulation_{scenario}_frames.bin,
 * renders as a colored transparent texture draped on the terrain's actual
 * displaced surface (pass `terrainGeometry` in options) — falls back to a
 * flat plane near z_min if no terrain geometry is supplied.
 *
 * Binary format (written by viewer/preprocess/export_simulation.py):
 *   [0:4]            b'SIML'        magic
 *   [4:8]            uint32         n_frames
 *   [8:12]           uint32         rows (256)
 *   [12:16]          uint32         cols (256)
 *   [16:16+n*4]      float32[n]     times_min
 *   [16+n*4:]        float32[n*r*c] depth values [m]  (0 = dry)
 *
 * Colormap (based on real simulated depth — no fake values). Graduated at
 * the low end so a thin, near-uniform sheet (e.g. early minutes under a
 * full-AOI rain mask) fades in gradually instead of popping into existence
 * only once a depression crosses a single hard threshold:
 *   0 m          → fully transparent
 *   0–1 mm      → trace film,   alpha 0.06
 *   1 mm–1 cm   → thin film,    alpha 0.16
 *   1 cm–0.1 m → light blue,    alpha 0.30
 *   0.1–0.5 m   → medium blue,   alpha 0.55
 *   0.5–1.5 m   → deep blue,     alpha 0.72
 *   >1.5 m      → indigo/violet, alpha 0.88
 *
 * Usage:
 *   const flood = createFloodLayer(scene, geoMeta);
 *   await flood.loadScenario('flash_1hr_100yr');
 *   flood.setFrame(5);     // update texture
 *   flood.dispose();
 *
 * Also reusable for other per-cell scalar fields (e.g. cumulative infiltration)
 * via the `options` argument — same SIML binary format, same setFrame()/
 * depthToRGBA() machinery, different source file + mesh identity + value scale:
 *   const infiltration = createFloodLayer(scene, geoMeta, {
 *     urlSuffix: 'infiltration', meshName: 'Infiltration',
 *     yOffset: 0.6, valueScale: 0.001,   // mm → m, to match depthToRGBA's metre buckets
 *   });
 */

import * as THREE from 'three';

const ROWS = 256;
const COLS = 256;
const VERT_EXAG = 8;

// Depth → RGBA colormap
function depthToRGBA(d, out, offset) {
  if (d <= 0) {
    out[offset]     = 0;
    out[offset + 1] = 0;
    out[offset + 2] = 0;
    out[offset + 3] = 0;
    return;
  }
  let r, g, b, a;
  if (d < 0.001) {
    r = 120; g = 200; b = 255; a = 15;
  } else if (d < 0.01) {
    r = 120; g = 200; b = 255; a = 40;
  } else if (d < 0.1) {
    r = 120; g = 200; b = 255; a = 80;
  } else if (d < 0.5) {
    r = 50;  g = 140; b = 230; a = 140;
  } else if (d < 1.5) {
    r = 20;  g = 80;  b = 200; a = 185;
  } else {
    r = 60;  g = 20;  b = 160; a = 224;
  }
  out[offset]     = r;
  out[offset + 1] = g;
  out[offset + 2] = b;
  out[offset + 3] = a;
}

export function createFloodLayer(scene, geoMeta, options = {}) {
  const { width_m, height_m, z_min } = geoMeta;
  const {
    urlSuffix       = 'frames',
    meshName        = 'Flood Depth',
    yOffset         = 0.5,
    valueScale      = 1.0,   // multiply raw frame values before depthToRGBA bucketing
    terrainGeometry = null,  // if given, drape on terrain's actual displaced surface
                              // instead of a flat plane near z_min
  } = options;

  let geo, mat;
  if (terrainGeometry) {
    // Same displaced, UV-mapped geometry as terrain.js's solid mesh — the
    // colored depth texture paints directly onto the real elevation instead
    // of floating as a flat sheet near the AOI's lowest point.
    geo = terrainGeometry;
    mat = new THREE.MeshBasicMaterial({
      transparent: true,
      depthWrite: false,
      side: THREE.DoubleSide,
      polygonOffset: true,
      polygonOffsetFactor: -6,
      polygonOffsetUnits: -6,
    });
  } else {
    geo = new THREE.PlaneGeometry(width_m, height_m);
    geo.rotateX(-Math.PI / 2);
    mat = new THREE.MeshBasicMaterial({
      transparent: true,
      depthWrite: false,
      side: THREE.DoubleSide,
    });
  }

  // DataTexture updated per frame
  const texData = new Uint8Array(ROWS * COLS * 4);
  const texture  = new THREE.DataTexture(texData, COLS, ROWS, THREE.RGBAFormat);
  texture.flipY  = true;     // match PNG overlay convention (row 0 = north)
  texture.needsUpdate = true;
  mat.map = texture;

  const mesh = new THREE.Mesh(geo, mat);
  mesh.name    = meshName;
  mesh.visible = false;

  // Flat fallback: position at terrain surface + offset (avoids z-fighting).
  // Draped mode: geometry already encodes elevation; polygonOffset above
  // handles z-fighting against the co-planar terrain solid mesh instead.
  mesh.position.y = terrainGeometry ? 0 : (z_min - z_min) * VERT_EXAG + yOffset;
  scene.add(mesh);

  // Internal state
  let frames    = null;    // Float32Array, length n_frames * ROWS * COLS
  let timesMin  = null;    // Float32Array, length n_frames
  let nFrames   = 0;

  function _clearTexture() {
    texData.fill(0);
    texture.needsUpdate = true;
  }

  async function loadScenario(id) {
    _clearTexture();
    frames   = null;
    timesMin = null;
    nFrames  = 0;

    const url = `/data/simulation_${id}_${urlSuffix}.bin`;
    const resp = await fetch(url);
    if (!resp.ok) throw new Error(`Flood frames not found: ${url}`);
    const buf  = await resp.arrayBuffer();
    const view = new DataView(buf);

    // Validate magic
    const magic = String.fromCharCode(
      view.getUint8(0), view.getUint8(1), view.getUint8(2), view.getUint8(3)
    );
    if (magic !== 'SIML') throw new Error(`Invalid SIML magic in ${url}`);

    nFrames          = view.getUint32(4,  true);
    const rows       = view.getUint32(8,  true);
    const cols       = view.getUint32(12, true);
    const headerSize = 16 + nFrames * 4;

    timesMin = new Float32Array(buf, 16, nFrames);
    frames   = new Float32Array(buf, headerSize, nFrames * rows * cols);

    return { nFrames, timesMin };
  }

  function setFrame(frameIdx) {
    if (!frames || frameIdx < 0 || frameIdx >= nFrames) {
      _clearTexture();
      return;
    }
    const base = frameIdx * ROWS * COLS;
    for (let i = 0; i < ROWS * COLS; i++) {
      const v = frames[base + i] * valueScale;
      // floor at 0.1mm — drops exact-zero/negative numerical noise while
      // keeping real thin-film depth visible (depthToRGBA's low-end buckets
      // fade it in gradually instead of a hard pop at a single threshold)
      const d = v >= 0.0001 ? v : 0;
      depthToRGBA(d, texData, i * 4);
    }
    texture.needsUpdate = true;
  }

  function reset() {
    _clearTexture();
  }

  function dispose() {
    scene.remove(mesh);
    if (!terrainGeometry) geo.dispose();   // shared geometry is owned by terrain.js
    mat.dispose();
    texture.dispose();
  }

  function getTimesMin() { return timesMin; }
  function getFrameCount() { return nFrames; }

  return { mesh, loadScenario, setFrame, reset, dispose, getTimesMin, getFrameCount };
}
