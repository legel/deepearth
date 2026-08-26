import * as THREE from 'three';

/**
 * createDropletFlow(url) → { mesh, tick(dt), setSpeed(x), setWaterAmount(0-1), setRainSpread(s) }
 *
 * Renders the droplet-based rainfall/runoff test (lidar/droplet_flow_test.py) — the "dumb
 * version" 3D water-flow prototype from simulation/PLAN_3D_DROPLET_PROTOTYPE.md. Real 3D flow
 * (gravity projected onto each mesh triangle's own plane, walking triangle-to-triangle) on a
 * small, fine-resolution test area — deliberately NOT the grid-based 2.5D shallow-water solver
 * used for the full-AOI Hurricane Ian simulation.
 *
 * Two visual layers, both from the same data:
 *   - Faint static trail lines — each droplet's complete path, all at once.
 *   - Animated points — droplets actually falling and flowing over time, looping.
 * Colored by how each droplet ends: blue = settled in a local low point (a real puddle — only
 * possible on the ground surface in the v2 DEM/roof-fusion architecture), green = ran off the
 * edge of the test-area mesh (real runoff), orange = still slowly moving when the step budget
 * ran out.
 *
 * Live controls (all mutate playback only — no data re-fetch):
 *   setSpeed(x)        — playback speed multiplier (1 = the baseline STEPS_PER_SECOND)
 *   setWaterAmount(f)  — fraction (0-1) of the loaded droplets actively shown/animated
 *   setRainSpread(s)   — seconds over which droplet start times are staggered (0 = every
 *                        droplet starts moving together, i.e. a single "burst"; higher values
 *                        spread starts out into more of a continuous, steady rain)
 *
 * Binary format (see droplet_flow_test.py):
 *   b'DROP' + uint32 n_droplets, then per droplet:
 *     uint32 n_points, float32[n_points*3] scene-space xyz (VERT_EXAG/z_min/origin baked in,
 *     same convention as terrain.js), uint8 settle_reason_code (0=local_min, 1=left_mesh,
 *     2=max_steps)
 */
const REASON_COLOR = {
  0: new THREE.Color(0x3399ff),   // local_min — settled/puddle
  1: new THREE.Color(0x33cc66),   // left_mesh — drained off the test area
  2: new THREE.Color(0xffaa33),   // max_steps — still moving
};

const BASE_STEPS_PER_SECOND = 30;   // baseline playback speed at speed=1

// Droplet paths sit essentially AT the same real-world surface (roof/ground) coordinates as
// the LiDAR points forming that surface — when the "dense point cloud" layer is also on, the
// two literally occupy near-identical depth values, so standard depth-tested rendering
// z-fights and the dense, opaque point cloud (many more points, similar/larger size) visually
// buries the thin droplet trail (2026-07-21: "current roof top layer covers the droplet").
// Fix: droplet materials disable depth testing/writing and get a high renderOrder so they
// always draw after — visually on top of — the point cloud, regardless of true depth. That's
// the right trade here (there's no real scenario where something should legitimately occlude
// a droplet trail sitting on the exact surface it's tracing).
const DROPLET_RENDER_ORDER = 50;

// Real, solver-driven horizontal creep after landing is genuinely small on this low-relief
// terrain (see meshShallowWater.js's history for the measured numbers) — a single 2.5px point
// crawling a few scene units over ~16s reads as motionless at normal playback speed. Rather
// than rescale positions (which broke correctness — see the file-level comment above), each
// active droplet gets a short trailing streak built from its own last COMET_STEPS real
// positions: a moving droplet reads as a visible line-segment "swoosh" pointing back along its
// actual path, a settled/frozen one just shows a zero-length streak (a dot). This can never
// leave the real mesh-constrained trajectory since it only ever draws positions the solver
// itself produced.
const COMET_STEPS = 12;   // ~0.4s of real path history at BASE_STEPS_PER_SECOND

// DO NOT scale a path's post-landing (x,z) displacement to improve motion legibility. That was
// tried and reverted: it makes droplets visibly travel over trees and rooftops, which water
// cannot do. It is a correctness bug, not an acceptable visual liberty —
// unlike vertical exaggeration (which only lifts a point straight up off the surface it's
// already correctly tracking), scaling horizontal displacement moves the point to an (x,z) the
// water never actually reached — off its real mesh triangle, potentially through/over
// neighboring roofs or the tree canopy the physics mesh doesn't even model (see
// meshShallowWater.js's COMET_STEPS comment for the mesh/point-cloud connection question this
// also surfaced). The real per-step positions from mesh_shallow_water.py are constrained to
// the actual ground+roof mesh topology and must be rendered exactly as solved — no post-hoc
// horizontal rescaling, ever.
export async function createDropletFlow(url = '/data/droplet_paths.bin') {
  const buf = await fetch(url).then(r => r.arrayBuffer());
  const dv = new DataView(buf);

  const magic = String.fromCharCode(dv.getUint8(0), dv.getUint8(1), dv.getUint8(2), dv.getUint8(3));
  if (magic !== 'DROP') throw new Error(`${url}: bad magic "${magic}"`);
  const nDroplets = dv.getUint32(4, true);

  const group = new THREE.Group();
  group.name = 'Droplet Flow Paths';

  const paths = [];       // Float32Array(n_points*3) per droplet, kept for animation
  const reasons = [];     // settle_reason_code per droplet
  const trailLines = [];  // THREE.Line per droplet, same order as paths — for water-amount cutoff
  let maxSteps = 1;

  let offset = 8;
  const counts = { 0: 0, 1: 0, 2: 0 };
  for (let i = 0; i < nDroplets; i++) {
    const nPoints = dv.getUint32(offset, true);
    offset += 4;
    if (nPoints < 2) {
      offset += 1;
      continue;
    }
    // buf.slice() copies out a fresh, zero-offset ArrayBuffer — a direct `new Float32Array(buf,
    // offset, ...)` view requires `offset` to be a multiple of 4, which isn't guaranteed here:
    // each droplet's record ends with a 1-byte settle_reason_code, so byte offsets drift off
    // the 4-byte boundary after the first droplet. slice() sidesteps the alignment requirement
    // entirely (and we wanted a copy here anyway, not a view aliasing the original buffer).
    const byteLen = nPoints * 3 * 4;
    const positions = new Float32Array(buf.slice(offset, offset + byteLen));
    offset += byteLen;
    const reason = dv.getUint8(offset);
    offset += 1;
    counts[reason] = (counts[reason] ?? 0) + 1;

    paths.push(positions);
    reasons.push(reason);
    maxSteps = Math.max(maxSteps, nPoints);

    // Faint static trail — full path, drawn once.
    const geo = new THREE.BufferGeometry();
    geo.setAttribute('position', new THREE.BufferAttribute(positions, 3));
    const mat = new THREE.LineBasicMaterial({
      color: REASON_COLOR[reason] ?? 0xffffff,
      transparent: true,
      opacity: 0.22,
      depthTest: false, depthWrite: false,   // see DROPLET_RENDER_ORDER below
    });
    const line = new THREE.Line(geo, mat);
    line.renderOrder = DROPLET_RENDER_ORDER;
    trailLines.push(line);
    group.add(line);
  }

  console.log(`Droplet flow: ${paths.length} paths (settled=${counts[0] || 0} `
    + `runoff=${counts[1] || 0} still-moving=${counts[2] || 0})`);

  // Animated "current position" points — one per droplet, advancing along its own path each
  // frame. Droplets that have already settled/left the mesh just sit at their final position
  // (visibly puddled / gone) rather than disappearing.
  const n = paths.length;
  const animPositions = new Float32Array(n * 3);
  const animColors = new Float32Array(n * 3);
  const startOffset = new Float32Array(n);   // seconds — set by setRainSpread()
  for (let i = 0; i < n; i++) {
    animPositions[i * 3]     = paths[i][0];
    animPositions[i * 3 + 1] = paths[i][1];
    animPositions[i * 3 + 2] = paths[i][2];
    const c = REASON_COLOR[reasons[i]] ?? REASON_COLOR[2];
    animColors[i * 3] = c.r; animColors[i * 3 + 1] = c.g; animColors[i * 3 + 2] = c.b;
  }
  const animGeo = new THREE.BufferGeometry();
  animGeo.setAttribute('position', new THREE.BufferAttribute(animPositions, 3));
  animGeo.setAttribute('color', new THREE.BufferAttribute(animColors, 3));
  const animMat = new THREE.PointsMaterial({
    size: 2.5, vertexColors: true, sizeAttenuation: true,
    transparent: true, depthTest: false, depthWrite: false,   // see DROPLET_RENDER_ORDER below
  });
  const animPoints = new THREE.Points(animGeo, animMat);
  animPoints.name = 'Droplet Flow (animated)';
  animPoints.renderOrder = DROPLET_RENDER_ORDER;
  group.add(animPoints);

  // Comet-tail streak per droplet — see COMET_STEPS comment above.
  const cometPositions = new Float32Array(n * 2 * 3);
  const cometColors = new Float32Array(n * 2 * 3);
  for (let i = 0; i < n; i++) {
    for (let k = 0; k < 2; k++) {
      cometPositions[(i * 2 + k) * 3]     = paths[i][0];
      cometPositions[(i * 2 + k) * 3 + 1] = paths[i][1];
      cometPositions[(i * 2 + k) * 3 + 2] = paths[i][2];
      const c = REASON_COLOR[reasons[i]] ?? REASON_COLOR[2];
      cometColors[(i * 2 + k) * 3] = c.r; cometColors[(i * 2 + k) * 3 + 1] = c.g; cometColors[(i * 2 + k) * 3 + 2] = c.b;
    }
  }
  const cometGeo = new THREE.BufferGeometry();
  cometGeo.setAttribute('position', new THREE.BufferAttribute(cometPositions, 3));
  cometGeo.setAttribute('color', new THREE.BufferAttribute(cometColors, 3));
  const cometMat = new THREE.LineBasicMaterial({
    vertexColors: true, transparent: true, opacity: 0.75,
    depthTest: false, depthWrite: false,   // see DROPLET_RENDER_ORDER above
  });
  const cometLines = new THREE.LineSegments(cometGeo, cometMat);
  cometLines.name = 'Droplet Flow (comet trail)';
  cometLines.renderOrder = DROPLET_RENDER_ORDER;
  group.add(cometLines);

  // ── Live playback controls ────────────────────────────────────────────────
  let speed = 1.0;
  let waterAmount = 1.0;
  let rainSpreadSeconds = 0;
  let activeCount = n;

  function setSpeed(x) { speed = x; }

  function setWaterAmount(fraction) {
    waterAmount = Math.max(0, Math.min(1, fraction));
    activeCount = Math.max(0, Math.round(n * waterAmount));
    animGeo.setDrawRange(0, activeCount);
    cometGeo.setDrawRange(0, activeCount * 2);
    for (let i = 0; i < n; i++) trailLines[i].visible = i < activeCount;
  }

  function setRainSpread(seconds) {
    rainSpreadSeconds = Math.max(0, seconds);
    for (let i = 0; i < n; i++) startOffset[i] = (i / n) * rainSpreadSeconds;
  }

  let elapsed = 0;
  function tick(dt) {
    if (!group.visible) return;
    elapsed += dt * speed;
    const cycleSeconds = rainSpreadSeconds + maxSteps / BASE_STEPS_PER_SECOND;
    const cycleT = elapsed % cycleSeconds;
    const posAttr = animGeo.attributes.position;
    const cometAttr = cometGeo.attributes.position;
    for (let i = 0; i < activeCount; i++) {
      const path = paths[i];
      const nPts = path.length / 3;
      const localT = cycleT - startOffset[i];
      const step = localT < 0 ? 0 : Math.min(Math.floor(localT * BASE_STEPS_PER_SECOND), nPts - 1);
      posAttr.array[i * 3]     = path[step * 3];
      posAttr.array[i * 3 + 1] = path[step * 3 + 1];
      posAttr.array[i * 3 + 2] = path[step * 3 + 2];

      const prevStep = Math.max(0, step - COMET_STEPS);
      cometAttr.array[i * 6]     = path[prevStep * 3];
      cometAttr.array[i * 6 + 1] = path[prevStep * 3 + 1];
      cometAttr.array[i * 6 + 2] = path[prevStep * 3 + 2];
      cometAttr.array[i * 6 + 3] = path[step * 3];
      cometAttr.array[i * 6 + 4] = path[step * 3 + 1];
      cometAttr.array[i * 6 + 5] = path[step * 3 + 2];
    }
    posAttr.needsUpdate = true;
    cometAttr.needsUpdate = true;
  }

  // Frees GPU buffers for all droplet geometries/materials and detaches the group from its
  // parent. Needed for the 2026-07-21 rain-intensity re-simulation feature: applying a new
  // intensity swaps in a freshly-fetched createDropletFlow() instance, and without this the old
  // one's typed arrays + WebGL buffers (up to 2,500 paths x up to ~540 points each) would leak
  // on every apply.
  function dispose() {
    group.parent?.remove(group);
    for (const line of trailLines) { line.geometry.dispose(); line.material.dispose(); }
    animGeo.dispose(); animMat.dispose();
    cometGeo.dispose(); cometMat.dispose();
  }

  return { mesh: group, tick, setSpeed, setWaterAmount, setRainSpread, dispose };
}
