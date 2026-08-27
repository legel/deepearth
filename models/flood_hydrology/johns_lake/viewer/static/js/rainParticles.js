/**
 * rainParticles.js — THREE.Points rain system
 *
 * Particles fall from above the scene at a rate proportional to rainfall
 * intensity (mm/hr). At 0 mm/hr all particles are hidden. At peak intensity
 * (e.g. 333 mm/hr for 100-yr flash storm) all N_PARTICLES are visible.
 *
 * Usage:
 *   const rain = createRainSystem(scene, geoMeta);
 *   // in animation loop:
 *   rain.update(rain_mm_hr, deltaTime_s);
 *   // on done:
 *   rain.dispose();
 */

import * as THREE from 'three';

const N_PARTICLES   = 2000;
const FALL_SPEED    = 180;     // scene units per second (scene Y scale = VERT_EXAG × metres)
const MAX_INTENSITY = 350;     // mm/hr at which all particles are active
// Scene geometry spans ~2000 scene units (width_m); at the default camera distance
// (~1000-1200 units away, see main.js `dist`), a sizeAttenuation point this small
// projected to ~1px and was effectively invisible. 6 reads clearly as rain droplets.
const PARTICLE_SIZE = 6.0;

export function createRainSystem(scene, geoMeta) {
  const { width_m, height_m, z_min, z_max } = geoMeta;
  const VERT_EXAG   = 8;
  const terrainTopY = (z_max - z_min) * VERT_EXAG;
  const rainTopY    = terrainTopY + 260;   // start well above terrain
  const rainBotY    = 0;                   // reset when they reach ground

  // Positions array — initialise randomly across the full XZ extent
  const positions = new Float32Array(N_PARTICLES * 3);
  for (let i = 0; i < N_PARTICLES; i++) {
    positions[i * 3]     = (Math.random() - 0.5) * width_m;
    positions[i * 3 + 1] = rainBotY + Math.random() * (rainTopY - rainBotY);
    positions[i * 3 + 2] = (Math.random() - 0.5) * height_m;
  }

  const geometry = new THREE.BufferGeometry();
  geometry.setAttribute('position', new THREE.BufferAttribute(positions, 3));

  // Draw range controls how many particles are actually rendered (0 = none)
  geometry.setDrawRange(0, 0);

  const material = new THREE.PointsMaterial({
    color: 0x88c8f0,
    size: PARTICLE_SIZE,
    transparent: true,
    opacity: 0.55,
    sizeAttenuation: true,
    depthWrite: false,
  });

  const points = new THREE.Points(geometry, material);
  points.name  = 'RainParticles';
  points.visible = false;
  scene.add(points);

  let currentIntensity = 0;

  function update(intensityMmHr, dt) {
    currentIntensity = intensityMmHr;
    const activeCount = Math.round(
      Math.min(intensityMmHr / MAX_INTENSITY, 1.0) * N_PARTICLES
    );
    points.visible = activeCount > 0;
    geometry.setDrawRange(0, activeCount);

    if (activeCount === 0) return;

    const pos = geometry.attributes.position;
    const fallDist = FALL_SPEED * dt;

    for (let i = 0; i < activeCount; i++) {
      pos.array[i * 3 + 1] -= fallDist;
      // Reset particle to top when it exits the bottom
      if (pos.array[i * 3 + 1] < rainBotY) {
        pos.array[i * 3]     = (Math.random() - 0.5) * width_m;
        pos.array[i * 3 + 1] = rainTopY;
        pos.array[i * 3 + 2] = (Math.random() - 0.5) * height_m;
      }
    }
    pos.needsUpdate = true;
  }

  function dispose() {
    scene.remove(points);
    geometry.dispose();
    material.dispose();
  }

  return { update, dispose, points };
}
