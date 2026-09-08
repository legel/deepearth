/**
 * Simulated vs observed discharge on a plain 2D canvas.
 *
 * No charting library on purpose: this is two line series, an axis pair and a playback cursor,
 * and a dependency would be larger than the code it replaced.
 *
 * The y-axis is LOGARITHMIC, and that is not decoration. Simulated discharge is integrated over
 * all four edges of the modelled box while the gauge measures one channel draining its own
 * catchment (46.6 km2 against 33.2 km2 at site3), so the two peaks differ by an order of
 * magnitude. On a linear axis the observed record collapses onto the baseline and the only thing
 * the chart can show -- that the timing agrees -- becomes invisible. Read shape and timing here,
 * never relative height.
 */
const PAD = { left: 52, right: 12, top: 12, bottom: 24 };
const Q_FLOOR = 10;  // cfs; below this the log axis is all noise

export function createChart(canvas, data) {
  const ctx = canvas.getContext('2d');
  const hasObs = Array.isArray(data.obs_cfs) && data.obs_cfs.length > 0;
  const tMax = Math.max(...data.time_h, hasObs ? Math.max(...data.obs_time_h) : 0);
  const peak = Math.max(Math.max(...data.sim_cfs), hasObs ? Math.max(...data.obs_cfs) : 0);
  const decades = Math.ceil(Math.log10(Math.max(peak, Q_FLOOR * 10) / Q_FLOOR));

  function draw(cursorHours) {
    const dpr = window.devicePixelRatio || 1;
    const w = canvas.clientWidth, h = canvas.clientHeight;
    if (canvas.width !== w * dpr) { canvas.width = w * dpr; canvas.height = h * dpr; }
    ctx.setTransform(dpr, 0, 0, dpr, 0, 0);
    ctx.clearRect(0, 0, w, h);

    const x = t => PAD.left + (t / tMax) * (w - PAD.left - PAD.right);
    const y = q => {
      const f = Math.log10(Math.max(q, Q_FLOOR) / Q_FLOOR) / decades;
      return h - PAD.bottom - f * (h - PAD.top - PAD.bottom);
    };

    ctx.strokeStyle = '#1f2a36';
    ctx.fillStyle = '#8fa1b3';
    ctx.font = '10px system-ui, sans-serif';
    ctx.lineWidth = 1;
    for (let d = 0; d <= decades; d++) {
      const q = Q_FLOOR * Math.pow(10, d);
      ctx.beginPath(); ctx.moveTo(PAD.left, y(q)); ctx.lineTo(w - PAD.right, y(q)); ctx.stroke();
      ctx.textAlign = 'right';
      ctx.fillText(q.toLocaleString(), PAD.left - 6, y(q) + 3);
    }
    ctx.textAlign = 'center';
    for (let t = 0; t <= tMax; t += 24) ctx.fillText(`${t}h`, x(t), h - 8);

    const series = (ts, qs, color, width) => {
      ctx.strokeStyle = color; ctx.lineWidth = width; ctx.beginPath();
      for (let i = 0; i < ts.length; i++) {
        const px = x(ts[i]), py = y(qs[i]);
        i ? ctx.lineTo(px, py) : ctx.moveTo(px, py);
      }
      ctx.stroke();
    };
    if (hasObs) series(data.obs_time_h, data.obs_cfs, '#f0a742', 1.6);
    series(data.time_h, data.sim_cfs, '#4ea1ff', 1.6);

    if (cursorHours != null) {
      ctx.strokeStyle = '#dfe7ef'; ctx.lineWidth = 1;
      ctx.beginPath(); ctx.moveTo(x(cursorHours), PAD.top);
      ctx.lineTo(x(cursorHours), h - PAD.bottom); ctx.stroke();
    }

    ctx.textAlign = 'left';
    ctx.fillStyle = '#4ea1ff'; ctx.fillText('simulated', PAD.left + 6, PAD.top + 10);
    if (hasObs) {
      ctx.fillStyle = '#f0a742';
      ctx.fillText('observed (USGS)', PAD.left + 6, PAD.top + 24);
    }
  }

  return { draw };
}
