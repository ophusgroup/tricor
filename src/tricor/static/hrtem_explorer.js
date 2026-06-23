// HRTEMExplorer — HRTEM frame (thickness/defocus) <-> local block g2 / g3.
// Hand-authored ESM for anywidget (no bundler), matching ptycho_explorer.js.

const RDBU_R = [
  [0.0, [5, 48, 97]], [0.1, [33, 102, 172]], [0.25, [67, 147, 195]],
  [0.4, [146, 197, 222]], [0.48, [209, 229, 240]], [0.5, [247, 247, 247]],
  [0.52, [253, 219, 199]], [0.6, [244, 165, 130]], [0.75, [214, 96, 77]],
  [0.9, [178, 24, 43]], [1.0, [103, 0, 31]],
];

function lerp(a, b, t) { return a + (b - a) * t; }
function ramp(stops, t) {
  t = Math.max(0, Math.min(1, t));
  for (let i = 1; i < stops.length; i++) {
    if (t <= stops[i][0]) {
      const [t0, c0] = stops[i - 1], [t1, c1] = stops[i];
      const f = (t - t0) / (t1 - t0 || 1);
      return `rgb(${Math.round(lerp(c0[0], c1[0], f))},${Math.round(lerp(c0[1], c1[1], f))},${Math.round(lerp(c0[2], c1[2], f))})`;
    }
  }
  const c = stops[stops.length - 1][1];
  return `rgb(${c[0]},${c[1]},${c[2]})`;
}
function grey(t) { t = Math.max(0, Math.min(1, t)); const v = Math.round(255 * t); return [v, v, v]; }
function percentile(sorted, p) {
  if (!sorted.length) return 0;
  const i = Math.max(0, Math.min(sorted.length - 1, Math.round(p * (sorted.length - 1))));
  return sorted[i];
}
function el(tag, cls, parent) {
  const e = document.createElement(tag);
  if (cls) e.className = cls;
  if (parent) parent.appendChild(e);
  return e;
}

// Fixed contrast range, symmetric about the mean (HRTEM frame, mean ~1).
function symClim(vals, p) {
  const f = vals.filter((v) => Number.isFinite(v)).slice().sort((a, b) => a - b);
  if (!f.length) return [0, 1];
  const m = f.reduce((a, b) => a + b, 0) / f.length;
  const lo = percentile(f, p), hi = percentile(f, 1 - p);
  const h = Math.max(Math.abs(m - lo), Math.abs(hi - m)) * 1.05 || 1;
  return [m - h, m + h];
}
function pctClim(vals, loP, hiP) {
  const f = vals.filter((v) => Number.isFinite(v)).slice().sort((a, b) => a - b);
  if (!f.length) return [0, 1];
  return [percentile(f, loP), percentile(f, hiP)];
}

// Greyscale image (nx,ny) with a FIXED clim (no per-frame rescale).
function drawGrey(canvas, vals, nx, ny, label, clim) {
  const ctx = canvas.getContext("2d");
  const W = canvas.width, H = canvas.height;
  ctx.clearRect(0, 0, W, H);
  if (!vals.length || nx * ny !== vals.length) return;
  let vmin, vmax;
  if (clim) { [vmin, vmax] = clim; }
  else { const f = vals.filter((v) => Number.isFinite(v)).slice().sort((a, b) => a - b); vmin = percentile(f, 0.01); vmax = percentile(f, 0.99); }
  const span = vmax - vmin || 1;
  const img = ctx.createImageData(nx, ny);
  for (let i = 0; i < nx; i++) {
    for (let j = 0; j < ny; j++) {
      const [r, g, b] = grey((vals[i * ny + j] - vmin) / span);
      const k = (j * nx + i) * 4;
      img.data[k] = r; img.data[k + 1] = g; img.data[k + 2] = b; img.data[k + 3] = 255;
    }
  }
  const off = document.createElement("canvas");
  off.width = nx; off.height = ny;
  off.getContext("2d").putImageData(img, 0, 0);
  ctx.imageSmoothingEnabled = true;
  ctx.drawImage(off, 0, 0, W, H);
  if (label) {
    ctx.fillStyle = "#fff"; ctx.font = "10px sans-serif"; ctx.textAlign = "left";
    ctx.fillText(label, 5, 13);
  }
}

// HRTEM frame (fixed clim) + draggable circular window.
function drawFrame(canvas, model, drag, clim) {
  const vals = model.get("slice_values") || [];
  const [nx, ny] = model.get("slice_shape") || [0, 0];
  drawGrey(canvas, vals, nx, ny, null, clim);
  if (!vals.length || nx * ny !== vals.length) return;
  const ctx = canvas.getContext("2d");
  const W = canvas.width, H = canvas.height;
  const [Lx, Ly] = model.get("extent") || [1, 1];
  const cx = drag.cx != null ? drag.cx : model.get("window_cx");
  const cy = drag.cy != null ? drag.cy : model.get("window_cy");
  const side = model.get("window_side");
  const cxp = (cx / Lx) * W, cyp = (cy / Ly) * H;
  const rx = (side / 2 / Lx) * W, ry = (side / 2 / Ly) * H;
  ctx.strokeStyle = "#e02424"; ctx.lineWidth = 2; ctx.fillStyle = "rgba(224,36,36,0.08)";
  for (const ox of [-W, 0, W]) {
    for (const oy of [-H, 0, H]) {
      ctx.beginPath();
      ctx.ellipse(cxp + ox, cyp + oy, rx, ry, 0, 0, 2 * Math.PI);
      ctx.fill(); ctx.stroke();
    }
  }
}

// Histogram of the current frame with two draggable contrast handles.
function drawHist(canvas, vals, clim, axis) {
  const ctx = canvas.getContext("2d");
  const W = canvas.width, H = canvas.height;
  ctx.clearRect(0, 0, W, H);
  if (!vals.length || !axis) return;
  const [lo, hi] = axis, span = hi - lo || 1, nb = 72;
  const counts = new Array(nb).fill(0);
  for (let i = 0; i < vals.length; i++) {
    const v = vals[i];
    if (!Number.isFinite(v)) continue;
    let b = Math.floor(((v - lo) / span) * nb);
    counts[Math.max(0, Math.min(nb - 1, b))]++;
  }
  const cmax = Math.max(...counts) || 1, ph = H - 4, bw = W / nb;
  ctx.fillStyle = "#5b8fb0";
  for (let i = 0; i < nb; i++) {
    const h = (counts[i] / cmax) * ph;
    ctx.fillRect(i * bw, 4 + ph - h, Math.max(1, bw - 0.4), h);
  }
  const vToX = (v) => ((v - lo) / span) * W;
  ctx.strokeStyle = "#e02424"; ctx.lineWidth = 2;
  for (const c of clim) {
    const x = vToX(c);
    ctx.beginPath(); ctx.moveTo(x, 0); ctx.lineTo(x, H); ctx.stroke();
  }
  ctx.fillStyle = "#444"; ctx.font = "9px sans-serif";
  ctx.textAlign = "left"; ctx.fillText(`contrast ${clim[0].toFixed(2)}`, 3, H - 3);
  ctx.textAlign = "right"; ctx.fillText(`${clim[1].toFixed(2)}`, W - 3, H - 3);
}

// ---- right panels: g3 heatmap + g2 line (shared r axis) -------------------
const MARGIN = { left: 42, right: 56, top: 18, bottom: 26 };
function rToX(r, rMax, W) { return MARGIN.left + (r / rMax) * (W - MARGIN.left - MARGIN.right); }

function drawG3(canvas, model) {
  const vals = model.get("g3_slice_values") || [];
  const [nphi, nr] = model.get("g3_slice_shape") || [0, 0];
  const rMax = model.get("r_max") || 10;
  const ctx = canvas.getContext("2d");
  const W = canvas.width, H = canvas.height;
  ctx.clearRect(0, 0, W, H);
  if (!vals.length || nphi * nr !== vals.length) return;
  const pw = W - MARGIN.left - MARGIN.right, ph = H - MARGIN.top - MARGIN.bottom;
  let vmax = model.get("g3_vmax");
  if (!(vmax > 0)) {
    const s = vals.filter((v) => Number.isFinite(v) && v > 0).slice().sort((a, b) => a - b);
    vmax = Math.max(1.5, percentile(s, 0.99));
  }
  const cw = pw / nr, chh = ph / nphi;
  for (let p = 0; p < nphi; p++) {
    for (let i = 0; i < nr; i++) {
      const v = vals[p * nr + i];
      const t = 0.5 + 0.5 * (v - 1) / (vmax - 1);
      ctx.fillStyle = ramp(RDBU_R, t);
      ctx.fillRect(MARGIN.left + i * cw, MARGIN.top + (nphi - 1 - p) * chh, cw + 0.6, chh + 0.6);
    }
  }
  ctx.strokeStyle = "#888"; ctx.lineWidth = 1; ctx.strokeRect(MARGIN.left, MARGIN.top, pw, ph);
  ctx.fillStyle = "#333"; ctx.font = "10px sans-serif"; ctx.textAlign = "right";
  [0, 45, 90, 135, 180].forEach((a) => {
    const y = MARGIN.top + (1 - a / 180) * ph;
    ctx.fillText(`${a}`, MARGIN.left - 4, y + 3);
  });
  ctx.save();
  ctx.translate(11, MARGIN.top + ph / 2); ctx.rotate(-Math.PI / 2);
  ctx.textAlign = "center"; ctx.fillText("angle (deg)", 0, 0); ctx.restore();
  ctx.textAlign = "left"; ctx.fillText("weighted g3 slice (r01 ~ NN band)", MARGIN.left, 12);
  const cbX = W - MARGIN.right + 12, cbW = 12, cbTop = MARGIN.top, cbH = ph;
  const grad = ctx.createLinearGradient(0, cbTop, 0, cbTop + cbH);
  for (let s = 0; s <= 1.0001; s += 0.05) grad.addColorStop(Math.min(s, 1), ramp(RDBU_R, 1 - s));
  ctx.fillStyle = grad; ctx.fillRect(cbX, cbTop, cbW, cbH);
  ctx.strokeStyle = "#888"; ctx.strokeRect(cbX, cbTop, cbW, cbH);
  ctx.fillStyle = "#333"; ctx.font = "9px sans-serif"; ctx.textAlign = "left";
  const vmin = Math.max(0, 2 - vmax);
  ctx.fillText(vmax.toFixed(1), cbX + cbW + 3, cbTop + 7);
  ctx.fillText("1", cbX + cbW + 3, cbTop + cbH / 2 + 3);
  ctx.fillText(vmin.toFixed(1), cbX + cbW + 3, cbTop + cbH - 1);
  ctx.fillText("g3", cbX - 2, cbTop - 6);
}

function drawG2(canvas, model) {
  const g2 = model.get("g2") || [];
  const rArr = model.get("r") || [];
  const rMax = model.get("r_max") || 10;
  const band = model.get("nn_band") || [0, 0];
  const ctx = canvas.getContext("2d");
  const W = canvas.width, H = canvas.height;
  ctx.clearRect(0, 0, W, H);
  if (!g2.length) return;
  const pw = W - MARGIN.left - MARGIN.right, ph = H - MARGIN.top - MARGIN.bottom;
  const ymax = Math.max(2, Math.ceil(Math.max(...g2.filter(Number.isFinite)) * 1.1));
  const yToPx = (y) => MARGIN.top + (1 - y / ymax) * ph;
  ctx.fillStyle = "rgba(224,36,36,0.10)";
  ctx.fillRect(rToX(band[0], rMax, W), MARGIN.top, rToX(band[1], rMax, W) - rToX(band[0], rMax, W), ph);
  ctx.strokeStyle = "#bbb"; ctx.setLineDash([4, 3]); ctx.beginPath();
  ctx.moveTo(MARGIN.left, yToPx(1)); ctx.lineTo(W - MARGIN.right, yToPx(1)); ctx.stroke();
  ctx.setLineDash([]);
  ctx.strokeStyle = "#1f6f43"; ctx.lineWidth = 1.5; ctx.beginPath();
  for (let i = 0; i < g2.length; i++) {
    const x = rToX(rArr[i] || (i * rMax) / g2.length, rMax, W);
    const y = yToPx(Number.isFinite(g2[i]) ? g2[i] : 0);
    i === 0 ? ctx.moveTo(x, y) : ctx.lineTo(x, y);
  }
  ctx.stroke();
  ctx.strokeStyle = "#888"; ctx.lineWidth = 1; ctx.strokeRect(MARGIN.left, MARGIN.top, pw, ph);
  ctx.fillStyle = "#333"; ctx.font = "10px sans-serif"; ctx.textAlign = "center";
  for (let r = 0; r <= rMax; r += 2) ctx.fillText(`${r}`, rToX(r, rMax, W), H - 8);
  ctx.fillText("radius r (Å)", MARGIN.left + pw / 2, H - 0.5);
  ctx.textAlign = "right";
  ctx.fillText(`${ymax}`, MARGIN.left - 4, MARGIN.top + 8);
  ctx.fillText("1", MARGIN.left - 4, yToPx(1) + 3);
  ctx.textAlign = "left"; ctx.fillText("weighted g2", MARGIN.left, 12);
}

// ---- render --------------------------------------------------------------
function render({ model, el: root }) {
  root.classList.add("hrtem-explorer");
  const [Lx, Ly] = model.get("extent") || [1, 1];

  // Fixed contrast ranges, computed once from the first frame / FFT and then
  // held constant as the sliders move (the frame range is histogram-editable).
  let frameClim = null, fftClim = null, histAxis = null;
  function ensureClims() {
    if (frameClim == null) {
      const fc = model.get("frame_clim");
      if (fc && fc.length === 2) frameClim = fc.slice();
      else { const sv = model.get("slice_values") || []; if (sv.length) frameClim = symClim(sv, 0.005); }
      if (frameClim) { const w = frameClim[1] - frameClim[0]; histAxis = [frameClim[0] - 0.3 * w, frameClim[1] + 0.3 * w]; }
    }
    if (fftClim == null) {
      const fc = model.get("fft_clim");
      if (fc && fc.length === 2) fftClim = fc.slice();
      else { const fv = model.get("fft_values") || []; if (fv.length) fftClim = pctClim(fv, 0.02, 0.997); }
    }
  }

  function mkRange(parent, label, min, max, step, val) {
    el("span", "px-label", parent).textContent = label;
    const s = el("input", null, parent);
    s.type = "range"; s.min = min; s.max = max; s.step = step; s.value = val;
    const v = el("span", "px-val", parent);
    return [s, v];
  }

  const wrap = el("div", "px-wrap", root);
  const left = el("div", "px-col", wrap);
  const frameCanvas = el("canvas", "px-frame", left);
  frameCanvas.width = 340; frameCanvas.height = Math.round(340 * Ly / Lx);
  const histCanvas = el("canvas", "px-hist", left);
  histCanvas.width = 340; histCanvas.height = 50;
  const [thickSlider, thickVal] = mkRange(el("div", "px-row", left), "thick",
    0, (model.get("thicknesses") || [0]).length - 1, 1, model.get("thickness_index"));
  const [dfSlider, dfVal] = mkRange(el("div", "px-row", left), "defocus", -120, 120, 5, model.get("defocus_offset"));
  const [rotSlider, rotVal] = mkRange(el("div", "px-row", left), "rotate", -180, 180, 5, model.get("window_angle"));

  const mid = el("div", "px-col", wrap);
  const inputCanvas = el("canvas", "px-input", mid); inputCanvas.width = 168; inputCanvas.height = 168;
  const fftCanvas = el("canvas", "px-input", mid); fftCanvas.width = 168; fftCanvas.height = 168;

  const right = el("div", "px-col", wrap);
  const g3Canvas = el("canvas", "px-g3", right); g3Canvas.width = 360; g3Canvas.height = 210;
  const g2Canvas = el("canvas", "px-g2", right); g2Canvas.width = 360; g2Canvas.height = 150;
  el("span", "px-label", el("div", "px-row", right)).textContent = "g3 max";
  const vmaxInput = el("input", null, right.lastChild);
  vmaxInput.type = "number"; vmaxInput.min = 0; vmaxInput.step = 0.5; vmaxInput.placeholder = "auto";
  vmaxInput.style.width = "60px";

  const status = el("div", "px-status", root);
  const drag = { cx: null, cy: null, active: false };

  const sgn = (x) => (x >= 0 ? "+" : "");
  function syncControls() {
    // Labels + status only — never reset slider thumbs (Python does not move
    // these controls, so resetting them is what made the defocus thumb jump).
    const thicks = model.get("thicknesses") || [0];
    thickVal.textContent = `${(thicks[model.get("thickness_index")] || 0).toFixed(0)} Å`;
    dfVal.textContent = `Δf ${sgn(model.get("defocus_offset"))}${model.get("defocus_offset").toFixed(0)} Å`;
    rotVal.textContent = `${(model.get("window_angle") || 0).toFixed(0)}°`;
    status.textContent = model.get("status") || "";
  }
  function redrawFrame() { ensureClims(); drawFrame(frameCanvas, model, drag, frameClim); drawHist(histCanvas, model.get("slice_values") || [], frameClim, histAxis); }
  function redrawInputs() {
    ensureClims();
    drawGrey(inputCanvas, model.get("input_values") || [], ...(model.get("input_shape") || [0, 0]), "input (net)");
    drawGrey(fftCanvas, model.get("fft_values") || [], ...(model.get("fft_shape") || [0, 0]), "diffractogram |FFT|", fftClim);
  }
  function redrawAll() { redrawFrame(); redrawInputs(); drawG3(g3Canvas, model); drawG2(g2Canvas, model); syncControls(); }

  // --- window drag on the frame ---
  function pointerToAngstrom(ev) {
    const rect = frameCanvas.getBoundingClientRect();
    return [((ev.clientX - rect.left) / rect.width) * Lx, ((ev.clientY - rect.top) / rect.height) * Ly];
  }
  const clamp = (c, L) => ((c % L) + L) % L;
  frameCanvas.addEventListener("pointerdown", (ev) => {
    drag.active = true;
    const [ax, ay] = pointerToAngstrom(ev);
    drag.cx = clamp(ax, Lx); drag.cy = clamp(ay, Ly);
    frameCanvas.setPointerCapture(ev.pointerId);
    drawFrame(frameCanvas, model, drag, frameClim);
  });
  frameCanvas.addEventListener("pointermove", (ev) => {
    if (!drag.active) return;
    const [ax, ay] = pointerToAngstrom(ev);
    drag.cx = clamp(ax, Lx); drag.cy = clamp(ay, Ly);
    drawFrame(frameCanvas, model, drag, frameClim);
  });
  function endDrag() {
    if (!drag.active) return;
    drag.active = false;
    model.set("window_cx", drag.cx); model.set("window_cy", drag.cy);
    model.save_changes();
    drag.cx = null; drag.cy = null;
  }
  frameCanvas.addEventListener("pointerup", endDrag);
  frameCanvas.addEventListener("pointercancel", endDrag);

  // --- contrast handles on the histogram ---
  let histDrag = null;
  function histX(ev) {
    const rect = histCanvas.getBoundingClientRect();
    return ((ev.clientX - rect.left) / rect.width) * histCanvas.width;
  }
  function setClimFromX(x) {
    if (!frameClim || !histAxis) return;
    const v = histAxis[0] + (x / histCanvas.width) * (histAxis[1] - histAxis[0]);
    const eps = 1e-4 * (histAxis[1] - histAxis[0]);
    if (histDrag === 0) frameClim[0] = Math.min(v, frameClim[1] - eps);
    else frameClim[1] = Math.max(v, frameClim[0] + eps);
    redrawFrame();
  }
  histCanvas.addEventListener("pointerdown", (ev) => {
    if (!frameClim || !histAxis) return;
    const x = histX(ev);
    const vToX = (val) => ((val - histAxis[0]) / (histAxis[1] - histAxis[0])) * histCanvas.width;
    histDrag = Math.abs(x - vToX(frameClim[0])) <= Math.abs(x - vToX(frameClim[1])) ? 0 : 1;
    histCanvas.setPointerCapture(ev.pointerId);
    setClimFromX(x);
  });
  histCanvas.addEventListener("pointermove", (ev) => { if (histDrag != null) setClimFromX(histX(ev)); });
  histCanvas.addEventListener("pointerup", () => { histDrag = null; });
  histCanvas.addEventListener("pointercancel", () => { histDrag = null; });

  // --- sliders (commit on release; labels live on input) ---
  thickSlider.addEventListener("input", () => {
    const thicks = model.get("thicknesses") || [0];
    thickVal.textContent = `${(thicks[parseInt(thickSlider.value, 10)] || 0).toFixed(0)} Å`;
  });
  thickSlider.addEventListener("change", () => { model.set("thickness_index", parseInt(thickSlider.value, 10)); model.save_changes(); });
  dfSlider.addEventListener("input", () => { dfVal.textContent = `Δf ${sgn(+dfSlider.value)}${(+dfSlider.value).toFixed(0)} Å`; });
  dfSlider.addEventListener("change", () => { model.set("defocus_offset", parseFloat(dfSlider.value)); model.save_changes(); });
  rotSlider.addEventListener("input", () => { rotVal.textContent = `${(+rotSlider.value).toFixed(0)}°`; });
  rotSlider.addEventListener("change", () => { model.set("window_angle", parseFloat(rotSlider.value)); model.save_changes(); });
  vmaxInput.addEventListener("change", () => {
    const v = parseFloat(vmaxInput.value);
    model.set("g3_vmax", Number.isFinite(v) && v > 0 ? v : -1.0); model.save_changes();
  });

  // react to Python-side updates (frame/inputs change; clims stay fixed)
  model.on("change:slice_values", redrawFrame);
  model.on("change:input_values", redrawInputs);
  model.on("change:fft_values", redrawInputs);
  model.on("change:g3_slice_values", () => drawG3(g3Canvas, model));
  model.on("change:g2", () => drawG2(g2Canvas, model));
  model.on("change:g3_vmax", () => drawG3(g3Canvas, model));
  model.on("change:status", syncControls);
  model.on("change:thickness", syncControls);
  model.on("change:defocus", syncControls);

  redrawAll();
}

export default { render };
