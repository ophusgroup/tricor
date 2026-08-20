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

// HRTEM frame (fixed clim) + draggable circular window.  Wide cells are
// transposed so the long axis is horizontal.
function drawFrame(canvas, model, drag, clim) {
  const vals = model.get("slice_values") || [];
  const [nx, ny] = model.get("slice_shape") || [0, 0];
  const ctx = canvas.getContext("2d");
  const W = canvas.width, H = canvas.height;
  ctx.clearRect(0, 0, W, H);
  if (!vals.length || nx * ny !== vals.length) return;
  let vmin, vmax;
  if (clim) { [vmin, vmax] = clim; }
  else { const f = vals.filter((v) => Number.isFinite(v)).slice().sort((a, b) => a - b); vmin = percentile(f, 0.01); vmax = percentile(f, 0.99); }
  const span = vmax - vmin || 1;
  const transpose = model.get("transpose");
  const iw = transpose ? ny : nx, ih = transpose ? nx : ny;
  const img = ctx.createImageData(iw, ih);
  for (let i = 0; i < nx; i++) {
    for (let j = 0; j < ny; j++) {
      const [r, g, b] = grey((vals[i * ny + j] - vmin) / span);
      const k = (transpose ? (i * ny + j) : (j * nx + i)) * 4;
      img.data[k] = r; img.data[k + 1] = g; img.data[k + 2] = b; img.data[k + 3] = 255;
    }
  }
  const off = document.createElement("canvas");
  off.width = iw; off.height = ih;
  off.getContext("2d").putImageData(img, 0, 0);
  ctx.imageSmoothingEnabled = true;
  ctx.drawImage(off, 0, 0, W, H);
  // window circle (horizontal axis is y when transposed, else x)
  const [Lx, Ly] = model.get("extent") || [1, 1];
  const cx = drag.cx != null ? drag.cx : model.get("window_cx");
  const cy = drag.cy != null ? drag.cy : model.get("window_cy");
  const side = model.get("window_side");
  const lh = transpose ? Ly : Lx, lv = transpose ? Lx : Ly;
  const ch = transpose ? cy : cx, cv = transpose ? cx : cy;
  const cxp = (ch / lh) * W, cyp = (cv / lv) * H, rx = (side / 2 / lh) * W, ry = (side / 2 / lv) * H;
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


// ---- angular-symmetry panel (the 13 x N_r network input) -----------------
const MAGMA = [[0.0, [0, 0, 4]], [0.25, [81, 18, 124]], [0.5, [183, 55, 121]],
               [0.75, [252, 137, 97]], [1.0, [252, 253, 191]]];

function rampRGB(stops, t) {
  t = Math.max(0, Math.min(1, t));
  for (let i = 1; i < stops.length; i++) {
    if (t <= stops[i][0]) {
      const [t0, c0] = stops[i - 1], [t1, c1] = stops[i];
      const f = (t - t0) / (t1 - t0 || 1);
      return [Math.round(lerp(c0[0], c1[0], f)), Math.round(lerp(c0[1], c1[1], f)),
              Math.round(lerp(c0[2], c1[2], f))];
    }
  }
  return stops[stops.length - 1][1];
}

function drawPolar(canvas, model) {
  const vals = model.get("polar_values") || [];
  const shape = model.get("polar_shape") || [0, 0];
  const M = shape[0], NR = shape[1];
  const ctx = canvas.getContext("2d");
  const W = canvas.width, H = canvas.height;
  ctx.clearRect(0, 0, W, H);
  if (!vals.length || !M || !NR) return;

  const gamma = model.get("polar_gamma") || 1.0;
  const perCh = model.get("polar_per_channel");
  const skip0 = model.get("polar_skip_m0");
  const rMax = model.get("polar_r_max") || 10;

  // Row 0 is the radial profile and is far larger than the modulation rows,
  // so it is excluded from a shared colour scale by default.
  // Row 0 is signed (the pair correlation dips below zero), so each row is
  // mapped from its own min to its own max.  Taking |.| instead would fold
  // the negative lobes and put a notch at every zero crossing.
  const lo = new Array(M).fill(0), scale = new Array(M).fill(1);
  if (perCh) {
    for (let m = 0; m < M; m++) {
      let mn = Infinity, mx = -Infinity;
      for (let i = 0; i < NR; i++) {
        const v = vals[m * NR + i];
        if (!Number.isFinite(v)) continue;
        if (v < mn) mn = v;
        if (v > mx) mx = v;
      }
      lo[m] = Number.isFinite(mn) ? mn : 0;
      scale[m] = mx > lo[m] ? mx - lo[m] : 1;
    }
  } else {
    let mx = 0;
    for (let m = (skip0 && M > 1) ? 1 : 0; m < M; m++) {
      for (let i = 0; i < NR; i++) { const v = Math.abs(vals[m * NR + i]); if (Number.isFinite(v) && v > mx) mx = v; }
    }
    for (let m = 0; m < M; m++) scale[m] = mx > 0 ? mx : 1;
  }

  const off = document.createElement("canvas");
  off.width = NR; off.height = M;
  const octx = off.getContext("2d");
  const img = octx.createImageData(NR, M);
  for (let m = 0; m < M; m++) {
    for (let i = 0; i < NR; i++) {
      let t = (vals[m * NR + i] - lo[m]) / scale[m];
      t = Math.pow(Math.max(0, Math.min(1, t)), gamma);
      const c = rampRGB(MAGMA, t);
      const k = (m * NR + i) * 4;
      img.data[k] = c[0]; img.data[k + 1] = c[1]; img.data[k + 2] = c[2]; img.data[k + 3] = 255;
    }
  }
  octx.putImageData(img, 0, 0);

  const ML = 26, MB = 16, MT = 12, MR = 4;
  const pw = W - ML - MR, ph = H - MT - MB;
  ctx.imageSmoothingEnabled = false;
  ctx.drawImage(off, ML, MT, pw, ph);

  ctx.fillStyle = "#444";
  ctx.font = "9px sans-serif";
  ctx.textAlign = "right";
  const rowH = ph / M;
  const ords = model.get("polar_orders") || [];
  for (let m = 0; m < M; m++) {
    const lab = ords.length === M ? ords[m] : m;
    if (M <= 8 || m % 2 === 0) ctx.fillText(String(lab), ML - 3, MT + (m + 0.75) * rowH);
  }
  ctx.textAlign = "center";
  for (let k = 0; k <= 2; k++) {
    const r = (rMax * k) / 2;
    ctx.fillText(r.toFixed(0), ML + (r / rMax) * pw, H - 4);
  }
  ctx.textAlign = "left";
  ctx.fillText("order m", 2, 9);
  ctx.textAlign = "right";
  ctx.fillText("r (Å)", W - 2, 9);
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

  const layout = model.get("layout") || "side";
  const transpose = model.get("transpose");
  let frameCanvas, histCanvas, g3Canvas, g2Canvas, vmaxInput;
  let polarCanvas, gammaSlider, gammaVal, perChBox;
  let thickSlider, thickVal, dfSlider, dfVal;
  const nThick = (model.get("thicknesses") || [0]).length;

  function mkPolarControls(parent) {
    el("span", "px-label", parent).textContent = "γ";
    gammaSlider = el("input", null, parent);
    gammaSlider.type = "range"; gammaSlider.min = 0.1; gammaSlider.max = 1.0;
    gammaSlider.step = 0.05; gammaSlider.value = model.get("polar_gamma");
    gammaSlider.style.width = "70px";
    gammaVal = el("span", "px-val", parent);
    const lab = el("label", "px-label", parent);
    perChBox = el("input", null, lab);
    perChBox.type = "checkbox";
    perChBox.checked = !!model.get("polar_per_channel");
    lab.appendChild(document.createTextNode(" per-order"));
  }
  function mkVmax(parent) {
    el("span", "px-label", parent).textContent = "g3 max";
    const v = el("input", null, parent);
    v.type = "number"; v.min = 0; v.step = 0.5; v.placeholder = "auto"; v.style.width = "60px";
    return v;
  }

  if (layout === "stacked") {
    // Wide cell: full-width frame on top, analysis panels in one row beneath.
    const wrap = el("div", "px-wrap px-stacked", root);
    const lh = transpose ? Ly : Lx, lv = transpose ? Lx : Ly;
    frameCanvas = el("canvas", "px-frame", wrap);
    frameCanvas.width = 960;
    frameCanvas.height = Math.max(80, Math.min(190, Math.round(960 * lv / lh)));
    [thickSlider, thickVal] = mkRange(el("div", "px-row", wrap), "thick", 0, nThick - 1, 1, model.get("thickness_index"));
    [dfSlider, dfVal] = mkRange(el("div", "px-row", wrap), "defocus", -120, 120, 5, model.get("defocus_offset"));

    const panels = el("div", "px-wrap px-panels", wrap);
    const cpol = el("div", "px-col", panels);
    polarCanvas = el("canvas", "px-polar", cpol); polarCanvas.width = 340; polarCanvas.height = 200;
    mkPolarControls(el("div", "px-row", cpol));
    histCanvas = el("canvas", "px-hist", cpol); histCanvas.width = 340; histCanvas.height = 46;
    const right = el("div", "px-col", panels);
    g3Canvas = el("canvas", "px-g3", right); g3Canvas.width = 340; g3Canvas.height = 186;
    g2Canvas = el("canvas", "px-g2", right); g2Canvas.width = 340; g2Canvas.height = 124;
    vmaxInput = mkVmax(el("div", "px-row", right));
  } else {
    // Three columns (frame | angular symmetry | g3 + g2) so the frame and the
    // angular map stay side by side while dragging the window.
    const wrap = el("div", "px-wrap", root);
    const left = el("div", "px-col", wrap);
    frameCanvas = el("canvas", "px-frame", left);
    frameCanvas.width = 300; frameCanvas.height = Math.round(300 * Ly / Lx);
    [thickSlider, thickVal] = mkRange(el("div", "px-row", left), "thick", 0, nThick - 1, 1, model.get("thickness_index"));
    [dfSlider, dfVal] = mkRange(el("div", "px-row", left), "defocus", -120, 120, 5, model.get("defocus_offset"));

    const mid = el("div", "px-col", wrap);
    polarCanvas = el("canvas", "px-polar", mid); polarCanvas.width = 330; polarCanvas.height = 196;
    mkPolarControls(el("div", "px-row", mid));
    histCanvas = el("canvas", "px-hist", mid); histCanvas.width = 330; histCanvas.height = 46;

    const right = el("div", "px-col", wrap);
    g3Canvas = el("canvas", "px-g3", right); g3Canvas.width = 330; g3Canvas.height = 186;
    g2Canvas = el("canvas", "px-g2", right); g2Canvas.width = 330; g2Canvas.height = 124;
    vmaxInput = mkVmax(el("div", "px-row", right));
  }

  const status = el("div", "px-status", root);
  const drag = { cx: null, cy: null, active: false };

  const sgn = (x) => (x >= 0 ? "+" : "");
  function syncControls() {
    if (gammaSlider) {
      gammaSlider.value = model.get("polar_gamma");
      gammaVal.textContent = (model.get("polar_gamma") || 1).toFixed(2);
      perChBox.checked = !!model.get("polar_per_channel");
    }
    // Labels + status only — never reset slider thumbs (Python does not move
    // these controls, so resetting them is what made the defocus thumb jump).
    const thicks = model.get("thicknesses") || [0];
    thickVal.textContent = `${(thicks[model.get("thickness_index")] || 0).toFixed(0)} Å`;
    dfVal.textContent = `Δf ${sgn(model.get("defocus_offset"))}${model.get("defocus_offset").toFixed(0)} Å`;
    status.textContent = model.get("status") || "";
  }
  function redrawFrame() { ensureClims(); drawFrame(frameCanvas, model, drag, frameClim); drawHist(histCanvas, model.get("slice_values") || [], frameClim, histAxis); }
  function redrawAll() { redrawFrame(); drawPolar(polarCanvas, model); drawG3(g3Canvas, model); drawG2(g2Canvas, model); syncControls(); }

  // --- window drag on the frame ---
  function pointerToAngstrom(ev) {
    const rect = frameCanvas.getBoundingClientRect();
    const fh = (ev.clientX - rect.left) / rect.width, fv = (ev.clientY - rect.top) / rect.height;
    // transposed: horizontal is y, vertical is x.
    return transpose ? [fv * Lx, fh * Ly] : [fh * Lx, fv * Ly];
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
  gammaSlider.addEventListener("input", () => {
    gammaVal.textContent = parseFloat(gammaSlider.value).toFixed(2);
    model.set("polar_gamma", parseFloat(gammaSlider.value));
    model.save_changes();
  });
  perChBox.addEventListener("change", () => {
    model.set("polar_per_channel", perChBox.checked);
    model.save_changes();
  });
  vmaxInput.addEventListener("change", () => {
    const v = parseFloat(vmaxInput.value);
    model.set("g3_vmax", Number.isFinite(v) && v > 0 ? v : -1.0); model.save_changes();
  });

  // react to Python-side updates (frame/inputs change; clims stay fixed)
  model.on("change:slice_values", redrawFrame);
  model.on("change:polar_values", () => drawPolar(polarCanvas, model));
  model.on("change:polar_gamma", () => { drawPolar(polarCanvas, model); syncControls(); });
  model.on("change:polar_per_channel", () => { drawPolar(polarCanvas, model); syncControls(); });
  model.on("change:g3_slice_values", () => drawG3(g3Canvas, model));
  model.on("change:g2", () => drawG2(g2Canvas, model));
  model.on("change:g3_vmax", () => drawG3(g3Canvas, model));
  model.on("change:status", syncControls);
  model.on("change:thickness", syncControls);
  model.on("change:defocus", syncControls);

  redrawAll();
}

export default { render };
