// PtychoExplorer — blurred potential slice  <->  local weighted g2 / g3.
// Hand-authored ESM for anywidget (no bundler), matching g3_explorer.js style.

// ---- colormaps -----------------------------------------------------------
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

function grey(t) {
  t = Math.max(0, Math.min(1, t));
  const v = Math.round(255 * t);
  return [v, v, v];
}

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

// ---- left panel: potential slice + window box ----------------------------
function drawSlice(canvas, model, drag) {
  const vals = model.get("slice_values") || [];
  const shape = model.get("slice_shape") || [0, 0];
  const [nx, ny] = shape;
  const ctx = canvas.getContext("2d");
  const W = canvas.width, H = canvas.height;
  ctx.clearRect(0, 0, W, H);
  if (!vals.length || nx * ny !== vals.length) return;

  const finite = vals.filter((v) => Number.isFinite(v)).slice().sort((a, b) => a - b);
  const vmin = percentile(finite, 0.01), vmax = percentile(finite, 0.99);
  const span = vmax - vmin || 1;

  // transpose => long axis (y) horizontal, x vertical (for wide cells).
  const transpose = model.get("transpose");
  const iw = transpose ? ny : nx, ih = transpose ? nx : ny;
  const img = ctx.createImageData(iw, ih);
  for (let i = 0; i < nx; i++) {
    for (let j = 0; j < ny; j++) {
      const v = vals[i * ny + j];
      const [r, g, b] = grey((v - vmin) / span);
      const k = (transpose ? (i * ny + j) : (j * nx + i)) * 4;
      img.data[k] = r; img.data[k + 1] = g; img.data[k + 2] = b; img.data[k + 3] = 255;
    }
  }
  const off = document.createElement("canvas");
  off.width = iw; off.height = ih;
  off.getContext("2d").putImageData(img, 0, 0);
  ctx.imageSmoothingEnabled = true;
  ctx.drawImage(off, 0, 0, W, H);

  // window (red circle of radius side/2 — isotropic; horizontal axis is y
  // when transposed, else x).
  const [Lx, Ly] = model.get("extent") || [1, 1];
  const cx = drag.cx != null ? drag.cx : model.get("window_cx");
  const cy = drag.cy != null ? drag.cy : model.get("window_cy");
  const side = model.get("window_side");
  const lh = transpose ? Ly : Lx, lv = transpose ? Lx : Ly;
  const ch = transpose ? cy : cx, cv = transpose ? cx : cy;
  const cxp = (ch / lh) * W, cyp = (cv / lv) * H;
  const rx = (side / 2 / lh) * W, ry = (side / 2 / lv) * H;
  ctx.strokeStyle = "#e02424";
  ctx.lineWidth = 2;
  ctx.fillStyle = "rgba(224,36,36,0.08)";
  // Draw at all periodic offsets so the circle wraps; canvas clips the rest.
  for (const ox of [-W, 0, W]) {
    for (const oy of [-H, 0, H]) {
      ctx.beginPath();
      ctx.ellipse(cxp + ox, cyp + oy, rx, ry, 0, 0, 2 * Math.PI);
      ctx.fill();
      ctx.stroke();
    }
  }
}

function drawHist(canvas, model) {
  const counts = model.get("hist_counts") || [];
  const edges = model.get("hist_edges") || [];
  const ctx = canvas.getContext("2d");
  const W = canvas.width, H = canvas.height;
  ctx.clearRect(0, 0, W, H);
  if (!counts.length) return;
  const cmax = Math.max(...counts) || 1;
  const ml = 4, mb = 16, pw = W - ml - 4, ph = H - mb - 4;
  ctx.fillStyle = "#5b8fb0";
  const bw = pw / counts.length;
  for (let i = 0; i < counts.length; i++) {
    const h = (counts[i] / cmax) * ph;
    ctx.fillRect(ml + i * bw, 4 + ph - h, Math.max(1, bw - 0.5), h);
  }
  ctx.fillStyle = "#444";
  ctx.font = "10px sans-serif";
  ctx.textAlign = "left";
  ctx.fillText(`${edges[0].toFixed(2)}`, ml, H - 4);
  ctx.textAlign = "right";
  ctx.fillText(`${edges[edges.length - 1].toFixed(2)} rad`, W - 4, H - 4);
}

// ---- input preview: the rotated, circular-windowed crop the net sees ------
// ---- right panels: g3 heatmap + g2 line (shared r axis) -------------------
const MARGIN = { left: 42, right: 56, top: 18, bottom: 26 };

function rToX(r, rMax, W) { return MARGIN.left + (r / rMax) * (W - MARGIN.left - MARGIN.right); }

function drawG3(canvas, model) {
  const vals = model.get("g3_slice_values") || [];
  const shape = model.get("g3_slice_shape") || [0, 0];
  const [nphi, nr] = shape;
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
      // angle axis: phi=0 at bottom -> invert row
      ctx.fillRect(MARGIN.left + i * cw, MARGIN.top + (nphi - 1 - p) * chh, cw + 0.6, chh + 0.6);
    }
  }
  // axes
  ctx.strokeStyle = "#888"; ctx.lineWidth = 1;
  ctx.strokeRect(MARGIN.left, MARGIN.top, pw, ph);
  ctx.fillStyle = "#333"; ctx.font = "10px sans-serif";
  ctx.textAlign = "right";
  [0, 45, 90, 135, 180].forEach((a) => {
    const y = MARGIN.top + (1 - a / 180) * ph;
    ctx.fillText(`${a}`, MARGIN.left - 4, y + 3);
  });
  ctx.save();
  ctx.translate(11, MARGIN.top + ph / 2); ctx.rotate(-Math.PI / 2);
  ctx.textAlign = "center"; ctx.fillText("angle (deg)", 0, 0); ctx.restore();
  ctx.textAlign = "left"; ctx.fillText("weighted g3 slice (r01 ~ NN band)", MARGIN.left, 12);

  // colorbar (RdBu_r, centred on g3 = 1)
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

  // NN band shading
  ctx.fillStyle = "rgba(224,36,36,0.10)";
  ctx.fillRect(rToX(band[0], rMax, W), MARGIN.top, rToX(band[1], rMax, W) - rToX(band[0], rMax, W), ph);
  // g=1 reference
  ctx.strokeStyle = "#bbb"; ctx.setLineDash([4, 3]); ctx.beginPath();
  ctx.moveTo(MARGIN.left, yToPx(1)); ctx.lineTo(W - MARGIN.right, yToPx(1)); ctx.stroke();
  ctx.setLineDash([]);
  // curve
  ctx.strokeStyle = "#1f6f43"; ctx.lineWidth = 1.5; ctx.beginPath();
  for (let i = 0; i < g2.length; i++) {
    const x = rToX(rArr[i] || (i * rMax) / g2.length, rMax, W);
    const y = yToPx(Number.isFinite(g2[i]) ? g2[i] : 0);
    i === 0 ? ctx.moveTo(x, y) : ctx.lineTo(x, y);
  }
  ctx.stroke();
  // axes
  ctx.strokeStyle = "#888"; ctx.lineWidth = 1; ctx.strokeRect(MARGIN.left, MARGIN.top, pw, ph);
  ctx.fillStyle = "#333"; ctx.font = "10px sans-serif";
  ctx.textAlign = "center";
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
  const scale = new Array(M).fill(1);
  if (perCh) {
    for (let m = 0; m < M; m++) {
      let mx = 0;
      for (let i = 0; i < NR; i++) { const v = vals[m * NR + i]; if (Number.isFinite(v) && v > mx) mx = v; }
      scale[m] = mx > 0 ? mx : 1;
    }
  } else {
    let mx = 0;
    for (let m = (skip0 && M > 1) ? 1 : 0; m < M; m++) {
      for (let i = 0; i < NR; i++) { const v = vals[m * NR + i]; if (Number.isFinite(v) && v > mx) mx = v; }
    }
    for (let m = 0; m < M; m++) scale[m] = mx > 0 ? mx : 1;
  }

  const off = document.createElement("canvas");
  off.width = NR; off.height = M;
  const octx = off.getContext("2d");
  const img = octx.createImageData(NR, M);
  for (let m = 0; m < M; m++) {
    for (let i = 0; i < NR; i++) {
      let t = vals[m * NR + i] / scale[m];
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
  root.classList.add("ptycho-explorer");
  const layout = model.get("layout") || "side";
  const transpose = model.get("transpose");
  const [Lx, Ly] = model.get("extent") || [1, 1];

  let sliceCanvas, sliceSlider, sliceVal, histCanvas, g3Canvas, g2Canvas, vmaxInput;
  let polarCanvas, gammaSlider, gammaVal, perChBox;

  function mkSlider(parent) {
    el("span", "px-label", parent).textContent = "slice";
    const s = el("input", null, parent);
    s.type = "range"; s.min = 0; s.step = 1;
    return s;
  }
  function mkPolar(parent, w, h) {
    const c = el("canvas", "px-polar", parent);
    c.width = w; c.height = h;
    return c;
  }
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
    v.type = "number"; v.min = 0; v.step = 0.5; v.placeholder = "auto";
    v.style.width = "60px";
    return v;
  }

  function mkHist(parent, w) {
    const c = el("canvas", "px-hist", parent);
    c.width = w; c.height = 46;
    return c;
  }

  if (layout === "stacked") {
    // Wide cell: full-width slice on top, analysis panels in one row beneath.
    const wrap = el("div", "px-wrap px-stacked", root);
    const lh = transpose ? Ly : Lx, lv = transpose ? Lx : Ly;
    sliceCanvas = el("canvas", "px-slice", wrap);
    sliceCanvas.width = 960;
    sliceCanvas.height = Math.max(80, Math.min(190, Math.round(960 * lv / lh)));
    const row = el("div", "px-row", wrap);
    sliceSlider = mkSlider(row);
    sliceVal = el("span", "px-val", row);

    const panels = el("div", "px-wrap px-panels", wrap);
    const cpol = el("div", "px-col", panels);
    polarCanvas = mkPolar(cpol, 340, 200);
    mkPolarControls(el("div", "px-row", cpol));
    histCanvas = mkHist(cpol, 340);
    const ccor = el("div", "px-col", panels);
    g3Canvas = el("canvas", "px-g3", ccor); g3Canvas.width = 340; g3Canvas.height = 186;
    g2Canvas = el("canvas", "px-g2", ccor); g2Canvas.width = 340; g2Canvas.height = 124;
    vmaxInput = mkVmax(el("div", "px-row", ccor));
  } else {
    // Three columns (slice | angular symmetry | g3 + g2) rather than one tall
    // stack, so the slice and the angular map stay side by side while dragging.
    const wrap = el("div", "px-wrap", root);
    const left = el("div", "px-col px-left", wrap);
    sliceCanvas = el("canvas", "px-slice", left);
    sliceCanvas.width = 300; sliceCanvas.height = 300;
    const srow = el("div", "px-row", left);
    sliceSlider = mkSlider(srow);
    sliceVal = el("span", "px-val", srow);

    const mid = el("div", "px-col", wrap);
    polarCanvas = mkPolar(mid, 330, 196);
    mkPolarControls(el("div", "px-row", mid));
    histCanvas = mkHist(mid, 330);

    const right = el("div", "px-col px-right", wrap);
    g3Canvas = el("canvas", "px-g3", right); g3Canvas.width = 330; g3Canvas.height = 186;
    g2Canvas = el("canvas", "px-g2", right); g2Canvas.width = 330; g2Canvas.height = 124;
    vmaxInput = mkVmax(el("div", "px-row", right));
  }

  const status = el("div", "px-status", root);

  const drag = { cx: null, cy: null, active: false };

  function syncControls() {
    sliceSlider.max = (model.get("n_slices") || 1) - 1;
    sliceSlider.value = model.get("slice_index");
    sliceVal.textContent = `${model.get("slice_index")} (z=${model.get("z0").toFixed(1)} Å)`;
    gammaSlider.value = model.get("polar_gamma");
    gammaVal.textContent = (model.get("polar_gamma") || 1).toFixed(2);
    perChBox.checked = !!model.get("polar_per_channel");
    status.textContent = model.get("status") || "";
  }
  function redrawAll() {
    drawSlice(sliceCanvas, model, drag);
    drawHist(histCanvas, model);
    drawPolar(polarCanvas, model);
    drawG3(g3Canvas, model);
    drawG2(g2Canvas, model);
    syncControls();
  }

  // --- window drag on the slice canvas ---
  function pointerToAngstrom(ev) {
    const rect = sliceCanvas.getBoundingClientRect();
    const fh = (ev.clientX - rect.left) / rect.width;
    const fv = (ev.clientY - rect.top) / rect.height;
    // transposed: horizontal is y, vertical is x.
    return transpose ? [fv * Lx, fh * Ly] : [fh * Lx, fv * Ly];
  }
  function clampCenter(c, L) {
    // Window wraps across periodic faces, so the centre may sit anywhere.
    return ((c % L) + L) % L;
  }
  sliceCanvas.addEventListener("pointerdown", (ev) => {
    drag.active = true;
    const [Lx, Ly] = model.get("extent") || [1, 1];
    const [ax, ay] = pointerToAngstrom(ev);
    drag.cx = clampCenter(ax, Lx); drag.cy = clampCenter(ay, Ly);
    sliceCanvas.setPointerCapture(ev.pointerId);
    drawSlice(sliceCanvas, model, drag);
  });
  sliceCanvas.addEventListener("pointermove", (ev) => {
    if (!drag.active) return;
    const [Lx, Ly] = model.get("extent") || [1, 1];
    const [ax, ay] = pointerToAngstrom(ev);
    drag.cx = clampCenter(ax, Lx); drag.cy = clampCenter(ay, Ly);
    drawSlice(sliceCanvas, model, drag); // live box only; correlations on release
  });
  function endDrag() {
    if (!drag.active) return;
    drag.active = false;
    model.set("window_cx", drag.cx);
    model.set("window_cy", drag.cy);
    model.save_changes();
    drag.cx = null; drag.cy = null;
  }
  sliceCanvas.addEventListener("pointerup", endDrag);
  sliceCanvas.addEventListener("pointercancel", endDrag);

  sliceSlider.addEventListener("input", () => {
    model.set("slice_index", parseInt(sliceSlider.value, 10));
    model.save_changes();
  });
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
    model.set("g3_vmax", Number.isFinite(v) && v > 0 ? v : -1.0);
    model.save_changes();
  });

  // react to Python-side updates
  model.on("change:slice_values", () => { drawSlice(sliceCanvas, model, drag); drawHist(histCanvas, model); });
  model.on("change:g3_slice_values", () => drawG3(g3Canvas, model));
  model.on("change:g2", () => drawG2(g2Canvas, model));
  model.on("change:g3_vmax", () => drawG3(g3Canvas, model));
  model.on("change:polar_values", () => drawPolar(polarCanvas, model));
  model.on("change:polar_gamma", () => { drawPolar(polarCanvas, model); syncControls(); });
  model.on("change:polar_per_channel", () => { drawPolar(polarCanvas, model); syncControls(); });
  model.on("change:status", syncControls);
  model.on("change:slice_index", syncControls);

  redrawAll();
}

export default { render };
