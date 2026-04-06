function clamp(value, min, max) {
  return Math.min(max, Math.max(min, value));
}

function makeColor(value, vmin, vmax, midpoint = null) {
  const span = Math.max(vmax - vmin, 1e-12);
  let t = clamp((value - vmin) / span, 0, 1);
  const stops = [
    [0.0, [19, 53, 97]],
    [0.2, [61, 122, 176]],
    [0.4, [150, 198, 226]],
    [0.56, [247, 243, 226]],
    [0.76, [233, 165, 84]],
    [1.0, [134, 32, 39]],
  ];
  const coolStops = [
    [0.0, [19, 53, 97]],
    [0.38, [84, 151, 201]],
    [0.72, [189, 220, 236]],
    [1.0, [247, 243, 226]],
  ];
  const warmStops = [
    [0.0, [247, 243, 226]],
    [0.35, [250, 203, 141]],
    [0.68, [219, 109, 71]],
    [1.0, [134, 32, 39]],
  ];

  if (midpoint !== null && midpoint > vmin && midpoint < vmax) {
    if (value <= midpoint) {
      t = 0.56 * clamp((value - vmin) / Math.max(midpoint - vmin, 1e-12), 0, 1);
    } else {
      t =
        0.56 +
        0.44 * clamp((value - midpoint) / Math.max(vmax - midpoint, 1e-12), 0, 1);
    }
  }
  if (midpoint !== null && midpoint >= vmax) {
    t = clamp((value - vmin) / Math.max(vmax - vmin, 1e-12), 0, 1);
    return interpolateColor(coolStops, t);
  }
  if (midpoint !== null && midpoint <= vmin) {
    t = clamp((value - vmin) / Math.max(vmax - vmin, 1e-12), 0, 1);
    return interpolateColor(warmStops, t);
  }

  return interpolateColor(stops, t);
}

function interpolateColor(stops, t) {
  
  let left = stops[0];
  let right = stops[stops.length - 1];
  for (let i = 0; i < stops.length - 1; i++) {
    if (t >= stops[i][0] && t <= stops[i + 1][0]) {
      left = stops[i];
      right = stops[i + 1];
      break;
    }
  }
  const local = (t - left[0]) / Math.max(right[0] - left[0], 1e-12);
  const rgb = left[1].map((start, idx) =>
    Math.round(start + local * (right[1][idx] - start))
  );
  return `rgb(${rgb[0]}, ${rgb[1]}, ${rgb[2]})`;
}

function percentile(values, p) {
  if (!values.length) return 1;
  const sorted = [...values].sort((a, b) => a - b);
  const idx = clamp(Math.floor(p * (sorted.length - 1)), 0, sorted.length - 1);
  return sorted[idx];
}

function makeEdgeTicks(edges, targetCount = 6) {
  if (!edges.length) return [0];
  if (edges.length <= targetCount + 1) return [...edges];
  const stride = Math.max(1, Math.ceil((edges.length - 1) / (targetCount - 1)));
  const ticks = [];
  for (let idx = 0; idx < edges.length; idx += stride) {
    ticks.push(edges[idx]);
  }
  const last = edges[edges.length - 1];
  if (ticks[ticks.length - 1] !== last) {
    ticks.push(last);
  }
  return ticks;
}

function formatTick(value) {
  if (Math.abs(value) < 1e-12) return "0";
  if (Math.abs(value - Math.round(value)) < 1e-9) return value.toFixed(0);
  return value.toFixed(1);
}

function uniqueTicks(values) {
  const out = [];
  values
    .filter((value) => Number.isFinite(value))
    .sort((a, b) => a - b)
    .forEach((value) => {
      if (!out.length || Math.abs(value - out[out.length - 1]) > 1e-6) {
        out.push(value);
      }
    });
  return out;
}

function updateSelect(select, labels, index) {
  select.replaceChildren();
  labels.forEach((label, idx) => {
    const option = document.createElement("option");
    option.value = String(idx);
    option.textContent = label;
    option.selected = idx === index;
    select.appendChild(option);
  });
}

function drawHeatmap(canvas, imageValues, shape, rEdges, phiEdgesDeg, title, normalize, sliceMax) {
  const ctx = canvas.getContext("2d");
  const dpr = window.devicePixelRatio || 1;
  const width = Math.max(Math.round(canvas.clientWidth || 820), 560);
  const height = Math.max(Math.round(canvas.clientHeight || 250), 180);
  const margin = { top: 30, right: 92, bottom: 42, left: 84 };
  const plotWidth = width - margin.left - margin.right;
  const plotHeight = height - margin.top - margin.bottom;
  const xMax = Math.max(rEdges.length ? rEdges[rEdges.length - 1] : 1, 1e-12);
  canvas.width = width * dpr;
  canvas.height = height * dpr;
  ctx.setTransform(dpr, 0, 0, dpr, 0, 0);
  ctx.clearRect(0, 0, width, height);

  ctx.fillStyle = "#f7f8f5";
  ctx.fillRect(0, 0, width, height);

  ctx.fillStyle = "#162134";
  ctx.font = "600 15px ui-sans-serif, system-ui, sans-serif";
  ctx.fillText(title, margin.left, 20);

  const phiCount = shape[0];
  const rCount = shape[1];
  const finiteValues = imageValues.filter((value) => Number.isFinite(value));
  const autoVmax = Math.max(percentile(finiteValues, 0.995) || 1, normalize ? 1.0 : 1e-6);
  const manualVmax = sliceMax > 0 ? sliceMax : null;
  const vmax = manualVmax ?? autoVmax;
  const vmin = 0;
  const midpoint = normalize ? 1.0 : 0.5 * (vmin + vmax);
  const cellHeight = plotHeight / Math.max(phiCount, 1);

  for (let phiIdx = 0; phiIdx < phiCount; phiIdx++) {
    for (let rIdx = 0; rIdx < rCount; rIdx++) {
      const value = imageValues[phiIdx * rCount + rIdx];
      ctx.fillStyle = makeColor(value, vmin, vmax, midpoint);
      const x0 = rEdges[rIdx] ?? 0;
      const x1 = rEdges[rIdx + 1] ?? x0;
      const x = margin.left + (x0 / xMax) * plotWidth;
      const cellWidth = ((x1 - x0) / xMax) * plotWidth;
      const y = margin.top + plotHeight - (phiIdx + 1) * cellHeight;
      ctx.fillRect(x, y, cellWidth + 0.5, cellHeight + 0.5);
    }
  }

  ctx.strokeStyle = "#3a506b";
  ctx.lineWidth = 1;
  ctx.strokeRect(margin.left, margin.top, plotWidth, plotHeight);

  ctx.fillStyle = "#2d3c4f";
  ctx.font = "12px ui-sans-serif, system-ui, sans-serif";
  ctx.textAlign = "center";
  ctx.fillText("r", margin.left + plotWidth / 2, height - 10);

  ctx.save();
  ctx.translate(18, margin.top + plotHeight / 2);
  ctx.rotate(-Math.PI / 2);
  ctx.fillText("phi (deg)", 0, 0);
  ctx.restore();

  const xTickValues = makeEdgeTicks(rEdges);
  xTickValues.forEach((value) => {
    const x = margin.left + (value / xMax) * plotWidth;
    ctx.beginPath();
    ctx.moveTo(x, margin.top + plotHeight);
    ctx.lineTo(x, margin.top + plotHeight + 5);
    ctx.stroke();
    ctx.fillText(formatTick(value), x, margin.top + plotHeight + 18);
  });

  const phiMax = phiEdgesDeg[phiEdgesDeg.length - 1] || 180;
  const yTickValues = [0, 45, 90, 135, 180].filter((value) => value <= phiMax);
  ctx.textAlign = "right";
  yTickValues.forEach((value) => {
    const y = margin.top + plotHeight - (value / phiMax) * plotHeight;
    ctx.beginPath();
    ctx.moveTo(margin.left - 5, y);
    ctx.lineTo(margin.left, y);
    ctx.stroke();
    ctx.fillText(formatTick(value), margin.left - 8, y + 4);
  });

  const bar = {
    x: margin.left + plotWidth + 20,
    y: margin.top + 10,
    width: 18,
    height: plotHeight - 20,
  };
  const steps = 160;
  for (let step = 0; step < steps; step++) {
    const t0 = step / steps;
    const value = vmax - t0 * (vmax - vmin);
    const y = bar.y + t0 * bar.height;
    ctx.fillStyle = makeColor(value, vmin, vmax, midpoint);
    ctx.fillRect(bar.x, y, bar.width, bar.height / steps + 1);
  }
  ctx.strokeStyle = "#405968";
  ctx.strokeRect(bar.x, bar.y, bar.width, bar.height);

  const legendTicks =
    normalize && vmax >= 1.0
      ? uniqueTicks([0, 1, vmax > 1.5 ? 0.5 * (1 + vmax) : NaN, vmax])
      : uniqueTicks([0, 0.5 * vmax, vmax]);

  ctx.textAlign = "left";
  legendTicks.forEach((value) => {
    const y = bar.y + (1 - value / Math.max(vmax, 1e-12)) * bar.height;
    ctx.beginPath();
    ctx.moveTo(bar.x + bar.width, y);
    ctx.lineTo(bar.x + bar.width + 5, y);
    ctx.stroke();
    ctx.fillText(formatTick(value), bar.x + bar.width + 8, y + 4);
  });

  ctx.save();
  ctx.translate(bar.x + bar.width + 34, bar.y + bar.height / 2);
  ctx.rotate(-Math.PI / 2);
  ctx.textAlign = "center";
  ctx.fillText(normalize ? "x random" : "slice weight", 0, 0);
  ctx.restore();
}

function drawPairProfile(svg, profile, rValues, rEdges, selection, dragging) {
  const width = Math.max(Math.round(svg.clientWidth || 820), 560);
  const height = Math.max(Math.round(svg.clientHeight || 180), 140);
  const margin = { top: 16, right: 92, bottom: 38, left: 84 };
  const plotWidth = width - margin.left - margin.right;
  const plotHeight = height - margin.top - margin.bottom;
  const yMax = Math.max(Math.max(...profile, 1e-12) * 1.05, 1.1);
  const xMax = Math.max(rEdges.length ? rEdges[rEdges.length - 1] : 1, 1e-12);

  const xScale = (value) => margin.left + (value / xMax) * plotWidth;
  const yScale = (value) => margin.top + plotHeight - (value / yMax) * plotHeight;

  const points = profile
    .map((value, idx) => `${xScale(rValues[idx])},${yScale(value)}`)
    .join(" ");

  const areaX0 = xScale(selection.min);
  const areaX1 = xScale(selection.max);

  svg.setAttribute("viewBox", `0 0 ${width} ${height}`);
  svg.innerHTML = `
    <rect class="panel-bg" x="0" y="0" width="${width}" height="${height}"></rect>
    <text class="panel-title" x="${margin.left}" y="12">2 body shell selector</text>
    <rect class="selection-area" x="${Math.min(areaX0, areaX1)}" y="${margin.top}" width="${Math.abs(areaX1 - areaX0)}" height="${plotHeight}"></rect>
    <line class="ideal-line" x1="${margin.left}" x2="${margin.left + plotWidth}" y1="${yScale(1.0)}" y2="${yScale(1.0)}"></line>
    <text class="ideal-label" x="${margin.left + 6}" y="${yScale(1.0) - 6}" text-anchor="start">g(r)=1</text>
    <polyline class="profile-line" points="${points}"></polyline>
    <line class="selection-handle ${dragging === "min" ? "dragging" : ""}" x1="${areaX0}" x2="${areaX0}" y1="${margin.top}" y2="${margin.top + plotHeight}"></line>
    <line class="selection-handle ${dragging === "max" ? "dragging" : ""}" x1="${areaX1}" x2="${areaX1}" y1="${margin.top}" y2="${margin.top + plotHeight}"></line>
    <rect class="plot-frame" x="${margin.left}" y="${margin.top}" width="${plotWidth}" height="${plotHeight}"></rect>
    <text class="axis-label" x="${margin.left + plotWidth / 2}" y="${height - 8}">r</text>
    <text class="axis-label" x="18" y="${margin.top + plotHeight / 2}" text-anchor="middle" transform="rotate(-90 18 ${margin.top + plotHeight / 2})">g(r)</text>
  `;

  const xTickValues = makeEdgeTicks(rEdges);
  const yTicks = [0, yMax / 2, yMax];

  xTickValues.forEach((value) => {
    const x = xScale(value);
    const tick = document.createElementNS("http://www.w3.org/2000/svg", "line");
    tick.setAttribute("class", "axis-tick");
    tick.setAttribute("x1", x);
    tick.setAttribute("x2", x);
    tick.setAttribute("y1", margin.top + plotHeight);
    tick.setAttribute("y2", margin.top + plotHeight + 5);
    svg.appendChild(tick);

    const label = document.createElementNS("http://www.w3.org/2000/svg", "text");
    label.setAttribute("class", "tick-label");
    label.setAttribute("x", x);
    label.setAttribute("y", margin.top + plotHeight + 18);
    label.setAttribute("text-anchor", "middle");
    label.textContent = formatTick(value);
    svg.appendChild(label);
  });

  yTicks.forEach((value) => {
    const y = yScale(value);
    const tick = document.createElementNS("http://www.w3.org/2000/svg", "line");
    tick.setAttribute("class", "axis-tick");
    tick.setAttribute("x1", margin.left - 5);
    tick.setAttribute("x2", margin.left);
    tick.setAttribute("y1", y);
    tick.setAttribute("y2", y);
    svg.appendChild(tick);

    const label = document.createElementNS("http://www.w3.org/2000/svg", "text");
    label.setAttribute("class", "tick-label");
    label.setAttribute("x", margin.left - 8);
    label.setAttribute("y", y + 4);
    label.setAttribute("text-anchor", "end");
    label.textContent = value.toFixed(2);
    svg.appendChild(label);
  });

  return { xScale, xMax, margin, plotWidth, plotHeight, width };
}

function render({ model, el }) {
  const root = document.createElement("div");
  root.className = "tricor-widget";

  const controls = document.createElement("div");
  controls.className = "tricor-controls";

  const select = document.createElement("select");
  select.className = "tricor-select";
  controls.appendChild(select);

  const sigmaControls = document.createElement("div");
  sigmaControls.className = "tricor-smoothing";

  const sigmaRLabel = document.createElement("label");
  sigmaRLabel.className = "tricor-number";
  const sigmaRText = document.createElement("span");
  sigmaRText.textContent = "sigma_r";
  const sigmaRInput = document.createElement("input");
  sigmaRInput.type = "number";
  sigmaRInput.min = "0";
  sigmaRInput.step = "0.05";
  sigmaRLabel.appendChild(sigmaRText);
  sigmaRLabel.appendChild(sigmaRInput);
  sigmaControls.appendChild(sigmaRLabel);

  const sigmaPhiLabel = document.createElement("label");
  sigmaPhiLabel.className = "tricor-number";
  const sigmaPhiText = document.createElement("span");
  sigmaPhiText.textContent = "sigma_phi";
  const sigmaPhiInput = document.createElement("input");
  sigmaPhiInput.type = "number";
  sigmaPhiInput.min = "0";
  sigmaPhiInput.step = "1";
  sigmaPhiLabel.appendChild(sigmaPhiText);
  sigmaPhiLabel.appendChild(sigmaPhiInput);
  sigmaControls.appendChild(sigmaPhiLabel);

  const sliceMaxLabel = document.createElement("label");
  sliceMaxLabel.className = "tricor-number";
  const sliceMaxText = document.createElement("span");
  sliceMaxText.textContent = "slice max";
  const sliceMaxInput = document.createElement("input");
  sliceMaxInput.type = "number";
  sliceMaxInput.min = "0";
  sliceMaxInput.step = "0.1";
  sliceMaxInput.placeholder = "auto";
  sliceMaxLabel.appendChild(sliceMaxText);
  sliceMaxLabel.appendChild(sliceMaxInput);
  sigmaControls.appendChild(sliceMaxLabel);

  controls.appendChild(sigmaControls);

  const toggleLabel = document.createElement("label");
  toggleLabel.className = "tricor-toggle";
  const checkbox = document.createElement("input");
  checkbox.type = "checkbox";
  toggleLabel.appendChild(checkbox);
  const toggleText = document.createElement("span");
  toggleText.textContent = "Normalize by ideal density";
  toggleLabel.appendChild(toggleText);
  controls.appendChild(toggleLabel);

  const status = document.createElement("div");
  status.className = "tricor-status";
  controls.appendChild(status);

  const topPanel = document.createElement("canvas");
  topPanel.className = "tricor-heatmap";
  const bottomPanel = document.createElementNS("http://www.w3.org/2000/svg", "svg");
  bottomPanel.classList.add("tricor-profile");
  bottomPanel.setAttribute("preserveAspectRatio", "xMinYMin meet");

  root.appendChild(controls);
  root.appendChild(topPanel);
  root.appendChild(bottomPanel);
  el.appendChild(root);

  let selection = {
    min: model.get("selection_min"),
    max: model.get("selection_max"),
  };
  let dragging = null;
  let resizeObserver = null;

  function commitSelection() {
    model.set("selection_min", selection.min);
    model.set("selection_max", selection.max);
    model.save_changes();
  }

  function redraw() {
    const labels = model.get("triplet_labels") || [];
    const tripletIndex = model.get("triplet_index") || 0;
    updateSelect(select, labels, tripletIndex);
    checkbox.checked = !!model.get("normalize");
    sigmaRInput.value = Number(model.get("sigma_r") || 0).toFixed(2);
    sigmaPhiInput.value = Number(model.get("sigma_phi") || 0).toFixed(1);
    const sliceMax = Number(model.get("slice_max") ?? -1);
    sliceMaxInput.value = sliceMax > 0 ? sliceMax.toFixed(2) : "";
    status.textContent = model.get("status") || "";

    const rValues = model.get("r") || [];
    const rEdges = model.get("r_edges") || [];
    const phiEdgesDeg = model.get("phi_edges_deg") || [];
    const pairProfile = model.get("pair_profile") || [];
    const sliceImage = model.get("slice_image") || [];
    const sliceShape = model.get("slice_shape") || [0, 0];
    selection.min = model.get("selection_min");
    selection.max = model.get("selection_max");

    drawHeatmap(
      topPanel,
      sliceImage,
      sliceShape,
      rEdges,
      phiEdgesDeg,
      labels[tripletIndex] || "3 body slice",
      checkbox.checked,
      sliceMax,
    );
    const scaleInfo = drawPairProfile(bottomPanel, pairProfile, rValues, rEdges, selection, dragging);

    const overlay = document.createElementNS("http://www.w3.org/2000/svg", "rect");
    overlay.setAttribute("class", "drag-overlay");
    overlay.setAttribute("x", scaleInfo.margin.left);
    overlay.setAttribute("y", scaleInfo.margin.top);
    overlay.setAttribute("width", scaleInfo.plotWidth);
    overlay.setAttribute("height", scaleInfo.plotHeight);
    bottomPanel.appendChild(overlay);

    const toValue = (clientX) => {
      const bounds = bottomPanel.getBoundingClientRect();
      const localX = ((clientX - bounds.left) / bounds.width) * scaleInfo.width;
      return clamp(
        ((localX - scaleInfo.margin.left) / scaleInfo.plotWidth) * scaleInfo.xMax,
        0,
        scaleInfo.xMax,
      );
    };

    overlay.onpointerdown = (event) => {
      const value = toValue(event.clientX);
      const distMin = Math.abs(value - selection.min);
      const distMax = Math.abs(value - selection.max);
      dragging = distMin <= distMax ? "min" : "max";
      overlay.setPointerCapture(event.pointerId);
      redraw();
    };

    overlay.onpointermove = (event) => {
      if (!dragging) return;
      const value = toValue(event.clientX);
      if (dragging === "min") {
        selection.min = clamp(value, 0, selection.max - 0.02);
      } else {
        selection.max = clamp(value, selection.min + 0.02, scaleInfo.xMax);
      }
      drawPairProfile(bottomPanel, pairProfile, rValues, rEdges, selection, dragging);
      bottomPanel.appendChild(overlay);
    };

    const finishDrag = () => {
      if (!dragging) return;
      dragging = null;
      commitSelection();
    };
    overlay.onpointerup = finishDrag;
    overlay.onpointercancel = finishDrag;
  }

  select.addEventListener("change", () => {
    model.set("triplet_index", Number.parseInt(select.value, 10));
    model.save_changes();
  });

  checkbox.addEventListener("change", () => {
    model.set("normalize", checkbox.checked);
    model.save_changes();
  });

  sigmaRInput.addEventListener("change", () => {
    model.set("sigma_r", Math.max(0, Number.parseFloat(sigmaRInput.value) || 0));
    model.save_changes();
  });

  sigmaPhiInput.addEventListener("change", () => {
    model.set("sigma_phi", Math.max(0, Number.parseFloat(sigmaPhiInput.value) || 0));
    model.save_changes();
  });

  sliceMaxInput.addEventListener("change", () => {
    const value = Number.parseFloat(sliceMaxInput.value);
    model.set("slice_max", Number.isFinite(value) && value > 0 ? value : -1.0);
    model.save_changes();
  });

  model.on(
    "change:triplet_labels change:triplet_index change:normalize change:sigma_r change:sigma_phi change:slice_max change:pair_profile change:slice_image change:slice_shape change:selection_min change:selection_max change:status change:r_edges change:phi_edges_deg",
    redraw,
  );

  if (typeof ResizeObserver !== "undefined") {
    resizeObserver = new ResizeObserver(() => redraw());
    resizeObserver.observe(root);
  }

  redraw();

  return () => {
    if (resizeObserver) {
      resizeObserver.disconnect();
    }
  };
}

export default { render };
