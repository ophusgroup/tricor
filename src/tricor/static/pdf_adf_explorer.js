function clamp(v, lo, hi) {
  return Math.min(hi, Math.max(lo, v));
}

function formatTick(value) {
  if (Math.abs(value) < 1e-12) return "0";
  if (Math.abs(value - Math.round(value)) < 1e-9) return value.toFixed(0);
  return value.toFixed(1);
}

function makeTicks(lo, hi, target) {
  const range = hi - lo;
  if (range <= 0) return [lo];
  const rough = range / Math.max(target - 1, 1);
  const mag = Math.pow(10, Math.floor(Math.log10(rough)));
  let step = mag;
  if (rough / mag > 5) step = 10 * mag;
  else if (rough / mag > 2) step = 5 * mag;
  else if (rough / mag > 1) step = 2 * mag;
  const ticks = [];
  let v = Math.ceil(lo / step) * step;
  while (v <= hi + step * 0.01) {
    ticks.push(v);
    v += step;
  }
  return ticks;
}

function updateSelect(select, labels, index) {
  select.replaceChildren();
  labels.forEach((label, idx) => {
    const opt = document.createElement("option");
    opt.value = String(idx);
    opt.textContent = label;
    opt.selected = idx === index;
    select.appendChild(opt);
  });
}

function drawLineChart(svg, cfg) {
  const {
    values, xValues, xLabel, yLabel, title,
    color = "#1b6370", fillColor = null,
    secondaryValues = null, secondaryColor = "#a05c2c",
    secondaryLabel = null,
    legendPrimary = null,
  } = cfg;

  const width = Math.max(Math.round(svg.clientWidth || 820), 560);
  const height = Math.max(Math.round(svg.clientHeight || 200), 140);
  const margin = { top: 28, right: 24, bottom: 40, left: 72 };
  const pw = width - margin.left - margin.right;
  const ph = height - margin.top - margin.bottom;

  const xMin = xValues[0] || 0;
  const xMax = xValues[xValues.length - 1] || 1;
  const xRange = Math.max(xMax - xMin, 1e-12);

  let yMax = Math.max(...values.filter(Number.isFinite), 1e-12) * 1.08;
  if (secondaryValues) {
    yMax = Math.max(yMax, ...secondaryValues.filter(Number.isFinite)) * 1.08;
  }
  const yMin = 0;
  const yRange = Math.max(yMax - yMin, 1e-12);

  const xS = (v) => margin.left + ((v - xMin) / xRange) * pw;
  const yS = (v) => margin.top + ph - ((v - yMin) / yRange) * ph;

  const pts = values
    .map((v, i) => `${xS(xValues[i])},${yS(v)}`)
    .join(" ");

  let secondaryPts = "";
  if (secondaryValues) {
    secondaryPts = secondaryValues
      .map((v, i) => `${xS(xValues[i])},${yS(v)}`)
      .join(" ");
  }

  const fillPath = fillColor
    ? `M${xS(xValues[0])},${yS(0)} ` +
      values.map((v, i) => `L${xS(xValues[i])},${yS(v)}`).join(" ") +
      ` L${xS(xValues[xValues.length - 1])},${yS(0)} Z`
    : "";

  svg.setAttribute("viewBox", `0 0 ${width} ${height}`);

  const xTicks = makeTicks(xMin, xMax, 8);
  const yTicks = makeTicks(yMin, yMax, 5);

  let gridLines = "";
  yTicks.forEach((v) => {
    const y = yS(v);
    gridLines += `<line class="grid-line" x1="${margin.left}" x2="${margin.left + pw}" y1="${y}" y2="${y}"/>`;
  });

  let xTickSvg = "";
  xTicks.forEach((v) => {
    const x = xS(v);
    xTickSvg += `<line class="axis-tick" x1="${x}" x2="${x}" y1="${margin.top + ph}" y2="${margin.top + ph + 5}"/>`;
    xTickSvg += `<text class="tick-label" x="${x}" y="${margin.top + ph + 18}" text-anchor="middle">${formatTick(v)}</text>`;
  });

  let yTickSvg = "";
  yTicks.forEach((v) => {
    const y = yS(v);
    yTickSvg += `<line class="axis-tick" x1="${margin.left - 5}" x2="${margin.left}" y1="${y}" y2="${y}"/>`;
    yTickSvg += `<text class="tick-label" x="${margin.left - 8}" y="${y + 4}" text-anchor="end">${formatTick(v)}</text>`;
  });

  let legendSvg = "";
  if (legendPrimary && secondaryLabel) {
    const lx = margin.left + pw - 140;
    const ly = margin.top + 14;
    legendSvg = `
      <line class="legend-swatch" x1="${lx}" x2="${lx + 18}" y1="${ly}" y2="${ly}" stroke="${color}"/>
      <text class="legend-item" x="${lx + 24}" y="${ly + 4}" fill="${color}">${legendPrimary}</text>
      <line class="legend-swatch" x1="${lx}" x2="${lx + 18}" y1="${ly + 16}" y2="${ly + 16}" stroke="${secondaryColor}" stroke-dasharray="6 4"/>
      <text class="legend-item" x="${lx + 24}" y="${ly + 20}" fill="${secondaryColor}">${secondaryLabel}</text>
    `;
  }

  svg.innerHTML = `
    <rect class="panel-bg" x="0" y="0" width="${width}" height="${height}"/>
    <text class="panel-title" x="${margin.left}" y="18">${title}</text>
    ${gridLines}
    ${fillColor ? `<path class="profile-fill" d="${fillPath}" fill="${fillColor}"/>` : ""}
    ${secondaryPts ? `<polyline class="profile-line-secondary" points="${secondaryPts}"/>` : ""}
    <polyline class="profile-line" points="${pts}" stroke="${color}"/>
    <rect class="plot-frame" x="${margin.left}" y="${margin.top}" width="${pw}" height="${ph}"/>
    <text class="axis-label" x="${margin.left + pw / 2}" y="${height - 8}" text-anchor="middle">${xLabel}</text>
    <text class="axis-label" x="18" y="${margin.top + ph / 2}" text-anchor="middle" transform="rotate(-90 18 ${margin.top + ph / 2})">${yLabel}</text>
    ${xTickSvg}
    ${yTickSvg}
    ${legendSvg}
  `;
}

function render({ model, el }) {
  const root = document.createElement("div");
  root.className = "pdfadf-widget";

  const controls = document.createElement("div");
  controls.className = "pdfadf-controls";

  const select = document.createElement("select");
  select.className = "pdfadf-select";
  controls.appendChild(select);

  const toggleLabel = document.createElement("label");
  toggleLabel.className = "pdfadf-toggle";
  const checkbox = document.createElement("input");
  checkbox.type = "checkbox";
  toggleLabel.appendChild(checkbox);
  const toggleText = document.createElement("span");
  toggleText.textContent = "Normalize";
  toggleLabel.appendChild(toggleText);
  controls.appendChild(toggleLabel);

  const status = document.createElement("div");
  status.className = "pdfadf-status";
  controls.appendChild(status);

  const g2Panel = document.createElementNS("http://www.w3.org/2000/svg", "svg");
  g2Panel.classList.add("pdfadf-panel");
  g2Panel.setAttribute("preserveAspectRatio", "xMinYMin meet");

  const adfPanel = document.createElementNS("http://www.w3.org/2000/svg", "svg");
  adfPanel.classList.add("pdfadf-panel");
  adfPanel.setAttribute("preserveAspectRatio", "xMinYMin meet");

  root.appendChild(controls);
  root.appendChild(g2Panel);
  root.appendChild(adfPanel);
  el.appendChild(root);

  let resizeObserver = null;

  function redraw() {
    const tripletLabels = model.get("triplet_labels") || [];
    const tripletIndex = model.get("triplet_index") || 0;
    const normalize = !!model.get("normalize");
    updateSelect(select, tripletLabels, tripletIndex);
    checkbox.checked = normalize;

    const r = model.get("r") || [];
    const phiDeg = model.get("phi_deg") || [];
    const g2Profile = model.get("g2_profile") || [];
    const g2Profile2 = model.get("g2_profile_2") || [];
    const adfProfile = model.get("adf_profile") || [];
    const g2Label1 = model.get("g2_label_1") || "g2";
    const g2Label2 = model.get("g2_label_2") || "";
    const tripletLabel = tripletLabels[tripletIndex] || "";

    status.textContent = model.get("status") || "";

    const showSecondary = g2Profile2.length > 0 && g2Label2 !== g2Label1;

    drawLineChart(g2Panel, {
      values: g2Profile,
      xValues: r,
      xLabel: "r (\u00C5)",
      yLabel: normalize ? "g(r)" : "counts",
      title: "Pair distribution function",
      color: "#1b6370",
      fillColor: "rgba(27, 99, 112, 0.5)",
      secondaryValues: showSecondary ? g2Profile2 : null,
      secondaryColor: "#a05c2c",
      legendPrimary: showSecondary ? g2Label1 : null,
      secondaryLabel: showSecondary ? g2Label2 : null,
    });

    drawLineChart(adfPanel, {
      values: adfProfile,
      xValues: phiDeg,
      xLabel: "angle (deg)",
      yLabel: normalize ? "P(\u03C6)" : "counts",
      title: `Angular distribution \u2014 ${tripletLabel}`,
      color: "#6b3a7d",
      fillColor: "rgba(107, 58, 125, 0.5)",
    });
  }

  select.addEventListener("change", () => {
    model.set("triplet_index", Number.parseInt(select.value, 10));
    model.save_changes();
  });

  checkbox.addEventListener("change", () => {
    model.set("normalize", checkbox.checked);
    model.save_changes();
  });

  model.on(
    "change:triplet_labels change:triplet_index change:normalize " +
    "change:r change:phi_deg change:g2_profile change:g2_profile_2 " +
    "change:adf_profile change:g2_label_1 change:g2_label_2 change:status",
    redraw,
  );

  if (typeof ResizeObserver !== "undefined") {
    resizeObserver = new ResizeObserver(() => redraw());
    resizeObserver.observe(root);
  }

  redraw();

  return () => {
    if (resizeObserver) resizeObserver.disconnect();
  };
}

export default { render };
