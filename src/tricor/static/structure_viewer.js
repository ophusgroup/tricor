// Load Three.js + OrbitControls via import map injection.
// OrbitControls ESM does `import { ... } from "three"` which
// requires the browser to resolve the bare specifier "three".
let THREE = null;
let OrbitControls = null;

async function ensureThree() {
  if (THREE) return;

  // Inject import map so "three" resolves to the CDN URL
  if (!document.querySelector('script[type="importmap"][data-tricor]')) {
    const map = document.createElement("script");
    map.type = "importmap";
    map.setAttribute("data-tricor", "1");
    map.textContent = JSON.stringify({
      imports: {
        "three": "https://cdn.jsdelivr.net/npm/three@0.170.0/build/three.module.js",
        "three/addons/": "https://cdn.jsdelivr.net/npm/three@0.170.0/examples/jsm/"
      }
    });
    document.head.appendChild(map);
    // Import maps must be added before any module imports, but
    // in Jupyter the page is already loaded. Fall back to manual
    // patching if the import map doesn't take effect.
  }

  // Try ESM import first
  try {
    THREE = await import("https://cdn.jsdelivr.net/npm/three@0.170.0/build/three.module.js");
    // Patch globalThis so OrbitControls can find "three"
    const threeKeys = Object.keys(THREE);
    if (!globalThis.__three_module_cache) {
      globalThis.__three_module_cache = THREE;
    }
  } catch (e) {
    throw new Error("Failed to load Three.js: " + e.message);
  }

  // OrbitControls needs "three" as bare specifier. Fetch source
  // and rewrite the import to use the full URL.
  try {
    const resp = await fetch("https://cdn.jsdelivr.net/npm/three@0.170.0/examples/jsm/controls/OrbitControls.js");
    let src = await resp.text();
    // Rewrite bare "three" import to absolute URL
    src = src.replace(
      /from\s+['"]three['"]/g,
      'from "https://cdn.jsdelivr.net/npm/three@0.170.0/build/three.module.js"'
    );
    const blob = new Blob([src], { type: "application/javascript" });
    const url = URL.createObjectURL(blob);
    const mod = await import(url);
    URL.revokeObjectURL(url);
    OrbitControls = mod.OrbitControls;
  } catch (e) {
    throw new Error("Failed to load OrbitControls: " + e.message);
  }
}

async function render({ model, el }) {
  try {
  await ensureThree();

  // -- container --
  const root = document.createElement("div");
  root.classList.add("tricor-structure-widget");
  el.appendChild(root);

  const canvasWrap = document.createElement("div");
  canvasWrap.classList.add("tricor-structure-canvas");
  root.appendChild(canvasWrap);

  const controlsPanel = document.createElement("div");
  controlsPanel.classList.add("tricor-structure-controls");
  root.appendChild(controlsPanel);

  // -- Three.js setup --
  const scene = new THREE.Scene();
  scene.background = new THREE.Color(model.get("background_color"));

  const camera = new THREE.PerspectiveCamera(45, 1, 0.1, 1000);
  const renderer = new THREE.WebGLRenderer({ antialias: true });
  renderer.setPixelRatio(window.devicePixelRatio);
  canvasWrap.appendChild(renderer.domElement);

  const controls = new OrbitControls(camera, renderer.domElement);
  controls.enableDamping = true;
  controls.dampingFactor = 0.12;

  // Lights
  const ambient = new THREE.AmbientLight(0xffffff, 0.6);
  scene.add(ambient);
  const dirLight = new THREE.DirectionalLight(0xffffff, 0.8);
  dirLight.position.set(1, 1, 1);
  scene.add(dirLight);

  // Shared geometries
  const sphereGeo = new THREE.SphereGeometry(1, 16, 12);
  const cylinderGeo = new THREE.CylinderGeometry(1, 1, 1, 8, 1);

  // -- meshes --
  let atomMesh = null;
  let bondMesh = null;
  let cellLine = null;

  const dummy = new THREE.Object3D();
  const color = new THREE.Color();

  function buildAtoms() {
    if (atomMesh) { scene.remove(atomMesh); atomMesh.dispose(); }
    const n = model.get("num_atoms");
    if (n === 0) return;
    const pos = model.get("atom_positions");
    const cols = model.get("atom_colors");
    const radii = model.get("atom_radii");
    const vis = model.get("atom_visible");
    const scale = model.get("atom_scale");

    const mat = new THREE.MeshStandardMaterial({ metalness: 0.1, roughness: 0.6 });
    atomMesh = new THREE.InstancedMesh(sphereGeo, mat, n);

    for (let i = 0; i < n; i++) {
      const r = vis[i] ? radii[i] * scale : 0;
      dummy.position.set(pos[i * 3], pos[i * 3 + 1], pos[i * 3 + 2]);
      dummy.scale.set(r, r, r);
      dummy.updateMatrix();
      atomMesh.setMatrixAt(i, dummy.matrix);
      color.setRGB(cols[i * 3], cols[i * 3 + 1], cols[i * 3 + 2]);
      atomMesh.setColorAt(i, color);
    }
    atomMesh.instanceMatrix.needsUpdate = true;
    atomMesh.instanceColor.needsUpdate = true;
    scene.add(atomMesh);
  }

  function updateAtomVisibility() {
    if (!atomMesh) return;
    const n = model.get("num_atoms");
    const vis = model.get("atom_visible");
    const radii = model.get("atom_radii");
    const pos = model.get("atom_positions");
    const scale = model.get("atom_scale");
    for (let i = 0; i < n; i++) {
      const r = vis[i] ? radii[i] * scale : 0;
      dummy.position.set(pos[i * 3], pos[i * 3 + 1], pos[i * 3 + 2]);
      dummy.scale.set(r, r, r);
      dummy.updateMatrix();
      atomMesh.setMatrixAt(i, dummy.matrix);
    }
    atomMesh.instanceMatrix.needsUpdate = true;
  }

  function buildBonds() {
    if (bondMesh) { scene.remove(bondMesh); bondMesh.dispose(); }
    const n = model.get("num_bonds");
    if (n === 0 || !model.get("show_bonds")) return;
    const starts = model.get("bond_starts");
    const ends = model.get("bond_ends");
    const cols = model.get("bond_colors");
    const vis = model.get("bond_visible");
    const br = model.get("bond_radius");

    const mat = new THREE.MeshStandardMaterial({ metalness: 0.05, roughness: 0.7 });
    bondMesh = new THREE.InstancedMesh(cylinderGeo, mat, n);

    const up = new THREE.Vector3(0, 1, 0);
    const start = new THREE.Vector3();
    const end = new THREE.Vector3();
    const mid = new THREE.Vector3();
    const dir = new THREE.Vector3();
    const quat = new THREE.Quaternion();

    for (let i = 0; i < n; i++) {
      if (!vis[i]) {
        dummy.scale.set(0, 0, 0);
        dummy.updateMatrix();
        bondMesh.setMatrixAt(i, dummy.matrix);
        color.setRGB(cols[i * 3], cols[i * 3 + 1], cols[i * 3 + 2]);
        bondMesh.setColorAt(i, color);
        continue;
      }
      start.set(starts[i * 3], starts[i * 3 + 1], starts[i * 3 + 2]);
      end.set(ends[i * 3], ends[i * 3 + 1], ends[i * 3 + 2]);
      mid.addVectors(start, end).multiplyScalar(0.5);
      dir.subVectors(end, start);
      const len = dir.length();
      dir.normalize();

      dummy.position.copy(mid);
      quat.setFromUnitVectors(up, dir);
      dummy.quaternion.copy(quat);
      dummy.scale.set(br, len, br);
      dummy.updateMatrix();
      bondMesh.setMatrixAt(i, dummy.matrix);

      color.setRGB(cols[i * 3], cols[i * 3 + 1], cols[i * 3 + 2]);
      bondMesh.setColorAt(i, color);
    }
    bondMesh.instanceMatrix.needsUpdate = true;
    bondMesh.instanceColor.needsUpdate = true;
    scene.add(bondMesh);
  }

  function buildCell() {
    if (cellLine) { scene.remove(cellLine); cellLine.dispose(); }
    if (!model.get("show_cell")) return;
    const verts = model.get("cell_vertices");
    const edges = model.get("cell_edges");
    const geo = new THREE.BufferGeometry();
    const positions = [];
    for (let i = 0; i < edges.length; i += 2) {
      const a = edges[i], b = edges[i + 1];
      positions.push(verts[a * 3], verts[a * 3 + 1], verts[a * 3 + 2]);
      positions.push(verts[b * 3], verts[b * 3 + 1], verts[b * 3 + 2]);
    }
    geo.setAttribute("position", new THREE.Float32BufferAttribute(positions, 3));
    const mat = new THREE.LineBasicMaterial({ color: 0x333333, linewidth: 1, transparent: true, opacity: 0.5 });
    cellLine = new THREE.LineSegments(geo, mat);
    scene.add(cellLine);
  }

  // -- camera framing --
  function frameCamera() {
    const verts = model.get("cell_vertices");
    let maxDist = 0;
    for (let i = 0; i < verts.length; i += 3) {
      const d = Math.sqrt(verts[i] ** 2 + verts[i + 1] ** 2 + verts[i + 2] ** 2);
      if (d > maxDist) maxDist = d;
    }
    camera.position.set(maxDist * 1.8, maxDist * 0.6, maxDist * 1.5);
    camera.lookAt(0, 0, 0);
    controls.target.set(0, 0, 0);
    controls.update();
  }

  // -- controls panel --
  function buildControls() {
    controlsPanel.innerHTML = "";

    // Atom size slider
    const atomGroup = makeSliderGroup("Atom size", 0.05, 1.5, model.get("atom_scale"), 0.01, (v) => {
      model.set("atom_scale", v);
      model.save_changes();
      updateAtomVisibility();
    });
    controlsPanel.appendChild(atomGroup);

    // Bond radius slider
    const bondRGroup = makeSliderGroup("Bond radius", 0.01, 0.3, model.get("bond_radius"), 0.01, (v) => {
      model.set("bond_radius", v);
      model.save_changes();
      buildBonds();
    });
    controlsPanel.appendChild(bondRGroup);

    // Bond cutoff slider
    const cutoffGroup = makeSliderGroup(
      "Bond cutoff (\u00C5)", 0.5, model.get("bond_cutoff_max"),
      model.get("bond_cutoff"), 0.05, (v) => {
        model.set("bond_cutoff", v);
        model.save_changes();
      }
    );
    controlsPanel.appendChild(cutoffGroup);

    // Bond pair checkboxes
    const pairLabels = model.get("bond_pair_labels");
    const pairVis = model.get("bond_pair_visible");
    if (pairLabels.length > 0) {
      const pairDiv = document.createElement("div");
      pairDiv.classList.add("tricor-control-group");
      const pairTitle = document.createElement("div");
      pairTitle.classList.add("tricor-control-label");
      pairTitle.textContent = "Bond types";
      pairDiv.appendChild(pairTitle);
      pairLabels.forEach((label, idx) => {
        const row = document.createElement("label");
        row.classList.add("tricor-checkbox-row");
        const cb = document.createElement("input");
        cb.type = "checkbox";
        cb.checked = pairVis[idx];
        cb.addEventListener("change", () => {
          const vis = [...model.get("bond_pair_visible")];
          vis[idx] = cb.checked;
          model.set("bond_pair_visible", vis);
          model.save_changes();
        });
        row.appendChild(cb);
        row.appendChild(document.createTextNode(" " + label));
        pairDiv.appendChild(row);
      });
      controlsPanel.appendChild(pairDiv);
    }

    // Show cell checkbox
    const cellCb = makeCheckbox("Show cell", model.get("show_cell"), (v) => {
      model.set("show_cell", v);
      model.save_changes();
      buildCell();
    });
    controlsPanel.appendChild(cellCb);

    // Show bonds checkbox
    const bondCb = makeCheckbox("Show bonds", model.get("show_bonds"), (v) => {
      model.set("show_bonds", v);
      model.save_changes();
      buildBonds();
    });
    controlsPanel.appendChild(bondCb);

    // Slab sliders
    const slabNames = [
      ["x", "slab_x_min", "slab_x_max"],
      ["y", "slab_y_min", "slab_y_max"],
      ["z", "slab_z_min", "slab_z_max"],
    ];
    slabNames.forEach(([axis, minKey, maxKey]) => {
      const group = makeDualSliderGroup(`Slab ${axis}`, 0.0, 1.0,
        model.get(minKey), model.get(maxKey), 0.01,
        (lo, hi) => {
          model.set(minKey, lo);
          model.set(maxKey, hi);
          model.save_changes();
        }
      );
      controlsPanel.appendChild(group);
    });
  }

  // -- UI helpers --
  function makeSliderGroup(label, min, max, value, step, onChange) {
    const group = document.createElement("div");
    group.classList.add("tricor-control-group");
    const lbl = document.createElement("div");
    lbl.classList.add("tricor-control-label");
    const valSpan = document.createElement("span");
    valSpan.textContent = value.toFixed(2);
    lbl.textContent = label + " ";
    lbl.appendChild(valSpan);
    group.appendChild(lbl);
    const slider = document.createElement("input");
    slider.type = "range";
    slider.min = min;
    slider.max = max;
    slider.step = step;
    slider.value = value;
    slider.classList.add("tricor-slider");
    slider.addEventListener("input", () => {
      const v = parseFloat(slider.value);
      valSpan.textContent = v.toFixed(2);
      onChange(v);
    });
    group.appendChild(slider);
    return group;
  }

  function makeDualSliderGroup(label, min, max, valueLo, valueHi, step, onChange) {
    const group = document.createElement("div");
    group.classList.add("tricor-control-group");
    const lbl = document.createElement("div");
    lbl.classList.add("tricor-control-label");
    lbl.textContent = label;
    group.appendChild(lbl);

    const loSlider = document.createElement("input");
    loSlider.type = "range"; loSlider.min = min; loSlider.max = max;
    loSlider.step = step; loSlider.value = valueLo;
    loSlider.classList.add("tricor-slider", "tricor-slider-dual");

    const hiSlider = document.createElement("input");
    hiSlider.type = "range"; hiSlider.min = min; hiSlider.max = max;
    hiSlider.step = step; hiSlider.value = valueHi;
    hiSlider.classList.add("tricor-slider", "tricor-slider-dual");

    const valText = document.createElement("span");
    valText.classList.add("tricor-dual-value");
    valText.textContent = `${valueLo.toFixed(2)} - ${valueHi.toFixed(2)}`;

    function update() {
      let lo = parseFloat(loSlider.value);
      let hi = parseFloat(hiSlider.value);
      if (lo > hi) { const tmp = lo; lo = hi; hi = tmp; }
      valText.textContent = `${lo.toFixed(2)} - ${hi.toFixed(2)}`;
      onChange(lo, hi);
    }
    loSlider.addEventListener("input", update);
    hiSlider.addEventListener("input", update);
    group.appendChild(loSlider);
    group.appendChild(hiSlider);
    group.appendChild(valText);
    return group;
  }

  function makeCheckbox(label, checked, onChange) {
    const group = document.createElement("div");
    group.classList.add("tricor-control-group");
    const lbl = document.createElement("label");
    lbl.classList.add("tricor-checkbox-row");
    const cb = document.createElement("input");
    cb.type = "checkbox";
    cb.checked = checked;
    cb.addEventListener("change", () => onChange(cb.checked));
    lbl.appendChild(cb);
    lbl.appendChild(document.createTextNode(" " + label));
    group.appendChild(lbl);
    return group;
  }

  // -- resize --
  function resize() {
    const w = canvasWrap.clientWidth || 600;
    const h = canvasWrap.clientHeight || 600;
    camera.aspect = w / h;
    camera.updateProjectionMatrix();
    renderer.setSize(w, h);
  }
  const resizeObs = new ResizeObserver(resize);
  resizeObs.observe(canvasWrap);

  // -- animation loop --
  function animate() {
    requestAnimationFrame(animate);
    controls.update();
    renderer.render(scene, camera);
  }

  // -- model listeners --
  model.on("change:atom_visible", () => updateAtomVisibility());
  model.on("change:bond_visible change:bond_starts change:bond_ends change:bond_colors change:num_bonds change:show_bonds", () => buildBonds());
  model.on("change:show_cell", () => buildCell());
  model.on("change:atom_scale", () => updateAtomVisibility());

  // -- init --
  buildControls();
  buildAtoms();
  buildBonds();
  buildCell();
  frameCamera();
  resize();
  animate();

  } catch (e) {
    el.innerHTML = '<pre style="color:red;padding:12px;font-size:13px">Structure viewer error:\n' + e.message + '\n' + e.stack + '</pre>';
  }
}

export default { render };
