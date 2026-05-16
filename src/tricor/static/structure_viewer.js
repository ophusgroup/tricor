// Load Three.js + OrbitControls with multi-CDN fallback.
// Some Jupyter deployments block jsdelivr (CSP / VPN / offline);
// we try a short list of well-known mirrors before giving up.
let THREE = null;
let OrbitControls = null;

// Build (three_url, orbit_url) pairs to try in order.
const THREE_VERSION = "0.170.0";
const THREE_CDN_MIRRORS = [
  [
    `https://cdn.jsdelivr.net/npm/three@${THREE_VERSION}/build/three.module.js`,
    `https://cdn.jsdelivr.net/npm/three@${THREE_VERSION}/examples/jsm/controls/OrbitControls.js`,
  ],
  [
    `https://unpkg.com/three@${THREE_VERSION}/build/three.module.js`,
    `https://unpkg.com/three@${THREE_VERSION}/examples/jsm/controls/OrbitControls.js`,
  ],
  [
    `https://esm.sh/three@${THREE_VERSION}/build/three.module.js`,
    `https://esm.sh/three@${THREE_VERSION}/examples/jsm/controls/OrbitControls.js`,
  ],
  [
    `https://cdnjs.cloudflare.com/ajax/libs/three.js/r170/three.module.js`,
    // cdnjs doesn't mirror OrbitControls reliably; leave null and
    // we'll reuse whichever Orbit URL the caller succeeded with.
    null,
  ],
];

async function _fetchText(url, timeoutMs = 8000) {
  const ctrl = new AbortController();
  const tid = setTimeout(() => ctrl.abort(), timeoutMs);
  try {
    const resp = await fetch(url, { signal: ctrl.signal });
    if (!resp.ok) throw new Error(`HTTP ${resp.status}`);
    return await resp.text();
  } finally {
    clearTimeout(tid);
  }
}

let _threeBlobUrl = null;  // Kept alive across session so OrbitControls can refer back.

async function ensureThree() {
  if (THREE) return;

  const errors = [];
  let threeSrc = null;
  let chosenMirrorIdx = -1;
  for (let i = 0; i < THREE_CDN_MIRRORS.length; i++) {
    const [tUrl] = THREE_CDN_MIRRORS[i];
    try {
      threeSrc = await _fetchText(tUrl);
      chosenMirrorIdx = i;
      break;
    } catch (e) {
      errors.push(`${tUrl}: ${e.message}`);
    }
  }
  if (threeSrc === null) {
    throw new Error(
      "Failed to load Three.js source from any CDN mirror.  Tried:\n" +
        errors.join("\n") +
        "\n\nYour Jupyter environment likely can't reach a CDN. " +
        "Workarounds: (a) run Jupyter on a network with internet access, " +
        "(b) open the exported HTML files directly, or " +
        "(c) file a tricor issue asking for a locally-bundled Three.js option.",
    );
  }
  // Stable blob URL for Three.js that OrbitControls can reference.
  const threeBlob = new Blob([threeSrc], { type: "application/javascript" });
  _threeBlobUrl = URL.createObjectURL(threeBlob);
  THREE = await import(_threeBlobUrl);
  // Deliberately NOT revoking _threeBlobUrl - needs to stay live so
  // OrbitControls (loaded next) can resolve its "three" import.
  if (!globalThis.__three_module_cache) {
    globalThis.__three_module_cache = THREE;
    globalThis.__three_blob_url = _threeBlobUrl;
  }

  // Fetch OrbitControls (any mirror works); rewrite bare-"three"
  // import to point at the SAME blob URL we imported Three.js from.
  // This way the browser only hits the CDN via fetch(); the second
  // import() is a blob-URL import which CSP doesn't gate.
  let orbitSrc = null;
  const orbitErrors = [];
  const orbitCandidates = [];
  for (const [tUrl, oUrl] of THREE_CDN_MIRRORS) {
    if (oUrl) orbitCandidates.push(oUrl);
  }
  // Prefer the sibling of whichever three mirror worked, first.
  const siblingOrbit = THREE_CDN_MIRRORS[chosenMirrorIdx][1];
  if (siblingOrbit) {
    orbitCandidates.unshift(siblingOrbit);
  }
  for (const candidate of orbitCandidates) {
    try {
      orbitSrc = await _fetchText(candidate);
      break;
    } catch (e) {
      orbitErrors.push(`${candidate}: ${e.message}`);
    }
  }
  if (orbitSrc === null) {
    throw new Error(
      "Failed to load OrbitControls from any mirror.  Tried:\n" +
        orbitErrors.join("\n"),
    );
  }
  orbitSrc = orbitSrc.replace(
    /from\s+['"]three['"]/g,
    `from "${_threeBlobUrl}"`,
  );
  const orbitBlob = new Blob([orbitSrc], { type: "application/javascript" });
  const orbitBlobUrl = URL.createObjectURL(orbitBlob);
  const mod = await import(orbitBlobUrl);
  URL.revokeObjectURL(orbitBlobUrl);
  OrbitControls = mod.OrbitControls;
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

  // Two cameras kept in sync so the user can toggle between them
  // without losing framing / orbit state.
  const perspCam = new THREE.PerspectiveCamera(45, 1, 0.1, 4000);
  const orthoCam = new THREE.OrthographicCamera(-1, 1, 1, -1, 0.1, 4000);
  let camera = model.get("orthographic") ? orthoCam : perspCam;
  const renderer = new THREE.WebGLRenderer({ antialias: true });
  renderer.setPixelRatio(window.devicePixelRatio);
  canvasWrap.appendChild(renderer.domElement);

  let controls = new OrbitControls(camera, renderer.domElement);
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
  let polyMesh = null;
  let polyEdgeLine = null;

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

  function buildPolyhedra() {
    // Clear any existing polyhedra mesh/edges.
    if (polyMesh) {
      scene.remove(polyMesh);
      if (polyMesh.geometry) polyMesh.geometry.dispose();
      if (polyMesh.material) polyMesh.material.dispose();
      polyMesh = null;
    }
    if (polyEdgeLine) {
      scene.remove(polyEdgeLine);
      if (polyEdgeLine.geometry) polyEdgeLine.geometry.dispose();
      if (polyEdgeLine.material) polyEdgeLine.material.dispose();
      polyEdgeLine = null;
    }
    if (!model.get("show_polyhedra")) return;
    const nPolys = model.get("num_polyhedra") || 0;
    if (nPolys === 0) return;

    const verts = new Float32Array(model.get("polyhedra_vertex_positions"));
    const POLY_N = model.get("polyhedra_n_vertices") || 4;
    const perPoly = !!model.get("polyhedra_per_poly_topology");
    const facesShared = model.get("polyhedra_faces") || [];
    const edgesShared = model.get("polyhedra_edges") || [];
    const facesPer = model.get("polyhedra_faces_per_poly") || [];
    const edgesPer = model.get("polyhedra_edges_per_poly") || [];
    const stride = POLY_N * 3;

    // Count faces / edges so we can size arrays.
    let totalFaces = 0;
    let totalEdges = 0;
    if (perPoly) {
      for (let t = 0; t < nPolys; t++) {
        totalFaces += (facesPer[t] || []).length;
        totalEdges += (edgesPer[t] || []).length;
      }
    } else {
      totalFaces = nPolys * facesShared.length;
      totalEdges = nPolys * edgesShared.length;
    }
    if (totalFaces === 0) return;

    const facePos = new Float32Array(totalFaces * 3 * 3);
    const edgePos = new Float32Array(totalEdges * 2 * 3);
    let fOut = 0;
    let eOut = 0;
    for (let t = 0; t < nPolys; t++) {
      const base = t * stride;
      const V = [];
      for (let v = 0; v < POLY_N; v++) {
        V.push([verts[base + v*3], verts[base + v*3 + 1], verts[base + v*3 + 2]]);
      }
      const faces = perPoly ? (facesPer[t] || []) : facesShared;
      for (let f = 0; f < faces.length; f++) {
        const [a, b, cc] = faces[f];
        const o = fOut * 9;
        facePos[o+0]=V[a][0]; facePos[o+1]=V[a][1]; facePos[o+2]=V[a][2];
        facePos[o+3]=V[b][0]; facePos[o+4]=V[b][1]; facePos[o+5]=V[b][2];
        facePos[o+6]=V[cc][0]; facePos[o+7]=V[cc][1]; facePos[o+8]=V[cc][2];
        fOut++;
      }
      const edges = perPoly ? (edgesPer[t] || []) : edgesShared;
      for (let e = 0; e < edges.length; e++) {
        const [a, b] = edges[e];
        const o = eOut * 6;
        edgePos[o+0]=V[a][0]; edgePos[o+1]=V[a][1]; edgePos[o+2]=V[a][2];
        edgePos[o+3]=V[b][0]; edgePos[o+4]=V[b][1]; edgePos[o+5]=V[b][2];
        eOut++;
      }
    }

    const col = model.get("polyhedra_color") || [0.28, 0.62, 0.95];
    const opacity = model.get("polyhedra_opacity") ?? 0.4;
    const geo = new THREE.BufferGeometry();
    geo.setAttribute("position", new THREE.BufferAttribute(facePos, 3));
    geo.computeVertexNormals();
    const mat = new THREE.MeshPhongMaterial({
      color: new THREE.Color(col[0], col[1], col[2]),
      shininess: 20,
      specular: 0x888888,
      transparent: true,
      opacity: opacity,
      side: THREE.DoubleSide,
      depthWrite: false,
    });
    polyMesh = new THREE.Mesh(geo, mat);
    scene.add(polyMesh);

    const egeo = new THREE.BufferGeometry();
    egeo.setAttribute("position", new THREE.BufferAttribute(edgePos, 3));
    const emat = new THREE.LineBasicMaterial({
      color: 0x111111, transparent: true, opacity: 0.55,
    });
    polyEdgeLine = new THREE.LineSegments(egeo, emat);
    scene.add(polyEdgeLine);
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
  let sceneRadius = 1;
  function computeSceneRadius() {
    const verts = model.get("cell_vertices");
    let maxDist = 0;
    for (let i = 0; i < verts.length; i += 3) {
      const d = Math.sqrt(verts[i] ** 2 + verts[i + 1] ** 2 + verts[i + 2] ** 2);
      if (d > maxDist) maxDist = d;
    }
    return Math.max(maxDist, 1);
  }

  function frameCamera() {
    sceneRadius = computeSceneRadius();
    // Position both cameras; keep them in sync so swapping is seamless.
    const pos = new THREE.Vector3(sceneRadius * 1.8, sceneRadius * 0.6, sceneRadius * 1.5);
    perspCam.position.copy(pos);
    perspCam.lookAt(0, 0, 0);
    orthoCam.position.copy(pos);
    orthoCam.lookAt(0, 0, 0);
    controls.target.set(0, 0, 0);
    updateOrthoFrustum();
    controls.update();
  }

  function updateOrthoFrustum() {
    const w = canvasWrap.clientWidth || 600;
    const h = canvasWrap.clientHeight || 600;
    const aspect = w / h;
    // Frame so the cube bounding sphere fits in the smaller dim, padded.
    const pad = 1.15;
    const halfH = sceneRadius * pad;
    const halfW = halfH * aspect;
    orthoCam.left = -halfW; orthoCam.right = halfW;
    orthoCam.top = halfH;   orthoCam.bottom = -halfH;
    orthoCam.updateProjectionMatrix();
  }

  function swapCamera(newOrtho) {
    // Keep position / target consistent between the two cameras so the
    // toggle looks like a projection swap, not a re-frame.
    const oldPos = camera.position.clone();
    const oldTarget = controls.target.clone();
    camera = newOrtho ? orthoCam : perspCam;
    camera.position.copy(oldPos);
    camera.lookAt(oldTarget);
    // Replace OrbitControls so it drives the active camera.
    controls.dispose();
    controls = new OrbitControls(camera, renderer.domElement);
    controls.enableDamping = true;
    controls.dampingFactor = 0.12;
    controls.target.copy(oldTarget);
    if (newOrtho) updateOrthoFrustum();
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

    // Orthographic toggle
    const orthoCb = makeCheckbox("Orthographic", model.get("orthographic"), (v) => {
      model.set("orthographic", v);
      model.save_changes();
    });
    controlsPanel.appendChild(orthoCb);

    // Show bonds checkbox
    const bondCb = makeCheckbox("Show bonds", model.get("show_bonds"), (v) => {
      model.set("show_bonds", v);
      model.save_changes();
      buildBonds();
    });
    controlsPanel.appendChild(bondCb);

    // --- Polyhedra controls ---
    const polyCb = makeCheckbox("Show polyhedra", model.get("show_polyhedra"), (v) => {
      model.set("show_polyhedra", v);
      model.save_changes();
    });
    controlsPanel.appendChild(polyCb);

    const polyKind = model.get("polyhedra_kind");
    if (polyKind) {
      // bond_length slider (radial centre of the first-shell window)
      const blGroup = makeSliderGroup(
        `${polyKind.slice(0,-1)} bond length (\u00C5)`,
        0.5, 5.0, model.get("polyhedra_bond_length"), 0.01,
        (v) => {
          model.set("polyhedra_bond_length", v);
          model.save_changes();
        }
      );
      controlsPanel.appendChild(blGroup);

      // bond_length_tol slider (fractional, 0-50%)
      const blTolGroup = makeSliderGroup(
        "radial tolerance (fraction)",
        0.02, 0.40, model.get("polyhedra_bond_length_tol"), 0.01,
        (v) => {
          model.set("polyhedra_bond_length_tol", v);
          model.save_changes();
        }
      );
      controlsPanel.appendChild(blTolGroup);

      // angle_tol_deg slider (0-45 deg)
      const atGroup = makeSliderGroup(
        "angle tolerance (\u00B0)",
        1.0, 45.0, model.get("polyhedra_angle_tol_deg"), 0.5,
        (v) => {
          model.set("polyhedra_angle_tol_deg", v);
          model.save_changes();
        }
      );
      controlsPanel.appendChild(atGroup);

      // polyhedra scale slider (0.2 - 1.0)
      const scGroup = makeSliderGroup(
        "polyhedra scale",
        0.2, 1.0, model.get("polyhedra_scale"), 0.01,
        (v) => {
          model.set("polyhedra_scale", v);
          model.save_changes();
        }
      );
      controlsPanel.appendChild(scGroup);

      // opacity slider
      const opGroup = makeSliderGroup(
        "polyhedra opacity",
        0.05, 0.95, model.get("polyhedra_opacity"), 0.05,
        (v) => {
          model.set("polyhedra_opacity", v);
          model.save_changes();
          buildPolyhedra();
        }
      );
      controlsPanel.appendChild(opGroup);
    }

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
    perspCam.aspect = w / h;
    perspCam.updateProjectionMatrix();
    updateOrthoFrustum();
    renderer.setSize(w, h);
  }
  const resizeObs = new ResizeObserver(resize);
  resizeObs.observe(canvasWrap);

  // -- animation loop --
  function animate() {
    requestAnimationFrame(animate);
    controls.update();
    renderer.render(scene, camera);   // `camera` is swappable via swapCamera()
  }

  // -- model listeners --
  model.on("change:atom_visible", () => updateAtomVisibility());
  model.on("change:bond_visible change:bond_starts change:bond_ends change:bond_colors change:num_bonds change:show_bonds", () => buildBonds());
  model.on("change:show_cell", () => buildCell());
  model.on("change:atom_scale", () => updateAtomVisibility());
  model.on(
    "change:show_polyhedra change:num_polyhedra change:polyhedra_vertex_positions change:polyhedra_color change:polyhedra_opacity",
    () => buildPolyhedra()
  );
  model.on("change:orthographic", () => swapCamera(model.get("orthographic")));

  // -- init --
  buildControls();
  buildAtoms();
  buildBonds();
  buildCell();
  buildPolyhedra();
  frameCamera();
  resize();
  animate();

  } catch (e) {
    el.innerHTML = '<pre style="color:red;padding:12px;font-size:13px">Structure viewer error:\n' + e.message + '\n' + e.stack + '</pre>';
  }
}

export default { render };
