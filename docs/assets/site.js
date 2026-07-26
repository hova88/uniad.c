"use strict";

const $ = (selector, root = document) => root.querySelector(selector);
const $$ = (selector, root = document) => [...root.querySelectorAll(selector)];
const clamp = (value, minimum, maximum) => Math.max(minimum, Math.min(maximum, value));
const escapeHtml = value => String(value).replace(/[&<>"']/g, character => ({
  "&": "&amp;", "<": "&lt;", ">": "&gt;", '"': "&quot;", "'": "&#39;"
}[character]));

function updateProgress() {
  const length = document.documentElement.scrollHeight - innerHeight;
  $("#reading-progress").style.width = `${length > 0 ? scrollY / length * 100 : 0}%`;
}
addEventListener("scroll", updateProgress, {passive: true});
updateProgress();

const tocObserver = new IntersectionObserver(entries => {
  const visible = entries.filter(entry => entry.isIntersecting)
    .sort((a, b) => a.boundingClientRect.top - b.boundingClientRect.top)[0];
  if (!visible) return;
  $$("[data-toc]").forEach(link => {
    link.classList.toggle("active", link.getAttribute("href") === `#${visible.target.id}`);
  });
}, {rootMargin: "-20% 0px -70% 0px"});
$$("[data-section]").forEach(section => tocObserver.observe(section));

/* Figure 2 — information aperture */
const aperture = $("#interface-aperture");
const apertureTakeaways = [
  "Only decoded geometry crosses the interface. Downstream can locate the agent, but cannot revisit how that result was inferred.",
  "Heading survives, but the evidence behind it is already gone.",
  "Occlusion state survives, preserving one source of uncertainty.",
  "Neighbor context now remains available for interaction reasoning.",
  "Appearance helps the next frame recognize the same participant.",
  "A query preserves the box and the beliefs that produced it, so downstream attention can still revise the interpretation."
];
function updateAperture() {
  const level = Number(aperture.value);
  $(".interface-explainer").style.setProperty("--signal", String(level / 5));
  $("#interface-received").textContent = level
    ? `box + ${level} latent belief${level === 1 ? "" : "s"}`
    : "decoded box only";
  $("#interface-takeaway").textContent = apertureTakeaways[level];
}
aperture.addEventListener("input", updateAperture);
updateAperture();

/* System — direct dependency selection */
const architecture = {
  camera: {title: "Cameras", reads: "six synchronized BGR views", insight: "Perspective evidence has appearance but no shared metric coordinate."},
  bev: {title: "BEV encoder", reads: "camera pyramids · calibration · ego motion · previous BEV", insight: "It creates the dense coordinate system every later task can revisit."},
  track: {title: "TrackFormer", reads: "dense BEV · surviving track queries · newborn detection queries", insight: "Identity remains in a sparse query rather than a post-hoc box association."},
  map: {title: "MapFormer", reads: "the same dense BEV", insight: "Road instances become queries that can condition motion before masks are decoded."},
  motion: {title: "MotionFormer", reads: "agent queries · map queries · BEV near each endpoint", insight: "Agent, map, and goal interactions run in parallel for every motion mode."},
  occ: {title: "OccFormer", reads: "BEV · agent identity · pooled motion features", insight: "Sparse futures are written back into a dense, time-indexed risk field."},
  planner: {title: "Planner", reads: "ego query · motion · BEV · occupancy · command", insight: "The sparse intent path and dense risk path meet here."}
};
function selectArchitecture(name) {
  const activeEdges = new Set();
  $$("[data-arch-edge]").forEach(path => {
    const ends = path.dataset.archEdge.split("-");
    const active = ends.includes(name);
    path.dataset.state = active ? "active" : "dim";
    if (active) ends.forEach(end => activeEdges.add(end));
  });
  $$("[data-arch-node]").forEach(node => {
    node.dataset.state = node.dataset.archNode === name
      ? "selected"
      : activeEdges.has(node.dataset.archNode) ? "related" : "dim";
    node.setAttribute("aria-pressed", String(node.dataset.archNode === name));
  });
  const item = architecture[name];
  $("#arch-title").textContent = item.title;
  $("#arch-reads").textContent = item.reads;
  $("#arch-insight").textContent = item.insight;
}
$$("[data-arch-node]").forEach(node => node.addEventListener("click", () => selectArchitecture(node.dataset.archNode)));
selectArchitecture("planner");

/* Figure 3 — ego-motion alignment */
const alignmentControl = $("#alignment-control");
function updateAlignment() {
  const progress = Number(alignmentControl.value) / 100;
  const x = -74 * (1 - progress);
  const y = 30 * (1 - progress);
  $("#alignment-scene").style.setProperty("--align-x", `${x}px`);
  $("#alignment-scene").style.setProperty("--align-y", `${y}px`);
  $("#alignment-score").value = `${Math.round(progress * 100)}% registered`;
  if (progress > .96) {
    $("#alignment-state").textContent = "Registered";
    $("#alignment-takeaway").textContent = "Now corresponding cells refer to the same physical locations, so temporal attention can compare evidence rather than ego motion.";
  } else if (progress > .55) {
    $("#alignment-state").textContent = "Partially aligned";
    $("#alignment-takeaway").textContent = "Overlap improves, but residual pose error still makes stationary actors appear to move.";
  } else {
    $("#alignment-state").textContent = "Misregistered";
    $("#alignment-takeaway").textContent = "The same stationary actor occupies different cells because the ego vehicle moved between frames.";
  }
}
alignmentControl.addEventListener("input", updateAlignment);
updateAlignment();

/* Figure 4 — one continuous query lifecycle */
const trackStates = [
  {x: 19, visible: 1, query: 20, qOpacity: 1, score: .62, visibility: "visible", assignment: "Hungarian → #17", memory: 1, state: "Newborn", takeaway: "A detection query receives identity #17 once through Hungarian matching."},
  {x: 30, visible: 1, query: 31, qOpacity: 1, score: .71, visibility: "visible", assignment: "#17 inherited", memory: 2, state: "Matched", takeaway: "The next frame does not rematch the object; the query inherits the same ground-truth identity."},
  {x: 44, visible: 1, query: 45, qOpacity: 1, score: .68, visibility: "visible", assignment: "#17 inherited", memory: 4, state: "Carried", takeaway: "QIM updates the query while the memory bank retains four recent features."},
  {x: 62, visible: 0, query: 63, qOpacity: 1, score: .28, visibility: "occluded", assignment: "#17 retained", memory: 4, state: "Occluded, not deleted", takeaway: "The box disappears and score falls below 0.35, yet identity survives in query state and memory."},
  {x: 77, visible: 0, query: 78, qOpacity: .16, score: .19, visibility: "absent ≈ 2 s", assignment: "#17 retired", memory: 0, state: "Timed out", takeaway: "Only sustained inactivity removes the query and releases its identity state."}
];
function updateTrack() {
  const state = trackStates[Number($("#track-time").value)];
  $("#tracked-car").style.left = `${state.x}%`;
  $("#tracked-car").style.opacity = state.visible;
  $("#query-token").style.left = `${state.query}%`;
  $("#query-token").style.opacity = state.qOpacity;
  $("#track-visibility").textContent = state.visibility;
  $("#track-score").textContent = state.score.toFixed(2);
  $("#track-assignment").textContent = state.assignment;
  $$(".memory-strip i").forEach((slot, index) => { slot.style.opacity = index < state.memory ? "1" : ".12"; });
  $("#track-state").textContent = state.state;
  $("#track-takeaway").textContent = state.takeaway;
}
$("#track-time").addEventListener("input", updateTrack);
updateTrack();

/* Figure 5 — draggable map endpoint */
const mapSvg = $(".map-scene svg");
const mapEndpoint = $("#map-endpoint");
let endpoint = {x: 500, y: 105};
function roadBounds(y) {
  const bulge = 40 * Math.sin(clamp(y / 520, 0, 1) * Math.PI);
  return {left: 300 + bulge, right: 700 - bulge};
}
function updateEndpoint() {
  const bounds = roadBounds(endpoint.y);
  const offRoad = endpoint.x < bounds.left || endpoint.x > bounds.right;
  const boundaryDistance = Math.min(Math.abs(endpoint.x - bounds.left), Math.abs(endpoint.x - bounds.right));
  const crossing = endpoint.y > 170 && endpoint.y < 240;
  const divider = Math.abs(endpoint.x - 500) < 35;
  mapEndpoint.setAttribute("cx", endpoint.x);
  mapEndpoint.setAttribute("cy", endpoint.y);
  $("#endpoint-halo").setAttribute("cx", endpoint.x);
  $("#endpoint-halo").setAttribute("cy", endpoint.y);
  $("#candidate-path").setAttribute("d", `M500 455Q${500 + (endpoint.x - 500) * .25} 315 ${endpoint.x} ${endpoint.y}`);
  let state;
  if (offRoad) {
    state = {label: "Off-road", attention: "road boundary", support: "low support", copy: "The endpoint is reachable but lies outside the drivable-area query; map context should suppress this mode."};
  } else if (boundaryDistance < 48) {
    state = {label: "Near boundary", attention: "lane boundary", support: "weak support", copy: "The endpoint remains drivable but boundary proximity makes this future less plausible."};
  } else if (crossing) {
    state = {label: "At crossing", attention: "pedestrian crossing", support: "conditional support", copy: "The map query changes the semantic context: motion is possible, but crossing interaction now matters."};
  } else if (divider) {
    state = {label: "On lane", attention: "lane divider", support: "high support", copy: "The endpoint agrees with drivable geometry and lane direction."};
  } else {
    state = {label: "Drivable", attention: "drivable area", support: "good support", copy: "The endpoint stays within the road surface, while lane structure still conditions its score."};
  }
  $("#map-state").textContent = state.label;
  $("#map-attention b").textContent = state.attention;
  $("#map-feasibility").value = state.support;
  $("#map-takeaway").textContent = state.copy;
}
function mapPointer(event) {
  const box = mapSvg.getBoundingClientRect();
  endpoint.x = clamp((event.clientX - box.left) / box.width * 1000, 50, 950);
  endpoint.y = clamp((event.clientY - box.top) / box.height * 520, 40, 470);
  updateEndpoint();
}
mapEndpoint.addEventListener("pointerdown", event => {
  mapEndpoint.setPointerCapture(event.pointerId);
  mapPointer(event);
});
mapEndpoint.addEventListener("pointermove", event => {
  if (mapEndpoint.hasPointerCapture(event.pointerId)) mapPointer(event);
});
mapEndpoint.addEventListener("keydown", event => {
  const direction = {ArrowLeft: [-12, 0], ArrowRight: [12, 0], ArrowUp: [0, -12], ArrowDown: [0, 12]}[event.key];
  if (!direction) return;
  event.preventDefault();
  endpoint.x = clamp(endpoint.x + direction[0], 50, 950);
  endpoint.y = clamp(endpoint.y + direction[1], 40, 470);
  updateEndpoint();
});
updateEndpoint();

/* Figure 6 — press-and-hold context counterfactual */
function setMotionContext(enabled) {
  $(".motion-explainer").dataset.context = enabled ? "on" : "off";
  $("#motion-state").textContent = enabled ? "Context on" : "Independent forecast";
  $("#motion-takeaway").textContent = enabled
    ? "The active mode yields before the crossing agent, stays inside the lane, and resamples evidence near its endpoint."
    : "Without agent, map, and goal context, the same anchor extrapolates forward through the conflict region.";
}
const contextButton = $("#remove-context");
["pointerdown", "touchstart"].forEach(type => contextButton.addEventListener(type, event => {
  event.preventDefault();
  setMotionContext(false);
}, {passive: false}));
["pointerup", "pointercancel", "pointerleave", "touchend"].forEach(type => contextButton.addEventListener(type, () => setMotionContext(true)));
contextButton.addEventListener("keydown", event => {
  if (event.key === " " || event.key === "Enter") { event.preventDefault(); setMotionContext(false); }
});
contextButton.addEventListener("keyup", event => {
  if (event.key === " " || event.key === "Enter") setMotionContext(true);
});

/* Figure 7 — sparse agents written into dense occupancy */
const occupancyCanvas = $("#occupancy-world");
const occupancyContext = occupancyCanvas.getContext("2d");
function drawOccupancyWorld() {
  const context = occupancyContext;
  const step = Number($("#occ-time").value);
  const width = occupancyCanvas.width;
  const height = occupancyCanvas.height;
  const time = step / 2;
  context.clearRect(0, 0, width, height);
  context.fillStyle = "#12191c";
  context.fillRect(0, 0, width, height);
  context.strokeStyle = "#263135";
  context.lineWidth = 1;
  for (let x = 20; x < width; x += 40) { context.beginPath(); context.moveTo(x, 0); context.lineTo(x, height); context.stroke(); }
  for (let y = 20; y < height; y += 40) { context.beginPath(); context.moveTo(0, y); context.lineTo(width, y); context.stroke(); }
  context.strokeStyle = "#596366";
  context.lineWidth = 3;
  context.beginPath(); context.moveTo(330, height); context.bezierCurveTo(350, 360, 340, 150, 370, 0); context.stroke();
  context.beginPath(); context.moveTo(670, height); context.bezierCurveTo(650, 360, 660, 150, 630, 0); context.stroke();
  context.strokeStyle = "#e7c84a";
  context.setLineDash([12, 10]);
  context.beginPath(); context.moveTo(500, height); context.lineTo(500, 0); context.stroke();
  context.setLineDash([]);
  const agents = [
    {x: 430 + step * 24, y: 420 - step * 52, color: "#5d88e8", label: "A"},
    {x: 655 - step * 38, y: 190 + step * 26, color: "#e75832", label: "B"}
  ];
  agents.forEach((agent, agentIndex) => {
    const spread = 1 + step;
    for (let gx = -spread; gx <= spread; gx += 1) for (let gy = -spread; gy <= spread; gy += 1) {
      const distance = Math.hypot(gx, gy);
      if (distance > spread + .2) continue;
      const cell = 38;
      context.fillStyle = agent.color + (distance < 1 ? "66" : "24");
      context.strokeStyle = agent.color + "88";
      context.fillRect(Math.round(agent.x / cell) * cell + gx * cell - 18, Math.round(agent.y / cell) * cell + gy * cell - 18, 36, 36);
      context.strokeRect(Math.round(agent.x / cell) * cell + gx * cell - 18, Math.round(agent.y / cell) * cell + gy * cell - 18, 36, 36);
    }
    context.fillStyle = agent.color;
    context.fillRect(agent.x - 10, agent.y - 16, 20, 32);
    context.fillStyle = "#fff";
    context.font = "bold 11px ui-monospace, monospace";
    context.textAlign = "center";
    context.fillText(agent.label, agent.x, agent.y + 4);
    context.strokeStyle = agent.color;
    context.lineWidth = 2;
    context.beginPath();
    context.moveTo(agentIndex ? 655 : 430, agentIndex ? 190 : 420);
    context.lineTo(agent.x, agent.y);
    context.stroke();
  });
  context.fillStyle = "#f7f6f1";
  context.fillRect(490, 500, 20, 38);
  $("#occupancy-step").value = `t + ${time.toFixed(1)} s · block ${step + 1} / 5`;
}
$("#occ-time").addEventListener("input", drawOccupancyWorld);
drawOccupancyWorld();

/* Figure 8 — drag risk through the reference plan */
const plannerCanvas = $("#planner-canvas");
const plannerContext = plannerCanvas.getContext("2d");
plannerCanvas.tabIndex = 0;
let riskPoint = {x: 520, y: 250};
function plannerPaths() {
  const reference = Array.from({length: 7}, (_, index) => {
    const t = index / 6;
    return [500 - 25 * t * t, 530 - t * 450];
  });
  const optimized = reference.map(([x, y], index) => {
    if (!index) return [x, y];
    const dx = x - riskPoint.x;
    const dy = y - riskPoint.y;
    const distance = Math.max(1, Math.hypot(dx, dy));
    const influence = Math.exp(-(distance * distance) / (2 * 105 * 105));
    return [x + (dx >= 0 ? 1 : -1) * influence * 82, y];
  });
  return {reference, optimized};
}
function drawPlanner() {
  const context = plannerContext;
  context.clearRect(0, 0, plannerCanvas.width, plannerCanvas.height);
  context.fillStyle = "#12191c";
  context.fillRect(0, 0, plannerCanvas.width, plannerCanvas.height);
  context.strokeStyle = "#263135";
  for (let x = 20; x < plannerCanvas.width; x += 40) { context.beginPath(); context.moveTo(x, 0); context.lineTo(x, plannerCanvas.height); context.stroke(); }
  for (let y = 20; y < plannerCanvas.height; y += 40) { context.beginPath(); context.moveTo(0, y); context.lineTo(plannerCanvas.width, y); context.stroke(); }
  const gradient = context.createRadialGradient(riskPoint.x, riskPoint.y, 5, riskPoint.x, riskPoint.y, 105);
  gradient.addColorStop(0, "rgba(231,88,50,.78)");
  gradient.addColorStop(1, "rgba(231,88,50,0)");
  context.fillStyle = gradient;
  context.beginPath(); context.arc(riskPoint.x, riskPoint.y, 105, 0, Math.PI * 2); context.fill();
  context.strokeStyle = "#e75832";
  context.lineWidth = 2;
  context.beginPath(); context.arc(riskPoint.x, riskPoint.y, 16, 0, Math.PI * 2); context.stroke();
  const {reference, optimized} = plannerPaths();
  const strokePath = (points, color, width, dashed = false) => {
    context.strokeStyle = color; context.lineWidth = width; context.setLineDash(dashed ? [7, 8] : []);
    context.beginPath();
    context.moveTo(...points[0]);
    points.slice(1, -1).forEach((point, index) => {
      const next = points[index + 2];
      context.quadraticCurveTo(point[0], point[1], (point[0] + next[0]) / 2, (point[1] + next[1]) / 2);
    });
    context.quadraticCurveTo(...points.at(-2), ...points.at(-1));
    context.stroke();
    context.setLineDash([]);
  };
  strokePath(reference, "#7d8587", 2.5, true);
  strokePath(optimized, "#e7c84a", 4);
  optimized.slice(1).forEach(point => { context.fillStyle = "#e7c84a"; context.beginPath(); context.arc(point[0], point[1], 4.5, 0, Math.PI * 2); context.fill(); });
  context.fillStyle = "#f7f6f1"; context.fillRect(490, 520, 20, 38);
  const deviation = optimized.reduce((sum, point, index) => sum + Math.hypot(point[0] - reference[index][0], point[1] - reference[index][1]), 0) / 100;
  const clearance = Math.min(...optimized.map(point => Math.hypot(point[0] - riskPoint.x, point[1] - riskPoint.y))) / 40;
  const referenceClearance = Math.min(...reference.map(point => Math.hypot(point[0] - riskPoint.x, point[1] - riskPoint.y))) / 40;
  $("#planner-reference-cost").textContent = deviation.toFixed(2);
  $("#planner-clearance").textContent = `${clearance.toFixed(2)} grid`;
  $("#planner-interpretation").textContent = referenceClearance < 2.5
    ? "Risk is close to the reference. Nearby time-indexed waypoints move while distant intent remains unchanged."
    : "Risk is distant, so the optimized plan stays close to the learned reference.";
}
function updateRiskPointer(event) {
  const box = plannerCanvas.getBoundingClientRect();
  riskPoint.x = clamp((event.clientX - box.left) / box.width * plannerCanvas.width, 70, 930);
  riskPoint.y = clamp((event.clientY - box.top) / box.height * plannerCanvas.height, 60, 500);
  drawPlanner();
}
plannerCanvas.addEventListener("pointerdown", event => { plannerCanvas.setPointerCapture(event.pointerId); updateRiskPointer(event); });
plannerCanvas.addEventListener("pointermove", event => { if (plannerCanvas.hasPointerCapture(event.pointerId)) updateRiskPointer(event); });
plannerCanvas.addEventListener("keydown", event => {
  const direction = {ArrowLeft: [-16, 0], ArrowRight: [16, 0], ArrowUp: [0, -16], ArrowDown: [0, 16]}[event.key];
  if (!direction) return;
  event.preventDefault();
  riskPoint.x = clamp(riskPoint.x + direction[0], 70, 930);
  riskPoint.y = clamp(riskPoint.y + direction[1], 60, 500);
  drawPlanner();
});
drawPlanner();

/* Figure 9 — training stage */
const learningStages = {
  1: {
    states: {image: "frozen", bev: "active", track: "active", future: "absent", plan: "absent"},
    labels: {image: "frozen", bev: "train", track: "train", future: "absent", plan: "absent"},
    gradient: "Ltrack + Lmap stabilize the shared representation",
    state: "Establish the interface",
    takeaway: "Track and map supervision first teach BEV a stable coordinate system and sparse task interface."
  },
  2: {
    states: {image: "frozen", bev: "frozen", track: "active", future: "active", plan: "active"},
    labels: {image: "frozen", bev: "frozen", track: "train", future: "train", plan: "train"},
    gradient: "Σ Ltask updates every head while the BEV coordinate system stays fixed",
    state: "Optimize the relay",
    takeaway: "Downstream objectives now shape the query interfaces, but cannot destabilize the expensive view transformation."
  }
};
function setLearningStage(stageNumber) {
  const stage = learningStages[stageNumber];
  $$("[data-training-stage]").forEach(button => button.setAttribute("aria-pressed", String(Number(button.dataset.trainingStage) === stageNumber)));
  $$("[data-learn-module]").forEach(module => {
    const name = module.dataset.learnModule;
    module.dataset.state = stage.states[name];
    $("b", module).textContent = stage.labels[name];
  });
  $("#gradient-label").textContent = stage.gradient;
  $("#learning-state").textContent = stage.state;
  $("#learning-takeaway").textContent = stage.takeaway;
}
$$("[data-training-stage]").forEach(button => button.addEventListener("click", () => setLearningStage(Number(button.dataset.trainingStage))));
setLearningStage(1);

/* Figure 10 — evidence lives on dependency edges */
const evidenceEdges = {
  "track-motion": {test: "Motion ablation · add tracking", before: "0.815", after: "0.751", metric: "minADE · lower is better · Δ −0.064", state: "Track → Motion", takeaway: "Agent identity and history improve motion prediction beyond a motion-only decoder."},
  "map-motion": {test: "Motion ablation · add online map", before: "0.751", after: "0.736", metric: "minADE · lower is better · Δ −0.015", state: "Map → Motion", takeaway: "Road-instance queries add a smaller but measurable gain after tracking is already present."},
  "motion-occ": {test: "Joint prediction · integrate all agent context", before: "37.0", after: "39.4", metric: "far-range occupancy IoU (%) · higher is better · Δ +2.4", state: "Motion → Occupancy", takeaway: "Coupling agent motion and dense occupancy improves far-range scene prediction over occupancy alone."},
  "occ-plan": {test: "Planner ablation · add occupancy optimization", before: "1.39", after: "1.05", metric: "3 s collision rate (%) · lower is better · Δ −0.34", state: "Occupancy → Planner", takeaway: "Time-indexed occupancy lowers collision rate, while the paper reports a modest L2 trade-off."}
};
function selectEvidence(name) {
  $$("[data-evidence-edge]").forEach(button => button.setAttribute("aria-pressed", String(button.dataset.evidenceEdge === name)));
  const edge = evidenceEdges[name];
  $("#evidence-test").textContent = edge.test;
  $("#evidence-before").textContent = edge.before;
  $("#evidence-after").textContent = edge.after;
  $("#evidence-metric").textContent = edge.metric;
  $("#evidence-state").textContent = edge.state;
  $("#evidence-takeaway").textContent = edge.takeaway;
}
$$("[data-evidence-edge]").forEach(button => button.addEventListener("click", () => selectEvidence(button.dataset.evidenceEdge)));
selectEvidence("track-motion");

/* Runtime contract — aligned comparison wipe */
function updateContractWipe() {
  $(".contract-explainer").style.setProperty("--wipe", `${$("#contract-wipe").value}%`);
}
$("#contract-wipe").addEventListener("input", updateContractWipe);
updateContractWipe();

/* Correctness ladder */
const proofRungs = [
  {status: "Proved", title: "Container safety", copy: "Magic, version, shape, finite-value, checksum, and size checks reject malformed inputs before inference.", next: "Next requirement", takeaway: "Operator fixtures must then show that individual C kernels agree with an independent numerical oracle."},
  {status: "Partial production coverage", title: "Released-weight prefix", copy: "Production CUDA reaches all six first-frame BEVFormer and TrackFormer decoder/reference-refinement layers plus final track heads and activation filtering.", next: "Next requirement", takeaway: "Verify continuous-frame previous-BEV alignment, implement track-ID/query/memory lifecycle, then finish the remaining task heads, transactional state update, and decode."},
  {status: "Current ceiling", title: "Synthetic graph", copy: "Two-frame carry, reset behavior, and final JSON agree with an independent PyTorch oracle.", next: "What remains", takeaway: "A production claim next requires checkpoint export, name and shape mapping, and learned-operator parity."},
  {status: "Mapping only", title: "Checkpoint equivalence", copy: "All 2,459 released tensors are in UAW2 and 2,451 keys have an observed consumer, but the complete learned CUDA graph does not exist yet.", next: "After equivalence", takeaway: "Even full tensor parity would not establish task accuracy until official data and evaluation are run."},
  {status: "Unclaimed", title: "Task accuracy", copy: "nuScenes data, official preprocessing, temporal protocol, and task evaluators would be required to reproduce reported metrics.", next: "Claim boundary", takeaway: "The repository deliberately makes no production-accuracy claim."}
];
function selectProof(index) {
  $$("[data-proof-rung]").forEach(button => button.setAttribute("aria-pressed", String(Number(button.dataset.proofRung) === index)));
  const rung = proofRungs[index];
  $("#proof-status").textContent = rung.status;
  $("#proof-title").textContent = rung.title;
  $("#proof-copy").textContent = rung.copy;
  $("#proof-next").textContent = rung.next;
  $("#proof-takeaway").textContent = rung.takeaway;
}
$$("[data-proof-rung]").forEach(button => button.addEventListener("click", () => selectProof(Number(button.dataset.proofRung))));
selectProof(2);

/* Figure 11 — exact canonical result, with advanced controls disclosed */
const canonicalCanvas = $("#canonical-canvas");
const canonicalContext = canonicalCanvas.getContext("2d");
canonicalCanvas.tabIndex = 0;
const canonicalBounds = {xMin: -4.5, xMax: 4.5, yMin: -4.5, yMax: 5};
const canonicalPadding = {left: 54, right: 24, top: 28, bottom: 48};
const canonicalLayers = new Set(["map", "occupancy", "tracks", "motion", "plan"]);
let canonicalResult = null;
let canonicalMode = 0;
let canonicalStep = 3;
let canonicalSelectedTrack = null;
let canonicalHitTargets = [];

function canonicalPoint([x, y]) {
  const plotWidth = canonicalCanvas.width - canonicalPadding.left - canonicalPadding.right;
  const plotHeight = canonicalCanvas.height - canonicalPadding.top - canonicalPadding.bottom;
  return [
    canonicalPadding.left + (x - canonicalBounds.xMin) / (canonicalBounds.xMax - canonicalBounds.xMin) * plotWidth,
    canonicalPadding.top + (canonicalBounds.yMax - y) / (canonicalBounds.yMax - canonicalBounds.yMin) * plotHeight
  ];
}
function canonicalMapPoints(map) {
  return Array.isArray(map.points) ? map.points : [[map.x0, map.y0], [map.x1, map.y1]];
}
function drawCanonicalGrid(context) {
  context.fillStyle = "#111719";
  context.fillRect(0, 0, canonicalCanvas.width, canonicalCanvas.height);
  context.font = "10px ui-monospace, monospace";
  context.textAlign = "center";
  context.textBaseline = "middle";
  for (let x = -4; x <= 4; x += 1) {
    const [px] = canonicalPoint([x, 0]);
    context.strokeStyle = x === 0 ? "#677276" : "#273135";
    context.lineWidth = x === 0 ? 1.5 : 1;
    context.beginPath(); context.moveTo(px, canonicalPadding.top); context.lineTo(px, canonicalCanvas.height - canonicalPadding.bottom); context.stroke();
    context.fillStyle = "#718084"; context.fillText(String(x), px, canonicalCanvas.height - 24);
  }
  for (let y = -4; y <= 5; y += 1) {
    const [, py] = canonicalPoint([0, y]);
    context.strokeStyle = y === 0 ? "#677276" : "#273135";
    context.lineWidth = y === 0 ? 1.5 : 1;
    context.beginPath(); context.moveTo(canonicalPadding.left, py); context.lineTo(canonicalCanvas.width - canonicalPadding.right, py); context.stroke();
    context.textAlign = "right"; context.fillStyle = "#718084"; context.fillText(String(y), 43, py); context.textAlign = "center";
  }
  context.fillStyle = "#93a0a3";
  context.fillText("x / right →", canonicalCanvas.width - 78, canonicalCanvas.height - 24);
  context.save(); context.translate(17, 72); context.rotate(-Math.PI / 2); context.fillText("y / forward →", 0, 0); context.restore();
}
function drawCanonicalOccupancy(context, result) {
  const horizons = result.occupancy || [];
  if (!horizons.length) return;
  const horizonIndex = Math.min(canonicalStep, horizons.length - 1);
  const cells = horizons[horizonIndex];
  const [x0, y0] = canonicalPoint([-4, -3]);
  const [x1, y1] = canonicalPoint([-3, -4]);
  cells.forEach((occupied, index) => {
    if (!occupied) return;
    const [px, py] = canonicalPoint([index % 8 - 3.5, Math.floor(index / 8) - 3.5]);
    context.fillStyle = "rgba(231,88,50,.20)"; context.strokeStyle = "rgba(231,88,50,.55)";
    context.fillRect(px - (x1 - x0) / 2, py - (y0 - y1) / 2, x1 - x0, y0 - y1);
    context.strokeRect(px - (x1 - x0) / 2, py - (y0 - y1) / 2, x1 - x0, y0 - y1);
  });
}
function drawCanonical(result) {
  const context = canonicalContext;
  canonicalHitTargets = [];
  context.clearRect(0, 0, canonicalCanvas.width, canonicalCanvas.height);
  drawCanonicalGrid(context);
  if (canonicalLayers.has("occupancy")) drawCanonicalOccupancy(context, result);
  if (canonicalLayers.has("map")) {
    result.map.forEach((map, index) => {
      const points = canonicalMapPoints(map);
      context.strokeStyle = index === 2 ? "#aeb8ba" : "#667376"; context.setLineDash(index === 2 ? [] : [8, 7]); context.lineWidth = index === 2 ? 2 : 1.5;
      context.beginPath(); points.forEach((value, pointIndex) => pointIndex ? context.lineTo(...canonicalPoint(value)) : context.moveTo(...canonicalPoint(value))); context.stroke();
    });
    context.setLineDash([]);
  }
  if (canonicalLayers.has("motion")) {
    result.motion.forEach(motion => {
      const track = result.tracks.find(candidate => candidate.id === motion.track_id);
      const mode = motion.modes[canonicalMode];
      if (!track || !mode) return;
      const selected = canonicalSelectedTrack === motion.track_id;
      const trajectory = [[track.x, track.y], ...mode.trajectory.slice(0, canonicalStep + 1)];
      context.strokeStyle = selected ? "#8fb0ff" : "rgba(93,136,232,.58)"; context.lineWidth = selected ? 4 : 2;
      context.beginPath(); trajectory.forEach((value, index) => index ? context.lineTo(...canonicalPoint(value)) : context.moveTo(...canonicalPoint(value))); context.stroke();
    });
  }
  if (canonicalLayers.has("plan")) {
    const plan = [[0, 0], ...result.ego_plan];
    context.strokeStyle = "#e7c84a"; context.lineWidth = 4; context.beginPath();
    plan.forEach((value, index) => index ? context.lineTo(...canonicalPoint(value)) : context.moveTo(...canonicalPoint(value))); context.stroke();
    plan.slice(1).forEach(value => { const [x, y] = canonicalPoint(value); context.fillStyle = "#e7c84a"; context.beginPath(); context.arc(x, y, 4.5, 0, Math.PI * 2); context.fill(); });
  }
  if (canonicalLayers.has("tracks")) {
    result.tracks.forEach(track => {
      const [x, y] = canonicalPoint([track.x, track.y]);
      const selected = canonicalSelectedTrack === track.id;
      context.fillStyle = selected ? "#ff9a78" : "#e75832"; context.strokeStyle = selected ? "#fff1eb" : "#e75832"; context.lineWidth = selected ? 3 : 1;
      context.fillRect(x - 7, y - 12, 14, 24); context.strokeRect(x - 10, y - 15, 20, 30);
      context.fillStyle = selected ? "#ffffff" : "#aeb8ba"; context.font = `${selected ? "bold " : ""}10px ui-monospace, monospace`; context.textAlign = "left"; context.fillText(`#${track.id}`, x + 13, y + 3);
      canonicalHitTargets.push({id: track.id, x, y, radius: 20});
    });
  }
  const [egoX, egoY] = canonicalPoint([0, 0]);
  context.fillStyle = "#f7f6f1"; context.fillRect(egoX - 8, egoY - 15, 16, 30);
  context.fillStyle = "#111719"; context.font = "bold 9px ui-monospace, monospace"; context.textAlign = "center"; context.fillText("E", egoX, egoY);
}
function renderCanonicalInspector() {
  if (!canonicalResult) return;
  const track = canonicalResult.tracks.find(candidate => candidate.id === canonicalSelectedTrack);
  if (!track) {
    $("#inspector-kind").textContent = "Record";
    $("#inspector-title").textContent = canonicalResult.scene;
    $("#inspector-values").innerHTML = `<div><dt>schema</dt><dd>${escapeHtml(canonicalResult.schema)}</dd></div><div><dt>frame</dt><dd>${escapeHtml(canonicalResult.coordinate_frame)}</dd></div><div><dt>command</dt><dd>${escapeHtml(canonicalResult.command)}</dd></div><div><dt>occupancy</dt><dd>${canonicalResult.occupancy.length} × 8 × 8</dd></div>`;
    $("#inspector-note").textContent = "Select an orange track. Advanced layer and time controls are available below the scene.";
    return;
  }
  const motion = canonicalResult.motion.find(candidate => candidate.track_id === track.id);
  const mode = motion && motion.modes[canonicalMode];
  const endpointValue = mode && mode.trajectory[Math.min(canonicalStep, mode.trajectory.length - 1)];
  $("#inspector-kind").textContent = "Selected track";
  $("#inspector-title").textContent = `query cell #${track.id}`;
  $("#inspector-values").innerHTML = `<div><dt>position</dt><dd>(${track.x.toFixed(2)}, ${track.y.toFixed(2)}) m</dd></div><div><dt>track score</dt><dd>${track.score.toFixed(6)}</dd></div><div><dt>mode</dt><dd>${canonicalMode} · p=${mode ? mode.score.toFixed(6) : "n/a"}</dd></div><div><dt>endpoint t${canonicalStep + 1}</dt><dd>${endpointValue ? `(${endpointValue[0].toFixed(2)}, ${endpointValue[1].toFixed(2)}) m` : "n/a"}</dd></div>`;
  $("#inspector-note").textContent = "The ID is the stable top-k BEV-cell index in the tiny runtime—not a production TrackFormer identity.";
}
function refreshCanonical() {
  if (!canonicalResult) return;
  $("#canonical-time-label").value = `${canonicalStep + 1} / 4 · occ h${Math.min(canonicalStep, canonicalResult.occupancy.length - 1)}`;
  drawCanonical(canonicalResult);
  renderCanonicalInspector();
}
$("#canonical-time").addEventListener("input", event => { canonicalStep = Number(event.target.value); refreshCanonical(); });
$$("[data-canonical-layer]").forEach(input => input.addEventListener("change", () => {
  input.checked ? canonicalLayers.add(input.dataset.canonicalLayer) : canonicalLayers.delete(input.dataset.canonicalLayer);
  refreshCanonical();
}));
$$("[data-canonical-mode]").forEach(button => button.addEventListener("click", () => {
  canonicalMode = Number(button.dataset.canonicalMode);
  $$("[data-canonical-mode]").forEach(item => item.setAttribute("aria-pressed", String(item === button)));
  refreshCanonical();
}));
canonicalCanvas.addEventListener("pointermove", event => {
  const bounds = canonicalCanvas.getBoundingClientRect();
  const x = (event.clientX - bounds.left) * canonicalCanvas.width / bounds.width;
  const y = (event.clientY - bounds.top) * canonicalCanvas.height / bounds.height;
  canonicalCanvas.style.cursor = canonicalHitTargets.some(target => Math.hypot(x - target.x, y - target.y) <= target.radius) ? "pointer" : "crosshair";
});
canonicalCanvas.addEventListener("click", event => {
  const bounds = canonicalCanvas.getBoundingClientRect();
  const x = (event.clientX - bounds.left) * canonicalCanvas.width / bounds.width;
  const y = (event.clientY - bounds.top) * canonicalCanvas.height / bounds.height;
  const target = canonicalHitTargets.find(candidate => Math.hypot(x - candidate.x, y - candidate.y) <= candidate.radius);
  canonicalSelectedTrack = target ? target.id : null;
  refreshCanonical();
});
canonicalCanvas.addEventListener("keydown", event => {
  if (!canonicalResult || !["ArrowLeft", "ArrowRight"].includes(event.key)) return;
  event.preventDefault();
  const ids = canonicalResult.tracks.map(track => track.id);
  const current = ids.indexOf(canonicalSelectedTrack);
  const direction = event.key === "ArrowRight" ? 1 : -1;
  canonicalSelectedTrack = ids[(current + direction + ids.length) % ids.length];
  refreshCanonical();
});
fetch("assets/demo-result.json").then(response => {
  if (!response.ok) throw new Error("result unavailable");
  return response.json();
}).then(result => {
  canonicalResult = result;
  $("#canonical-scene").textContent = `${result.scene} · ${result.profile}`;
  $("#frame-readout").textContent = result.frame_index;
  $("#tracks-readout").textContent = result.tracks.length;
  $("#collision-readout").textContent = Number(result.collision_score).toFixed(3);
  refreshCanonical();
}).catch(() => {
  $("#canonical-scene").textContent = "demo-result.json unavailable";
  $("#collision-readout").textContent = "n/a";
});
