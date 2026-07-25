"use strict";

const $ = (selector, root = document) => root.querySelector(selector);
const $$ = (selector, root = document) => [...root.querySelectorAll(selector)];
const escapeHtml = value => String(value).replace(/[&<>"']/g, char => ({
  "&": "&amp;", "<": "&lt;", ">": "&gt;", '"': "&quot;", "'": "&#39;"
}[char]));

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

const systemSteps = [
  {
    shape: "6 × 4 feature levels",
    copy: "ResNet-101 and FPN turn six BGR views into four 256-channel feature scales.",
    nodes: [["6 cameras", "perspective"], ["R101 + FPN", "4 scales"], ["Fcam", "view features"]],
    reads: "six synchronized camera images",
    preserves: "appearance at four spatial scales",
    missing: "metric position and temporal identity",
    consumer: "BEVFormer spatial cross-attention",
    prompt: "Why not plan from image features?",
    answer: "A pixel displacement has no fixed metric meaning across cameras or depth. The next stage must establish a common coordinate frame."
  },
  {
    shape: "200 × 200 × 256",
    copy: "The BEV encoder alternates temporal self-attention and spatial cross-attention to form a shared world state.",
    nodes: [["Fcam + Bₜ₋₁", "aligned"], ["BEV encoder ×6", "temporal + spatial"], ["Bₜ", "dense world"]],
    reads: "multi-scale views, calibration, ego motion, previous BEV",
    preserves: "metric geometry and cross-frame context",
    missing: "explicit agent identity and road instances",
    consumer: "TrackFormer, MapFormer, OccFormer, Planner",
    prompt: "Why is the previous BEV aligned before attention?",
    answer: "Without rotation and translation, ego motion masquerades as world motion. Attention would compare different physical locations."
  },
  {
    shape: "Na × 256 · 300 × 256",
    copy: "TrackFormer emits a dynamic set of agent queries; MapFormer emits 300 road-instance queries.",
    nodes: [["Bₜ", "shared"], ["Track / Map", "parallel"], ["Qᴀ + Qᴍ", "sparse entities"]],
    reads: "the same dense BEV world state",
    preserves: "agent identity, road semantics, and query features",
    missing: "multimodal futures and free-space risk",
    consumer: "MotionFormer; agent queries also feed OccFormer",
    prompt: "Why pass queries instead of boxes and masks?",
    answer: "A decoded result collapses uncertainty and context. A query remains updateable and can attend to new evidence downstream."
  },
  {
    shape: "Na × 6 × 12 · 5 × 200²",
    copy: "MotionFormer predicts multimodal agent futures; OccFormer recursively generates identity-preserving dense occupancy.",
    nodes: [["Qᴀ + Qᴍ", "interaction"], ["Motion / Occ", "coupled"], ["trajectory + Ô", "future"]],
    reads: "agent queries, map queries, BEV, and prior occupancy state",
    preserves: "six hypotheses per agent plus dense time-indexed risk",
    missing: "ego command and one committed action",
    consumer: "Planner and the inference-time collision optimizer",
    prompt: "Why keep both trajectories and occupancy?",
    answer: "Trajectories preserve agent identity and alternatives; occupancy answers the planner's cell-level safety question without object lookup."
  },
  {
    shape: "6 × (x, y)",
    copy: "Ego query, command, BEV, and occupancy meet in Planner to produce six waypoints over three seconds.",
    nodes: [["Ego + command", "intent"], ["Planner ×3", "BEV attention"], ["τ*", "safe plan"]],
    reads: "ego track, six ego motion modes, command, BEV, occupancy",
    preserves: "intent while resolving it against scene context",
    missing: "closed-loop feedback from executing the plan",
    consumer: "vehicle control outside UniAD's open-loop evaluation",
    prompt: "Does a low open-loop error prove safe driving?",
    answer: "No. It measures agreement with recorded futures, not recovery under closed-loop disturbances. UniAD's evaluation stops before control."
  }
];

function renderSystemStep(index) {
  const step = systemSteps[index];
  $("#system-stage").innerHTML = step.nodes.map((node, nodeIndex) =>
    `${nodeIndex ? '<span class="system-arrow">→</span>' : ""}<div class="system-node ${nodeIndex === 1 ? "accent" : ""}"><b>${escapeHtml(node[0])}</b><small>${escapeHtml(node[1])}</small></div>`
  ).join("");
  $("#system-shape").textContent = step.shape;
  $("#system-copy").textContent = step.copy;
  $("#system-reads").textContent = step.reads;
  $("#system-preserves").textContent = step.preserves;
  $("#system-missing").textContent = step.missing;
  $("#system-consumer").textContent = step.consumer;
  $("#system-prompt").textContent = step.prompt;
  $("#system-answer").textContent = step.answer;
}
$$("[data-system-step]").forEach((button, index) => button.addEventListener("click", () => {
  $$("[data-system-step]").forEach(item => item.setAttribute("aria-pressed", String(item === button)));
  renderSystemStep(index);
}));
renderSystemStep(0);

const lifeStates = [
  {
    title: "Newborn detection queries search the BEV",
    copy: "Freshly initialized detection queries discover objects that appear for the first time.",
    rule: "900 detection queries compete for unmatched objects",
    score: "class score not yet thresholded",
    identity: "none",
    memory: "empty",
    queries: [[18, 18, "q₁", ""], [44, 22, "q₂", ""], [70, 16, "q₃", ""], [28, 52, "q₄", ""], [60, 56, "q₅", "dim"]],
    agents: [[72, 40, "new agent"]]
  },
  {
    title: "Hungarian matching assigns a new identity",
    copy: "A newborn query is bipartite-matched to ground truth; q₃ receives identity #17.",
    rule: "newborn only: Hungarian assignment",
    score: "activate if p ≥ 0.40",
    identity: "q₃ → ground-truth #17",
    memory: "first feature appended",
    queries: [[18, 18, "q₁", "dim"], [44, 22, "q₂", "dim"], [67, 27, "q₃", ""], [28, 52, "q₄", "dim"]],
    agents: [[72, 40, "#17"]],
    links: [[70, 34, 75, 12]]
  },
  {
    title: "The query carries identity into the next frame",
    copy: "q₃ enters QIM and the memory bank, then reads the same agent as a track query in the next frame.",
    rule: "existing track is not rematched",
    score: "keep active if p ≥ 0.35",
    identity: "#17 inherited from frame t−1",
    memory: "up to four recent features",
    queries: [[58, 28, "track #17", ""]],
    agents: [[64, 45, "#17"]],
    links: [[62, 36, 90, 48]]
  },
  {
    title: "Short occlusion does not terminate the track",
    copy: "The query survives a brief dip below 0.35; the memory bank retains its four most recent features.",
    rule: "QIM retains temporarily inactive track",
    score: "p < 0.35",
    identity: "#17 retained",
    memory: "4 / 4 entries available",
    queries: [[53, 30, "track #17", "dim"]],
    agents: [],
    links: []
  },
  {
    title: "Only sustained inactivity removes the query",
    copy: "After roughly two continuous seconds of low scores, the lifecycle ends and the identity leaves the active query set.",
    rule: "remove after continuous low-score timeout",
    score: "p < 0.35 for ≈ 2 s",
    identity: "#17 retired",
    memory: "released with query",
    queries: [[49, 32, "removed", "dim"]],
    agents: [],
    links: []
  }
];

function renderLife(index) {
  const state = lifeStates[index];
  const queries = state.queries.map(query =>
    `<span class="life-query ${query[3]}" style="--x:${query[0]}%;--y:${query[1]}%">${escapeHtml(query[2])}</span>`
  ).join("");
  const agents = state.agents.map(agent =>
    `<span class="life-agent" style="--x:${agent[0]}%;--y:${agent[1]}%" data-id="${escapeHtml(agent[2])}"></span>`
  ).join("");
  const links = (state.links || []).map(link =>
    `<i class="life-link" style="--x:${link[0]}%;--y:${link[1]}%;--w:${link[2]}px;--r:${link[3]}deg"></i>`
  ).join("");
  $("#lifecycle-view").innerHTML = `<div class="life-scene">${queries}${agents}${links}
    <p class="life-caption"><b>${escapeHtml(state.title)}</b>${escapeHtml(state.copy)}</p>
    <dl class="life-ledger">
      <div><dt>assignment rule</dt><dd>${escapeHtml(state.rule)}</dd></div>
      <div><dt>score gate</dt><dd>${escapeHtml(state.score)}</dd></div>
      <div><dt>identity</dt><dd>${escapeHtml(state.identity)}</dd></div>
      <div><dt>memory bank</dt><dd>${escapeHtml(state.memory)}</dd></div>
    </dl>
  </div>`;
}
$$("[data-life]").forEach((button, index) => button.addEventListener("click", () => {
  $$("[data-life]").forEach(item => item.setAttribute("aria-pressed", String(item === button)));
  renderLife(index);
}));
renderLife(0);

const mapDots = $("#map-dots");
for (let index = 0; index < 90; index += 1) {
  const dot = document.createElement("i");
  dot.style.setProperty("--o", String(((index * 37) % 100) / 160));
  mapDots.append(dot);
}

const interactions = {
  agent: {
    title: "Who yields to whom?",
    copy: "Motion queries self-attend, then cross-attend all agent queries to encode following, yielding, and conflict.",
    query: "Qmotion ∈ RNa×6×256",
    keyValue: "QA ∈ RNa×256",
    operator: "self-attention + cross-attention",
    output: "Qa: interaction-conditioned mode feature",
    changes: "The same anchor can accelerate, brake, or yield after reading neighboring identities.",
    visual: () => {
      const nodes = [[68, 47, true], [77, 25], [83, 67], [55, 70], [48, 31]];
      return `<div class="interaction-graph">${nodes.map(node =>
        `<i class="graph-node ${node[2] ? "focus" : ""}" style="--x:${node[0]}%;--y:${node[1]}%"></i>`
      ).join("")}${nodes.slice(1).map((node, i) =>
        `<i class="graph-edge" style="--x:69%;--y:51%;--w:${85 + i * 12}px;--r:${-42 + i * 31}deg"></i>`
      ).join("")}</div>`;
    }
  },
  map: {
    title: "What geometry permits the motion?",
    copy: "The motion query cross-attends map queries, turning lane direction, road boundaries, and crossings into trajectory constraints.",
    query: "Qmotion ∈ RNa×6×256",
    keyValue: "QM ∈ R300×256",
    operator: "agent-to-map cross-attention",
    output: "Qm: topology-conditioned mode feature",
    changes: "A kinematically plausible endpoint can lose support when it crosses a boundary or leaves drivable topology.",
    visual: () => '<div class="interaction-road"></div><i class="graph-node focus" style="--x:68%;--y:48%;--s:32px"></i>'
  },
  goal: {
    title: "What is around the intended endpoint?",
    copy: "The previous layer's endpoint becomes a reference point for sparse deformable attention into BEV, refining the goal layer by layer.",
    query: "endpoint x̂T(l−1)",
    keyValue: "Bt ∈ R200×200×256",
    operator: "deformable BEV attention",
    output: "Qg: local evidence around the proposed goal",
    changes: "Layer l does not merely extend layer l−1; it resamples the world around the newly proposed destination.",
    visual: () => `<svg class="goal-field" viewBox="0 0 340 200"><path d="M18 170C96 164 168 130 318 26"/><path d="M18 170C100 172 196 168 320 154"/><path d="M18 170C100 152 186 105 288 52"/><circle cx="318" cy="26" r="7" fill="#e75832"/></svg>`
  }
};

function renderInteraction(name) {
  const interaction = interactions[name];
  $("#interaction-canvas").innerHTML = `<div class="interaction-copy"><b>${escapeHtml(interaction.title)}</b><p>${escapeHtml(interaction.copy)}</p></div>
    ${interaction.visual()}
    <dl class="interaction-ledger">
      <div><dt>query / reference</dt><dd>${escapeHtml(interaction.query)}</dd></div>
      <div><dt>keys + values</dt><dd>${escapeHtml(interaction.keyValue)}</dd></div>
      <div><dt>operator</dt><dd>${escapeHtml(interaction.operator)}</dd></div>
      <div><dt>branch output</dt><dd>${escapeHtml(interaction.output)}</dd></div>
    </dl>
    <p class="interaction-consequence"><span>What changes?</span>${escapeHtml(interaction.changes)}</p>`;
}
$$("[data-interaction]").forEach(button => button.addEventListener("click", () => {
  $$("[data-interaction]").forEach(item => item.setAttribute("aria-selected", String(item === button)));
  renderInteraction(button.dataset.interaction);
}));
renderInteraction("agent");

function drawOccupancy(canvas, step, output) {
  const context = canvas.getContext("2d");
  const size = canvas.width;
  context.clearRect(0, 0, size, size);
  context.fillStyle = "#f3f1ea";
  context.fillRect(0, 0, size, size);
  context.strokeStyle = "#d3d0c8";
  context.lineWidth = 1;
  for (let position = 0; position <= size; position += 20) {
    context.beginPath(); context.moveTo(position, 0); context.lineTo(position, size); context.stroke();
    context.beginPath(); context.moveTo(0, position); context.lineTo(size, position); context.stroke();
  }
  const objects = [
    [66 + step * 16, 190 - step * 21, 0],
    [180 - step * 10, 90 + step * 8, 1],
    [128 + step * 5, 145 - step * 5, 2]
  ];
  objects.forEach(([x, y, index]) => {
    const radius = output ? 18 + step * 2 : 10;
    context.fillStyle = index === 0 ? "rgba(231,88,50,.78)" : `rgba(39,89,199,${.36 + index * .13})`;
    context.beginPath();
    context.ellipse(x, y, radius, radius * 1.35, .18, 0, Math.PI * 2);
    context.fill();
    if (output) {
      context.strokeStyle = index === 0 ? "#e75832" : "#2759c7";
      context.setLineDash([3, 3]);
      context.stroke();
      context.setLineDash([]);
    }
  });
  context.fillStyle = "#191d20";
  context.fillRect(size / 2 - 5, size - 36, 10, 22);
}
function updateOccupancy(step) {
  $("#occ-time-label").value = `t + ${step}`;
  drawOccupancy($("#occ-input"), Math.max(0, step - 1), false);
  drawOccupancy($("#occ-output"), step, true);
  $("#occ-block-readout").textContent = `${step + 1} / 5`;
  $("#occ-input-readout").textContent = step === 0
    ? "downsampled current BEV at 1/4 resolution"
    : `F${step - 1} carried from the previous temporal block`;
  $("#occ-agent-readout").textContent = `G${step} from timestep-specific MLP${step ? " and shared agent identity" : ""}`;
  $("#occ-mask-readout").textContent = `attention constraint + auxiliary Ô${step}`;
  $("#occ-output-readout").textContent = `F${step} with identity-conditioned evidence at ${step * .5} s`;
}
$("#occ-time").addEventListener("input", event => updateOccupancy(Number(event.target.value)));
updateOccupancy(2);

const plannerCanvas = $("#planner-canvas");
const plannerContext = plannerCanvas.getContext("2d");
let plannerCommand = "forward";
function drawPlanner() {
  const context = plannerContext;
  const width = plannerCanvas.width;
  const height = plannerCanvas.height;
  const optimized = $("#collision-toggle").checked;
  context.clearRect(0, 0, width, height);
  context.fillStyle = "#151b1e";
  context.fillRect(0, 0, width, height);
  context.strokeStyle = "#293236";
  for (let x = 0; x < width; x += 40) {
    context.beginPath(); context.moveTo(x, 0); context.lineTo(x, height); context.stroke();
  }
  for (let y = 0; y < height; y += 40) {
    context.beginPath(); context.moveTo(0, y); context.lineTo(width, y); context.stroke();
  }
  const obstacles = [[420, 245, 42], [510, 168, 34], [310, 122, 28]];
  obstacles.forEach(([x, y, radius], index) => {
    const gradient = context.createRadialGradient(x, y, 2, x, y, radius * 2.1);
    gradient.addColorStop(0, "rgba(231,88,50,.72)");
    gradient.addColorStop(1, "rgba(231,88,50,0)");
    context.fillStyle = gradient;
    context.beginPath(); context.arc(x, y, radius * 2.1, 0, Math.PI * 2); context.fill();
    context.fillStyle = "#e75832";
    context.fillRect(x - 8, y - 14, 16, 28);
    context.fillStyle = "#93a0a3";
    context.font = "10px monospace";
    context.fillText(`O${index + 1}`, x + 13, y + 4);
  });
  const direction = plannerCommand === "left" ? -1 : plannerCommand === "right" ? 1 : 0;
  const raw = Array.from({length: 7}, (_, index) => {
    const t = index / 6;
    return [width / 2 + direction * t * t * 190, height - 46 - t * 410];
  });
  const safe = raw.map(([x, y], index) => {
    if (!optimized) return [x, y];
    const obstacle = obstacles[index < 3 ? 0 : 1];
    const dx = x - obstacle[0], dy = y - obstacle[1];
    const distance = Math.hypot(dx, dy);
    if (distance < 110) return [x + (dx >= 0 ? 1 : -1) * (110 - distance) * .72, y];
    return [x, y];
  });
  context.strokeStyle = "#7d898c";
  context.setLineDash([6, 7]);
  context.lineWidth = 2;
  context.beginPath();
  raw.forEach((point, index) => index ? context.lineTo(...point) : context.moveTo(...point));
  context.stroke();
  context.setLineDash([]);
  context.strokeStyle = "#e7c84a";
  context.lineWidth = 4;
  context.beginPath();
  safe.forEach((point, index) => index ? context.lineTo(...point) : context.moveTo(...point));
  context.stroke();
  safe.slice(1).forEach(point => {
    context.fillStyle = "#e7c84a";
    context.beginPath(); context.arc(point[0], point[1], 5, 0, Math.PI * 2); context.fill();
  });
  context.fillStyle = "#f7f6f1";
  context.fillRect(width / 2 - 9, height - 58, 18, 36);
  context.fillStyle = "#879397";
  context.font = "11px monospace";
  context.fillText("τ̂ learned reference", 24, height - 42);
  context.fillStyle = "#e7c84a";
  context.fillText("τ* optimized plan", 24, height - 24);
  const referenceCost = safe.reduce((sum, point, index) => {
    const dx = point[0] - raw[index][0];
    const dy = point[1] - raw[index][1];
    return sum + dx * dx + dy * dy;
  }, 0) / 1000;
  let closestClearance = Infinity;
  const obstacleCost = safe.slice(1).reduce((sum, point, index) => {
    const timeObstacle = obstacles[Math.min(index, obstacles.length - 1)];
    const clearance = Math.hypot(point[0] - timeObstacle[0], point[1] - timeObstacle[1]) - timeObstacle[2];
    closestClearance = Math.min(closestClearance, clearance);
    return sum + Math.exp(-Math.max(0, clearance) / 42);
  }, 0);
  $("#planner-reference-cost").textContent = referenceCost.toFixed(2);
  $("#planner-obstacle-cost").textContent = obstacleCost.toFixed(3);
  $("#planner-clearance").textContent = `${Math.max(0, closestClearance / 40).toFixed(2)} grid units`;
  $("#planner-interpretation").textContent = optimized
    ? "reference bent where time-matched occupancy is close"
    : "learned reference shown without the inference correction";
}
$$("[data-command]").forEach(button => button.addEventListener("click", () => {
  plannerCommand = button.dataset.command;
  $$("[data-command]").forEach(item => item.setAttribute("aria-pressed", String(item === button)));
  drawPlanner();
}));
$("#collision-toggle").addEventListener("change", drawPlanner);
drawPlanner();

const contracts = {
  tiny: {
    nodes: [
      ["Input", "6 × 3 × 8 × 8", "procedural planes"],
      ["World", "8 × 8 × 16", "4 KiB temporal BEV"],
      ["Queries", "64 → top 8", "stable scalar score"],
      ["Future", "3 × 4", "analytic modes"],
      ["Action", "6 points", "plan + collision"]
    ],
    note: "Fixed capacities, seeded weights, and no remote dependencies; executable against an independent PyTorch oracle.",
    runnable: "yes · CPU and optional CUDA marker path",
    weights: "seed-derived synthetic container",
    proof: "two-frame agreement with an independent PyTorch oracle",
    boundary: "operator and checkpoint coverage stop before production equivalence"
  },
  upstream: {
    nodes: [
      ["Input", "6 × BGR", "R101 + 4-level FPN"],
      ["World", "200² × 256", "6-layer BEV encoder"],
      ["Queries", "900 + 300", "agent + map"],
      ["Future", "6 × 12 · 5", "motion · occupancy"],
      ["Action", "6 points", "3-layer planner"]
    ],
    note: "The production contract described by the paper and pinned source; this repository rejects the profile because no compatible weight export exists.",
    runnable: "no · declared contract is rejected before inference",
    weights: "released PyTorch checkpoint has no C11 name/shape mapping",
    proof: "paper, supplement, pinned config, and source-level architecture study",
    boundary: "no ResNet/FPN, deformable attention, learned decoders, export, or nuScenes evaluator"
  }
};
function renderContract(name) {
  const contract = contracts[name];
  $("#contract-flow").innerHTML = contract.nodes.map((node, index) =>
    `<div class="contract-node"><small>0${index + 1} · ${escapeHtml(node[0])}</small><b>${escapeHtml(node[1])}</b><span>${escapeHtml(node[2])}</span></div>`
  ).join("");
  $("#contract-note").textContent = contract.note;
  $("#contract-runnable").textContent = contract.runnable;
  $("#contract-weights").textContent = contract.weights;
  $("#contract-proof").textContent = contract.proof;
  $("#contract-boundary").textContent = contract.boundary;
}
$$("[data-contract]").forEach(button => button.addEventListener("click", () => {
  $$("[data-contract]").forEach(item => item.setAttribute("aria-pressed", String(item === button)));
  renderContract(button.dataset.contract);
}));
renderContract("tiny");

const canonicalCanvas = $("#canonical-canvas");
const canonicalContext = canonicalCanvas.getContext("2d");
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
  if (Array.isArray(map.points)) return map.points;
  return [[map.x0, map.y0], [map.x1, map.y1]];
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
    context.beginPath();
    context.moveTo(px, canonicalPadding.top);
    context.lineTo(px, canonicalCanvas.height - canonicalPadding.bottom);
    context.stroke();
    context.fillStyle = "#718084";
    context.fillText(String(x), px, canonicalCanvas.height - 24);
  }
  for (let y = -4; y <= 5; y += 1) {
    const [, py] = canonicalPoint([0, y]);
    context.strokeStyle = y === 0 ? "#677276" : "#273135";
    context.lineWidth = y === 0 ? 1.5 : 1;
    context.beginPath();
    context.moveTo(canonicalPadding.left, py);
    context.lineTo(canonicalCanvas.width - canonicalPadding.right, py);
    context.stroke();
    context.textAlign = "right";
    context.fillStyle = "#718084";
    context.fillText(String(y), 43, py);
    context.textAlign = "center";
  }
  context.fillStyle = "#93a0a3";
  context.fillText("x / right →", canonicalCanvas.width - 78, canonicalCanvas.height - 24);
  context.save();
  context.translate(17, 72);
  context.rotate(-Math.PI / 2);
  context.fillText("y / forward →", 0, 0);
  context.restore();
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
    const x = index % 8 - 3.5;
    const y = Math.floor(index / 8) - 3.5;
    const [px, py] = canonicalPoint([x, y]);
    context.fillStyle = "rgba(231,88,50,.20)";
    context.strokeStyle = "rgba(231,88,50,.55)";
    context.lineWidth = 1;
    context.fillRect(px - (x1 - x0) / 2, py - (y0 - y1) / 2, x1 - x0, y0 - y1);
    context.strokeRect(px - (x1 - x0) / 2, py - (y0 - y1) / 2, x1 - x0, y0 - y1);
  });
  context.fillStyle = "#e78368";
  context.font = "10px ui-monospace, monospace";
  context.textAlign = "left";
  context.fillText(`occupancy horizon ${horizonIndex} · ${cells.filter(Boolean).length}/64 cells`, 66, 18);
}

function drawCanonical(result) {
  const context = canonicalContext;
  canonicalHitTargets = [];
  context.clearRect(0, 0, canonicalCanvas.width, canonicalCanvas.height);
  drawCanonicalGrid(context);

  if (canonicalLayers.has("occupancy")) drawCanonicalOccupancy(context, result);

  if (canonicalLayers.has("map")) {
    (result.map || []).forEach((map, index) => {
      const points = canonicalMapPoints(map);
      if (points.some(point => point.some(value => !Number.isFinite(value)))) return;
      context.strokeStyle = index === 2 ? "#aeb8ba" : "#667376";
      context.setLineDash(index === 2 ? [] : [8, 7]);
      context.lineWidth = index === 2 ? 2 : 1.5;
      context.beginPath();
      points.forEach((value, pointIndex) => pointIndex
        ? context.lineTo(...canonicalPoint(value))
        : context.moveTo(...canonicalPoint(value)));
      context.stroke();
    });
    context.setLineDash([]);
  }

  if (canonicalLayers.has("motion")) {
    (result.motion || []).forEach(motion => {
      const track = (result.tracks || []).find(candidate => candidate.id === motion.track_id);
      const mode = motion.modes[canonicalMode];
      if (!track || !mode) return;
      const selected = canonicalSelectedTrack === motion.track_id;
      const trajectory = [[track.x, track.y], ...mode.trajectory.slice(0, canonicalStep + 1)];
      context.strokeStyle = selected ? "#8fb0ff" : "rgba(93,136,232,.58)";
      context.lineWidth = selected ? 4 : 2;
      context.beginPath();
      trajectory.forEach((value, index) => index
        ? context.lineTo(...canonicalPoint(value))
        : context.moveTo(...canonicalPoint(value)));
      context.stroke();
      trajectory.slice(1).forEach((value, index) => {
        const [x, y] = canonicalPoint(value);
        context.fillStyle = selected ? "#dce6ff" : "#5d88e8";
        context.beginPath();
        context.arc(x, y, selected ? 4 : 2.5, 0, Math.PI * 2);
        context.fill();
        if (selected) {
          context.fillStyle = "#aeb8ba";
          context.font = "9px ui-monospace, monospace";
          context.textAlign = "left";
          context.fillText(`t${index + 1}`, x + 6, y - 6);
        }
      });
    });
  }

  if (canonicalLayers.has("plan")) {
    const plan = [[0, 0], ...(result.ego_plan || [])];
    context.strokeStyle = "#e7c84a";
    context.lineWidth = 4;
    context.beginPath();
    plan.forEach((value, index) => index
      ? context.lineTo(...canonicalPoint(value))
      : context.moveTo(...canonicalPoint(value)));
    context.stroke();
    plan.slice(1).forEach((value, index) => {
      const [x, y] = canonicalPoint(value);
      context.fillStyle = "#e7c84a";
      context.beginPath();
      context.arc(x, y, 4.5, 0, Math.PI * 2);
      context.fill();
      context.fillStyle = "#d8d39c";
      context.font = "9px ui-monospace, monospace";
      context.textAlign = "left";
      context.fillText(String(index + 1), x + 7, y);
    });
  }

  if (canonicalLayers.has("tracks")) {
    (result.tracks || []).forEach(track => {
      const [x, y] = canonicalPoint([track.x, track.y]);
      const selected = canonicalSelectedTrack === track.id;
      context.fillStyle = selected ? "#ff9a78" : "#e75832";
      context.strokeStyle = selected ? "#fff1eb" : "#e75832";
      context.lineWidth = selected ? 3 : 1;
      context.fillRect(x - 7, y - 12, 14, 24);
      context.strokeRect(x - 10, y - 15, 20, 30);
      context.fillStyle = selected ? "#ffffff" : "#aeb8ba";
      context.font = `${selected ? "bold " : ""}10px ui-monospace, monospace`;
      context.textAlign = "left";
      context.fillText(`#${track.id}`, x + 13, y + 3);
      canonicalHitTargets.push({id: track.id, x, y, radius: 20});
    });
  }

  const [egoX, egoY] = canonicalPoint([0, 0]);
  context.fillStyle = "#f7f6f1";
  context.fillRect(egoX - 8, egoY - 15, 16, 30);
  context.fillStyle = "#111719";
  context.font = "bold 9px ui-monospace, monospace";
  context.textAlign = "center";
  context.fillText("E", egoX, egoY);
}

function renderCanonicalInspector() {
  if (!canonicalResult) return;
  const track = canonicalResult.tracks.find(candidate => candidate.id === canonicalSelectedTrack);
  if (!track) {
    $("#inspector-kind").textContent = "Record";
    $("#inspector-title").textContent = canonicalResult.scene;
    $("#inspector-values").innerHTML = `
      <div><dt>schema</dt><dd>${escapeHtml(canonicalResult.schema)}</dd></div>
      <div><dt>frame</dt><dd>${escapeHtml(canonicalResult.coordinate_frame)}</dd></div>
      <div><dt>command</dt><dd>${escapeHtml(canonicalResult.command)}</dd></div>
      <div><dt>occupancy</dt><dd>${canonicalResult.occupancy.length} × 8 × 8</dd></div>`;
    $("#inspector-note").textContent = "Select an orange track. The viewport deliberately includes negative y, where this synthetic input placed all eight top-k cells.";
    return;
  }
  const motion = canonicalResult.motion.find(candidate => candidate.track_id === track.id);
  const mode = motion && motion.modes[canonicalMode];
  const endpoint = mode && mode.trajectory[Math.min(canonicalStep, mode.trajectory.length - 1)];
  $("#inspector-kind").textContent = "Selected track";
  $("#inspector-title").textContent = `query cell #${track.id}`;
  $("#inspector-values").innerHTML = `
    <div><dt>position</dt><dd>(${track.x.toFixed(2)}, ${track.y.toFixed(2)}) m</dd></div>
    <div><dt>track score</dt><dd>${track.score.toFixed(6)}</dd></div>
    <div><dt>mode</dt><dd>${canonicalMode} · p=${mode ? mode.score.toFixed(6) : "n/a"}</dd></div>
    <div><dt>endpoint t${canonicalStep + 1}</dt><dd>${endpoint ? `(${endpoint[0].toFixed(2)}, ${endpoint[1].toFixed(2)}) m` : "n/a"}</dd></div>`;
  $("#inspector-note").textContent = "The ID is the stable top-k BEV-cell index in the tiny runtime—not a production TrackFormer identity.";
}

function refreshCanonical() {
  if (!canonicalResult) return;
  const occupancyHorizon = Math.min(canonicalStep, canonicalResult.occupancy.length - 1);
  $("#canonical-time-label").value = `${canonicalStep + 1} / 4 · occ h${occupancyHorizon}`;
  drawCanonical(canonicalResult);
  renderCanonicalInspector();
}

$("#canonical-time").addEventListener("input", event => {
  canonicalStep = Number(event.target.value);
  refreshCanonical();
});
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
  canonicalCanvas.style.cursor = canonicalHitTargets.some(target => Math.hypot(x - target.x, y - target.y) <= target.radius)
    ? "pointer" : "crosshair";
});
canonicalCanvas.addEventListener("click", event => {
  const bounds = canonicalCanvas.getBoundingClientRect();
  const x = (event.clientX - bounds.left) * canonicalCanvas.width / bounds.width;
  const y = (event.clientY - bounds.top) * canonicalCanvas.height / bounds.height;
  const target = canonicalHitTargets.find(candidate => Math.hypot(x - candidate.x, y - candidate.y) <= candidate.radius);
  canonicalSelectedTrack = target ? target.id : null;
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
