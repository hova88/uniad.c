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
    nodes: [["6 cameras", "perspective"], ["R101 + FPN", "4 scales"], ["Fcam", "view features"]]
  },
  {
    shape: "200 × 200 × 256",
    copy: "The BEV encoder alternates temporal self-attention and spatial cross-attention to form a shared world state.",
    nodes: [["Fcam + Bₜ₋₁", "aligned"], ["BEV encoder ×6", "temporal + spatial"], ["Bₜ", "dense world"]]
  },
  {
    shape: "Na × 256 · 300 × 256",
    copy: "TrackFormer emits a dynamic set of agent queries; MapFormer emits 300 road-instance queries.",
    nodes: [["Bₜ", "shared"], ["Track / Map", "parallel"], ["Qᴀ + Qᴍ", "sparse entities"]]
  },
  {
    shape: "Na × 6 × 12 · 5 × 200²",
    copy: "MotionFormer predicts multimodal agent futures; OccFormer recursively generates identity-preserving dense occupancy.",
    nodes: [["Qᴀ + Qᴍ", "interaction"], ["Motion / Occ", "coupled"], ["trajectory + Ô", "future"]]
  },
  {
    shape: "6 × (x, y)",
    copy: "Ego query, command, BEV, and occupancy meet in Planner to produce six waypoints over three seconds.",
    nodes: [["Ego + command", "intent"], ["Planner ×3", "BEV attention"], ["τ*", "safe plan"]]
  }
];

function renderSystemStep(index) {
  const step = systemSteps[index];
  $("#system-stage").innerHTML = step.nodes.map((node, nodeIndex) =>
    `${nodeIndex ? '<span class="system-arrow">→</span>' : ""}<div class="system-node ${nodeIndex === 1 ? "accent" : ""}"><b>${escapeHtml(node[0])}</b><small>${escapeHtml(node[1])}</small></div>`
  ).join("");
  $("#system-shape").textContent = step.shape;
  $("#system-copy").textContent = step.copy;
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
    queries: [[18, 18, "q₁", ""], [44, 22, "q₂", ""], [70, 16, "q₃", ""], [28, 52, "q₄", ""], [60, 56, "q₅", "dim"]],
    agents: [[72, 40, "new agent"]]
  },
  {
    title: "Hungarian matching assigns a new identity",
    copy: "A newborn query is bipartite-matched to ground truth; q₃ receives identity #17.",
    queries: [[18, 18, "q₁", "dim"], [44, 22, "q₂", "dim"], [67, 27, "q₃", ""], [28, 52, "q₄", "dim"]],
    agents: [[72, 40, "#17"]],
    links: [[70, 34, 75, 12]]
  },
  {
    title: "The query carries identity into the next frame",
    copy: "q₃ enters QIM and the memory bank, then reads the same agent as a track query in the next frame.",
    queries: [[58, 28, "track #17", ""]],
    agents: [[64, 45, "#17"]],
    links: [[62, 36, 90, 48]]
  },
  {
    title: "Short occlusion does not terminate the track",
    copy: "The query survives a brief dip below 0.35; the memory bank retains its four most recent features.",
    queries: [[53, 30, "track #17", "dim"]],
    agents: [],
    links: []
  },
  {
    title: "Only sustained inactivity removes the query",
    copy: "After roughly two continuous seconds of low scores, the lifecycle ends and the identity leaves the active query set.",
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
  $("#lifecycle-view").innerHTML = `<div class="life-scene">${queries}${agents}${links}<p class="life-caption"><b>${escapeHtml(state.title)}</b>${escapeHtml(state.copy)}</p></div>`;
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
    visual: () => '<div class="interaction-road"></div><i class="graph-node focus" style="--x:68%;--y:48%;--s:32px"></i>'
  },
  goal: {
    title: "What is around the intended endpoint?",
    copy: "The previous layer's endpoint becomes a reference point for sparse deformable attention into BEV, refining the goal layer by layer.",
    visual: () => `<svg class="goal-field" viewBox="0 0 340 200"><path d="M18 170C96 164 168 130 318 26"/><path d="M18 170C100 172 196 168 320 154"/><path d="M18 170C100 152 186 105 288 52"/><circle cx="318" cy="26" r="7" fill="#e75832"/></svg>`
  }
};

function renderInteraction(name) {
  const interaction = interactions[name];
  $("#interaction-canvas").innerHTML = `<div class="interaction-copy"><b>${escapeHtml(interaction.title)}</b><p>${escapeHtml(interaction.copy)}</p></div>${interaction.visual()}`;
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
    note: "Fixed capacities, seeded weights, and no remote dependencies; executable against an independent PyTorch oracle."
  },
  upstream: {
    nodes: [
      ["Input", "6 × BGR", "R101 + 4-level FPN"],
      ["World", "200² × 256", "6-layer BEV encoder"],
      ["Queries", "900 + 300", "agent + map"],
      ["Future", "6 × 12 · 5", "motion · occupancy"],
      ["Action", "6 points", "3-layer planner"]
    ],
    note: "The production contract described by the paper and pinned source; this repository rejects the profile because no compatible weight export exists."
  }
};
function renderContract(name) {
  const contract = contracts[name];
  $("#contract-flow").innerHTML = contract.nodes.map((node, index) =>
    `<div class="contract-node"><small>0${index + 1} · ${escapeHtml(node[0])}</small><b>${escapeHtml(node[1])}</b><span>${escapeHtml(node[2])}</span></div>`
  ).join("");
  $("#contract-note").textContent = contract.note;
}
$$("[data-contract]").forEach(button => button.addEventListener("click", () => {
  $$("[data-contract]").forEach(item => item.setAttribute("aria-pressed", String(item === button)));
  renderContract(button.dataset.contract);
}));
renderContract("tiny");

const canonicalCanvas = $("#canonical-canvas");
const canonicalContext = canonicalCanvas.getContext("2d");
function drawCanonical(result, reveal = 1) {
  const context = canonicalContext;
  const width = canonicalCanvas.width;
  const height = canonicalCanvas.height;
  const originX = width / 2;
  const originY = height - 52;
  const scale = 58;
  const point = ([x, y]) => [originX + x * scale, originY - y * scale];
  context.clearRect(0, 0, width, height);
  context.fillStyle = "#13191c";
  context.fillRect(0, 0, width, height);
  context.strokeStyle = "#273135";
  context.lineWidth = 1;
  for (let x = 0; x <= width; x += scale) {
    context.beginPath(); context.moveTo(x, 0); context.lineTo(x, height); context.stroke();
  }
  for (let y = originY; y >= 0; y -= scale) {
    context.beginPath(); context.moveTo(0, y); context.lineTo(width, y); context.stroke();
  }
  (result.map || []).forEach(map => {
    context.strokeStyle = "#667376";
    context.setLineDash([8, 7]);
    context.beginPath(); context.moveTo(...point([map.x0, map.y0])); context.lineTo(...point([map.x1, map.y1])); context.stroke();
  });
  context.setLineDash([]);
  (result.motion || []).forEach(motion => motion.modes.forEach((mode, index) => {
    context.strokeStyle = `rgba(93,136,232,${.75 - index * .18})`;
    context.lineWidth = 2;
    context.beginPath();
    mode.trajectory.forEach((value, valueIndex) => valueIndex ? context.lineTo(...point(value)) : context.moveTo(...point(value)));
    context.stroke();
  }));
  context.strokeStyle = "#e7c84a";
  context.lineWidth = 4;
  context.beginPath();
  (result.ego_plan || []).forEach((value, index) => {
    const [x, y] = point(value);
    const animated = [x, originY + (y - originY) * reveal];
    index ? context.lineTo(...animated) : context.moveTo(...animated);
  });
  context.stroke();
  (result.tracks || []).forEach(track => {
    const [x, y] = point([track.x, track.y]);
    context.fillStyle = "#e75832";
    context.fillRect(x - 6, y - 10, 12, 20);
    context.fillStyle = "#889597";
    context.font = "9px monospace";
    context.fillText(`#${track.id}`, x + 10, y + 4);
  });
  context.fillStyle = "#f7f6f1";
  context.fillRect(originX - 7, originY - 18, 14, 36);
}
fetch("assets/demo-result.json").then(response => {
  if (!response.ok) throw new Error("result unavailable");
  return response.json();
}).then(result => {
  $("#frame-readout").textContent = result.frame_index;
  $("#tracks-readout").textContent = result.tracks.length;
  $("#collision-readout").textContent = Number(result.collision_score).toFixed(3);
  if (matchMedia("(prefers-reduced-motion: reduce)").matches) {
    drawCanonical(result);
  } else {
    const start = performance.now();
    const animate = now => {
      const progress = Math.min(1, (now - start) / 900);
      drawCanonical(result, .3 + .7 * progress);
      if (progress < 1) requestAnimationFrame(animate);
    };
    requestAnimationFrame(animate);
  }
}).catch(() => {
  $("#collision-readout").textContent = "n/a";
});
