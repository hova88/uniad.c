# Interaction design study for the UniAD visual essay

This document is the design contract for the article's explanatory views. It
exists to prevent interaction from becoming decoration, a dashboard, or a
second navigation system.

## The comprehension test

Every interactive view must answer five questions before implementation:

1. What misconception or unresolved question does the reader bring in?
2. What is the smallest meaningful action the reader can take?
3. Which visual variable changes as a direct consequence?
4. What invariant stays visible so the reader can compare before and after?
5. What one-sentence conclusion becomes obvious without reading a manual?

If those answers are weak, the figure should remain static.

## Findings from the current page

| View | Current action | Failure |
|---|---|---|
| System | choose one of five tabs | replaces the entire state, hiding continuity and causality |
| Track | choose one of five lifecycle buttons | makes the reader reconstruct time from disconnected cards |
| Motion | choose one of three tabs | incorrectly makes three parallel branches feel mutually exclusive |
| Occupancy | scrub future time | the gesture is appropriate, but unrelated blobs hide pixel-agent interaction |
| Planner | choose command / toggle optimizer | path response is useful, but the surrounding parameter dashboard dominates |
| Runtime contract | toggle tiny / upstream | comparison is real, but replacement makes missing operators hard to locate |
| Canonical result | layer, mode, time, and selection controls | technically accurate, but too much control surface for the narrative default |

The shared failure is indirectness: the reader manipulates interface state,
then looks elsewhere to infer what changed. The redesign moves manipulation
into the explanatory object itself.

## Interaction grammar

### One scene, one gesture

Each figure keeps a stable coordinate system and permits one primary gesture.
Secondary detail may appear on focus or selection, but it must not create
another workflow.

### Continuous variables deserve continuous input

Time, ego displacement, endpoints, and obstacle position use direct dragging or
scrubbing. Lifecycle states should be traversed as one timeline rather than
five unrelated buttons.

### Discrete structure deserves direct selection

Architecture nodes and evidence dependencies may be selected in place. The
selected object highlights its incoming and outgoing edges while unrelated
structure recedes. A separate tab bar is unnecessary.

### Preserve a baseline

Whenever the point is a difference, the prior state remains as a thin ghost:
unaligned BEV, independent trajectory, learned reference plan, or production
operator. This avoids memory-based comparison.

### Progressive disclosure

The default state contains only the scene, the prompt, a short instruction, and
the conclusion. Tensor shapes, exact scores, and source lineage remain
available through local annotations or a single details disclosure.

### Motion is semantic

Transitions last roughly 180–280 ms and map to meaning:

- position changes represent geometry or time;
- opacity changes represent evidence availability or attention;
- line weight changes represent active dependency;
- color changes represent semantic role, not generic selection.

No decorative looping animation is permitted. Reduced-motion mode applies the
final state immediately.

### Input parity

Every essential action must work with mouse, touch, and keyboard. Hover can
preview, never unlock unique content. Hit targets are at least 40 CSS pixels.

## Section storyboard

| Section | Question | Primary action | Stable baseline | Visual consequence | Intended realization |
|---|---|---|---|---|---|
| Problem | Why do accurate modules still produce brittle plans? | drag an interface aperture from rich query to decoded result | the same observed agent and planner | context and uncertainty are progressively discarded | a box preserves location but not the beliefs planning needs |
| System | Who reads which representation? | select a module directly in the dependency graph | the full two-path graph | only true inputs, outputs, and bypasses remain saturated | UniAD is a graph with sparse and dense paths, not a simple chain |
| BEV | Why must the previous frame be aligned? | drag ego displacement | current BEV grid | past actors slide into or out of registration; overlap changes | temporal attention is meaningful only after ego-motion compensation |
| Track | Where does identity survive an occlusion? | scrub one frame timeline | one car and one query token | score, assignment, memory, and visibility evolve continuously | identity belongs to query state, not a freshly decoded box |
| Map | How does road geometry constrain a possible future? | drag one endpoint | lane, boundary, crossing, and current agent | feasibility and attended map element change at the endpoint | kinematics alone cannot distinguish on-road from off-road futures |
| Motion | What changes when context is removed? | press and hold “remove context” | contextual trajectory and its ghost | independent forecast crosses a conflict or boundary | agent, map, and goal context jointly shape each motion mode |
| Occupancy | How do sparse futures become cell-level risk? | scrub future time | one BEV world | agents move while identity-colored occupancy cells accumulate | OccFormer writes sparse agent futures back into dense space |
| Planning | How does occupancy alter intent without replacing it? | drag one risk hotspot | learned reference path | optimized path bends while the ghost reference stays fixed | optimization trades deviation for clearance at matched time |
| Learning | Why not train everything from the start? | move between two stage stops | the same module graph | trainable/frozen modules and gradient destinations change | staged training first stabilizes the shared coordinate system |
| Evidence | Which claimed dependency has experimental support? | select one dependency edge | the complete task graph | exact before/after metric appears on that edge | ablations support specific information routes, not “unification” in general |
| Runtime | Where does the C11 slice stop matching production? | drag a comparison wipe | aligned production and tiny pipelines | missing operators are exposed at the same semantic positions | representation-flow similarity is not checkpoint equivalence |
| Correctness | What additional evidence unlocks the next claim? | select a rung | the full ladder | requirement and currently available artifact appear locally | running code is only the third of five evidence levels |

## Canonical result policy

The canonical JSON explorer remains valuable, but it is evidence detail rather
than the primary explanation. Its narrative default should show the complete
scene with a single instruction: select a track. Layer, mode, and future controls
move into one native `details` disclosure. This keeps auditability without
turning the article into a monitoring console.

## Acceptance criteria

- No explanatory view uses a detached tab bar to replace the entire scene.
- MotionFormer never implies that its three branches are mutually exclusive.
- Every continuous interaction preserves a visible baseline.
- Each figure states its prompt and conclusion inside the figure boundary.
- A first-time reader can identify the intended gesture without prose outside
  the figure.
- The article remains understandable without interaction and with JavaScript
  disabled.
- Keyboard, touch, pointer, and reduced-motion paths are verified.
- Mobile views retain the same causal comparison rather than falling back to a
  static screenshot.
