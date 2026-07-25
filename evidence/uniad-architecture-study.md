# UniAD architecture study

This note records the primary-source findings used to write the technical
article. It is tied to OpenDriveLab/UniAD commit
`609ee083ea51c3521c323f1279dfc4cee0e60467`, the CVPR 2023 paper, and its
supplement. It is not a claim that the C runtime implements the production
network.

## Central thesis

UniAD is not a set of parallel heads attached to a shared image encoder. It is
a hierarchical, planning-oriented graph. Sparse entity queries and dense scene
features move through different but coupled paths:

```text
images → temporal BEV ───────────────┬──────────────→ Planner
                    ├→ Track queries ┼→ Motion queries → ego plan query
                    ├→ Map queries ──┘
                    └→ OccFormer ← agent / motion queries → dense occupancy
```

The differentiable query interfaces are the main mechanism for avoiding the
information loss of box-only module boundaries. They carry a 256-dimensional
context rather than only a decoded task result.

## Input and BEV state

- Six camera images are normalized in BGR order and passed through a frozen
  ResNet-101 plus FPN in the released base configuration.
- Four feature levels have 256 channels.
- BEVFormer produces a `200 × 200 × 256` state over approximately
  `[-51.2 m, 51.2 m]` in x and y.
- The encoder has six layers. Each layer performs temporal self-attention,
  spatial cross-attention, and an FFN.
- Previous BEV is rotated and shifted using ego motion / CAN bus metadata.
- Training queues contain five frames in the full experiment; the released
  stage-2 configuration uses a queue length of three.

## TrackFormer

- 900 object queries enter a six-layer detection / tracking decoder.
- Newborn detection queries discover agents not seen before.
- Track queries inherit the identity assignment of prior frames and continue
  attending to the BEV state.
- Hungarian matching is used for newborn queries. Existing track queries keep
  their previous ground-truth index rather than being matched from scratch.
- The Query Interaction Module updates active queries and applies random drop /
  false-positive augmentation during training.
- A four-entry memory bank carries temporal query features.
- Inference thresholds are 0.4 for detection queries and 0.35 for track
  queries. A low-scoring track is retained through short occlusions and removed
  only after a continuous inactive period (two seconds in the supplement).
- A dedicated ego-vehicle query is appended. It is not part of ordinary
  agent-to-ground-truth matching and becomes the sparse ego representation used
  by MotionFormer and Planner.

## MapFormer

- MapFormer adapts Panoptic SegFormer to online BEV mapping.
- It uses 300 thing queries for lanes, boundaries, and pedestrian crossings,
  plus a class-fixed stuff query for drivable area.
- Six location-decoder layers locate map elements; four thing-mask decoder
  layers and six stuff-mask decoder layers produce masks in the released
  configuration.
- Every decoder layer is supervised, but only final-layer thing query features
  become `Q_M` for downstream agent–map interaction.

## MotionFormer

- Input agent features `Q_A` have dynamic agent count `N_a`; map features
  `Q_M` have 300 entries. Both use dimension 256.
- Each agent, including ego, has six motion modes and twelve future positions.
- Class-grouped k-means trajectory anchors initialize the six modes.
- Agent-level anchors express a local motion prior. Scene-level anchors are
  rotated and translated by the detected agent pose.
- Motion query position combines four terms:
  scene-level anchor, agent-level anchor, current agent position, and the goal
  predicted by the previous decoder layer.
- Each of three MotionFormer layers computes three interactions in parallel:
  agent–agent, agent–map, and agent–goal.
- Agent–agent and agent–map use self-attention followed by cross-attention.
  Agent–goal uses deformable attention into the BEV around the previous
  predicted endpoint.
- The three results are concatenated and fused to form the next query context.
- The decoder predicts per-step displacement / velocity and cumulatively sums
  it into trajectories.
- Training targets may be adjusted by a nonlinear smoother to compensate for
  imperfect upstream location and heading. This target smoothing is
  training-only and is distinct from planner collision optimization.
- The motion loss is a MultiPath-style Gaussian-mixture objective:
  classification and negative log likelihood each have weight 0.5; minFDE has
  an additional 0.25 weight in the released configuration.

## OccFormer

- Occupancy is predicted for five timestamps: current plus four future steps,
  covering two seconds at 2 Hz.
- Motion queries are max-pooled over the six modes and concatenated with track
  query and agent-position embeddings.
- A timestep-specific MLP turns that sparse representation into agent feature
  `G^t`.
- BEV is downsampled to 1/4 resolution. Each temporal block downsamples again
  to 1/8 for pixel–agent interaction.
- Dense pixels act as queries; agent features act as keys and values.
- A coarse mask restricts each pixel to the corresponding agent. It is produced
  by the dot product of a learned agent mask feature and the dense state.
- The updated dense state is upsampled and residually added to the previous
  state before entering the next time block.
- A shared convolutional decoder restores full `200 × 200` resolution.
- A second agent projection produces occupancy features. Their dot product with
  the decoded dense state yields identity-preserving occupancy logits.
- Binary cross entropy has weight 5 and Dice loss has weight 1. The coarse
  attention mask receives an auxiliary loss of the same form.

## Planner

- Three navigation commands (left, right, forward) are learned embeddings.
- Ego track query, six ego motion-mode queries, and the command embedding are
  fused by MLPs and max pooled across motion modes into one plan query.
- A three-layer transformer decoder cross-attends this query to positional BEV
  features.
- An MLP regresses six 2D displacements, cumulatively summed into a three-second
  ego trajectory.
- Training uses imitation loss plus three collision margins:
  `(weight, expansion) = (1.0, 0.0), (0.4, 0.5), (0.1, 1.0)`.
- At inference, optional occupancy-aware nonlinear optimization balances
  distance to the learned reference trajectory against Gaussian obstacle costs.
  The released parameters use a five-meter occupancy filter, `σ = 1`, and
  collision weight 5.

## Learning schedule

- Stage 1 loads BEVFormer weights and trains tracking plus mapping for six
  epochs. The image backbone is frozen.
- Stage 2 adds motion, occupancy, and planning for twenty epochs. The image
  backbone and BEV encoder are frozen to control memory.
- Full experiments use five-frame sequences, batch size one, AdamW,
  learning rate `2e-4`, backbone multiplier `0.1`, weight decay `1e-2`, and
  sixteen A100 GPUs.
- The complete loss is the sum of tracking, mapping, motion, occupancy, and
  planning terms, with auxiliary decoder losses inside multiple tasks.

## Evaluation and interpretation

- Tracking: AMOTA, AMOTP, recall, identity switches.
- Mapping: per-class IoU.
- Motion: minADE, minFDE, miss rate, EPA, and minFDE-AP; standard displacement
  metrics are computed only on matched true positives.
- Occupancy: scene IoU and identity-sensitive VPQ in near and far ranges.
- Planning: L2 displacement and collision rate at future timestamps.
- The published full R101 graph is about 125M parameters, 1709 GFLOPs, and
  1.8 FPS on an A100 under the paper's measurement setup.
- The ablations support a dependency claim rather than a generic
  multi-task-learning claim: mapping improves motion; motion query helps
  occupancy; BEV attention, collision loss, and occupancy optimization each
  improve planning.

## Claim boundary for UniAD.c

The local C11 runtime implements a deterministic pedagogical analogue of the
representation flow. It does not implement the ResNet/FPN, BEVFormer,
deformable attention, query lifecycle, learned anchors, dense OccFormer,
checkpoint mapping, official preprocessing, or nuScenes evaluator. Article
figures must visually distinguish production architecture facts from the
runnable synthetic graph.
