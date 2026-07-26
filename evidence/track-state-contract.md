# TrackFormer cross-frame state contract

## Authoritative released path

This contract follows the inference branch in:

- `projects/mmdet3d_plugin/uniad/detectors/uniad_track.py`;
- `projects/mmdet3d_plugin/uniad/dense_heads/track_head_plugin/tracker.py`;
- `projects/mmdet3d_plugin/uniad/dense_heads/track_head_plugin/modules.py`;
- `projects/configs/stage2_e2e/base_e2e.py`.

The config overrides the detector constructor defaults. Production behavior is
therefore defined by:

| field | released base-E2E value |
|---|---:|
| fresh actor queries | 900 |
| SDC queries | 1 |
| public active-track capacity | 300 |
| maximum next-frame decoder queries | 1201 |
| new-ID `score_thresh` | 0.4 |
| active-result `filter_score_thresh` | 0.35 |
| miss tolerance | 5 frames |
| memory-bank length | 4 |
| memory save threshold | 0.0 |
| memory save period | 3 |
| embedding dimension | 256 |

The existing fixed-901 first-frame CUDA decoder is not a complete
cross-frame implementation. Any active track is appended after a new set of
901 queries by QueryInteractionModule, so the next decoder must accept
`901 + active_count`, bounded here at 1201.

## Official inference order

For every successful frame:

1. split persistent instances into inactive and active (`obj_idx >= 0`);
2. velocity-update active normalized reference points into the new ego frame;
3. concatenate inactive then active instances;
4. execute BEVFormer and the six-layer TrackFormer decoder;
5. compute final class logits, metric boxes, output embeddings and references;
6. set query 900 to the reserved SDC ID `-2`;
7. update IDs and miss counters:
   - score `>=0.4` resets `disappear_time`;
   - an unassigned query with score `>=0.4` receives the next global ID;
   - an assigned query below `0.35` increments its miss counter;
   - a counter reaching five retires the query to ID `-1`;
8. expose active results where ID is nonnegative and score is `>=0.35`;
9. run MemoryBank temporal attention, then periodic bank insertion;
10. select every nonnegative-ID instance, update its query embedding through
    QueryInteractionModule, and concatenate it after 901 fresh instances for
    the next frame.

## Persistent device state

The production context requires two copies: committed state and candidate
state. Each copy has fixed capacity 1201 and contains:

| tensor | shape | storage |
|---|---|---|
| query position/content | `1201×256` each | FP16 |
| reference points | `1201×3` | FP32 |
| output embedding | `1201×256` | FP16 |
| predicted boxes | `1201×10` | FP32 |
| scores | `1201` | FP32 |
| object IDs | `1201` | int32 |
| disappear counters | `1201` | uint8 or int32 |
| memory bank | `1201×4×256` | FP16 |
| memory padding mask | `1201×4` | uint8 |
| save period | `1201` | uint8 |
| valid query count | scalar | uint32 |
| next global object ID | scalar | int32 |

The SDC instance remains in the per-frame prediction tensors but is excluded
from actor ID assignment and active-track concatenation.

## Transaction boundary

All frame work writes candidate state. Committed previous BEV, queries,
memory bank, IDs, counters, scene token and frame index may change only after:

1. all five task heads complete;
2. finite-value and capacity checks pass;
3. final decoded result fits the public fixed-capacity contract;
4. the single context stream reaches the commit event.

`UA_ERR_UNSUPPORTED_PROFILE`, numerical failure, capacity failure or any CUDA
error discards candidate state. A scene-token change clears both copies,
previous/aligned BEV, global-ID counter and frame index. This rule is why the
currently dormant previous-BEV commit must not be called from the still
unsupported production entry.

## Numerical gates required before promotion

- first frame with zero active queries;
- first frame with more than 300 score-qualified candidates;
- continuation with 1, 299 and 300 active tracks;
- score exactly at 0.4 and 0.35 after FP16-logit conversion;
- five consecutive misses and retirement;
- scene switch followed by ID restart at zero;
- four memory insertions and padding-mask shift;
- QueryInteraction output for 1, 32 and 300 active tracks;
- failed-frame rollback proving byte-identical committed state;
- two interleaved scenes on distinct contexts.

No cross-frame equivalence claim is made until these fixtures and a real
continuous mini sequence pass against PyTorch.
