# Candidate ledger

| Candidate | Evidence needed | Status |
|---|---|---|
| Scalar CPU reference | oracle and sanitizer agreement | implemented |
| Stable top-k | tie and tail fixtures | implemented |
| Temporal carry-over | sequential vs reset result | implemented |
| SIMD / OpenMP | representative CPU profile and unchanged oracle | deferred |
| Complete CUDA graph | every stage oracle, device residency, transfer counters | incomplete |
| Production export | full name/shape/operator map from pinned checkpoint | blocked by absent assets |
| nuScenes accuracy | official dataset/checkpoint/evaluator pairing | unclaimed |
| nuPlan | pinned usable config/checkpoint/tool chain | future work |

Optimization is deliberately downstream of measurement. No throughput claim is
derived from the synthetic graph.
