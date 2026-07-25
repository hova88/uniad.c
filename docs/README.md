# UniAD.c: from pixels to a plan

The interactive article gives a source-guided explanation of UniAD's
planning-oriented design: how six camera views enter a temporal BEV, how
TrackFormer and MapFormer produce queries that remain useful downstream, how
MotionFormer models agent–agent, agent–map, and agent–goal interaction, how
OccFormer writes sparse agent futures back into a dense risk field, and how
Planner reduces ego intent, navigation command, BEV, and occupancy to six
future waypoints.

The repository also contains an executable C11 synthetic vertical slice. It
reduces the production representation flow to fixed capacities:
`6×3×8×8` camera planes, an `8×8×16` temporal BEV, 64 candidates reduced to a
stable top eight tracks, `3×4` multimodal futures, three occupancy frames, and
a six-point ego plan. It validates interfaces, temporal state, and the result
contract—not the numerical behavior of the released checkpoint.

Correctness cannot skip levels. Container tests establish input safety;
operator fixtures and a two-frame PyTorch oracle establish synthetic-graph
equivalence. Neither establishes production-checkpoint equivalence or nuScenes
task accuracy. Open the [interactive article](index.html) for the full
architecture study, production/tiny contract comparison, and canonical-result
visualization.
