# UniAD.c: an evidence-bounded vertical slice

This article accompanies a runnable synthetic graph that makes UniAD's
representation flow inspectable without downloading datasets or checkpoints.
The production contract is documented, but only the reduced demo executes.

The correctness ladder matters: malformed-container tests establish input
safety; operator and two-frame oracle comparisons establish synthetic graph
equivalence; neither establishes released-checkpoint equivalence or task
accuracy. See the interactive HTML article for pipeline, residency, inventory,
and canonical-result visualization.
