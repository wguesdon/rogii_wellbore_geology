# Agent instructions

Read `LEDGER.tsv`, `harness/forward_axes.json`, and
`harness/forward_target.json` before proposing work.

Run one bounded experiment at a time. Every candidate must clear the
known winner check inside `harness/gate.py` before its result is recorded.
Gate each candidate beside its within well rolled null. A candidate whose
rolled null matches its measured improvement has not earned a keep verdict.

Do not select features, bases, or hyperparameters on the same wells used to
quote the result. `harness/ensemble.py` splits wells for selection and
assessment in both directions to expose this bias.

`harness/breadth_gate.py` records whether an open model axis has a concrete
next action. It rejects unsupported campaign completion claims.
