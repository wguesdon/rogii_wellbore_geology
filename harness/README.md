# LLM experiment harness

The harness separates context from enforcement. The JSON files and the ledger
are the compact context. The Python entry points decide what can be recorded.

| File | Purpose |
| --- | --- |
| `gate.py` | Validates a known winner, evaluates a candidate and a within well rolled null, then appends a ledger record. |
| `ensemble.py` | Uses reciprocal well splits to assess greedy base selection without reporting a best of many in sample score. |
| `breadth_gate.py` | Tracks open research axes and prevents unsupported campaign completion. |
| `forward_axes.json` | The ordered research queue, evidence, and close criteria. |
| `forward_target.json` | Objective and stop conditions. |
| `../tracking/ledger.tsv` | The gated record of candidates. |

The ROGII gate requires the competition data, prediction banks, and feature
artifacts. They are deliberately absent from this public repository. The
logic is included unchanged so it can be inspected or reused with an
appropriately structured experiment directory.
