# ROGII Wellbore Geology Prediction

Final code, the experiment harness, and the Kaggle writeup for the
[ROGII Wellbore Geology Prediction competition](https://www.kaggle.com/competitions/rogii-wellbore-geology-prediction).

The final result was **131st of 6,191 teams** with a private RMSE of **7.860**.
Two selections were submitted. The multipath solution was the better private
entry. The single path solution was retained because it removed one disputed
inference block while sharing the same nine seed base.

| Selection | Public RMSE | CV RMSE | Private RMSE |
| --- | ---: | ---: | ---: |
| `solutions/multipath` | 6.818 | 7.159 | 7.860 |
| `solutions/single_path` | 6.618 | 7.211 | 7.883 |

## Repository contents

* `solutions/multipath/solution.py` is the final selected model. It averages
  two per well datum estimates in estimate space.
* `solutions/single_path/solution.py` is the other selected model. It omits
  the second datum estimate.
* `harness/` contains the LLM experiment gates used during the competition.
* `AGENTS.md` is the canonical operating contract for an agent or a human.
* `CLAUDE.md` is the Claude Code entry point and points back to `AGENTS.md`.
* `QUEUE.md` records the next bounded experiments and their commands.
* `SESSION_SUMMARY.md` is the dated competition handoff, including late
  measurements and the final selection state.
* `tracking/ledger.tsv` records every gated base and its rolled null result.
* `tracking/track_kaggle_runs.py` maintains a local SQLite record of Kaggle
  scores. The database is created locally and is not committed.
* `writeup/kaggle_writeup.md` is the Kaggle writeup copy.

No competition data, checkpoints, fitted models, or generated predictions are
committed. They are Kaggle inputs or large derived artifacts.

## Run the final inference pipeline on Kaggle

The final inference pipeline is a Kaggle kernel. Training and artifact creation happened
before the final submissions. It needs the competition source and these seven
Kaggle datasets:

```text
wguesdon/rogii-models-v6
wguesdon/rogii-div-models-v1
wguesdon/rogii-realmlp-models
wguesdon/rogii-realmlp-wheels
wguesdon/rogii-cnn1d-models
wguesdon/rogii-ratecoupled-gbdt
wguesdon/rogii-seqalt-gru-psr4avg9
```

1. Create a Kaggle script kernel from either `solutions/multipath` or
   `solutions/single_path`.
2. Copy the matching `kernel_metadata.template.json` to
   `kernel-metadata.json` and replace `YOUR_KAGGLE_USERNAME`.
3. Attach the competition source and all seven datasets above. Keep Internet
   disabled. The code installs the bundled RealMLP wheel offline.
4. Upload `solution.py` as the kernel code and run it.
5. Collect `submission.csv`, `projection_report.csv`, and
   `trust_datum_report.csv` from the output.

The templates point at datasets owned by `wguesdon`. Access to those artifact
datasets is required. A public clone can inspect the exact inference code, but
cannot reproduce the final prediction without the fitted artifacts.

## Run locally

Use Python 3.11 or later and `uv`.

```bash
uv sync
mkdir -p data/raw kaggle_datasets
make run-multipath
```

For local execution, place the competition files under `data/raw` and the
seven Kaggle dataset directories under `kaggle_datasets`. The directory names
must match the final path component of each dataset slug. The scripts search
those paths before searching `/kaggle/input`.

Use the single path selection instead with:

```bash
make run-single-path
```

## Use the experiment harness

The harness was built for a full experiment checkout with OOF grids,
prediction banks, and raw competition data. Its code is included as a release
of the method, not as a standalone generic package.

```bash
make harness-status
uv run python tracking/track_kaggle_runs.py log \
  --kernel your_name/your_kernel --version 1 --cv 7.1593 \
  --public-score 6.818 --private-score 7.860
make track-list
```

Read [`harness/README.md`](harness/README.md) and
[`AGENTS.md`](AGENTS.md) before adapting it. `AGENTS.md`, `QUEUE.md`,
`tracking/ledger.tsv`, and `SESSION_SUMMARY.md` are the original compact read
path. The first three are the operating context. The session summary preserves
the final competition handoff. `gate.py` retains the competition specific known
winner checks so that a copied experiment does not silently report a number
from a broken data layout.

## Writeup

Read the full account in [the Kaggle writeup copy](writeup/kaggle_writeup.md).
It explains the model, the selection decision, and the controls that shaped the
experiment loop.
