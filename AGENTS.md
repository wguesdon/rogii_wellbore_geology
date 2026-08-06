# ROGII Wellbore Geology

Predict TVT along horizontal wellbores. Pooled RMSE over 3,783,989 rows and 773 train wells,
lower is better. Deadline **2026-08-05**. Test is a hidden live-kernel rerun; public LB is ~26%
of it.

This file is the entry point for **any** coding agent or human working here. `CLAUDE.md` is a
pointer to it. Keep them that way: one canonical document, no second copy to drift.

## Start here, in this order

```bash
cd Competitions/rogii_wellbore_geology
make preflight        # ~3 min. Verifies env, data, grid, the deployed protocol and the PF.
make state            # ledger tail, current queue item, recent measurement commits
```

`make preflight` is the contract. It reproduces the deployed pipeline's three records and
checks the row-alignment invariant that has already caused one wrong measurement. **If it
fails, stop.** A failing KNOWN-WINNER arm means every number this checkout produces is
inadmissible, not merely suspect.

Then read, and only these: **this file**, `QUEUE.md`, `LEDGER.tsv`. That is the whole read
path. `SESSION_SUMMARY.md` carries the live session handoff and any blocking user action. Do
not read `archive/` unless a grep sends you there; it is 7,000 lines of superseded strategy
kept for provenance.

## Ground rules for working in this repo

These are not defaults you can infer from the code. Several are unusual.

**Python.** Run `.venv/bin/python` from this directory, always. A plain `uv run` pulls a
195-package CUDA torch stack. Never `pip install` in any form; dependencies go through `uv add`.
Never install into a system or global environment.

**Shell.** Never run `sudo` or `su`. If something truly needs root, print the command and stop.
Use `podman`, never `docker`. Never run `rm -rf /`, `chmod 777`, `chown`, `eval`, `dd if=`,
`mkfs`, `shutdown`, `reboot` or `systemctl`. `rsync` is allowed for pushing to machines the
owner controls, but never with any deleting flag (`--delete`, `--del`, `--delete-after`,
`--delete-excluded`).

**Git.** Work on `master` in the shared `wguesdon/Kaggle` monorepo and keep every change inside
`Competitions/rogii_wellbore_geology/`. Pull with `git pull --rebase origin master`. Never
`git push --force`, `git push -f`, `git push --delete`, `git reset --hard`, `git clean -f`,
`git branch -D`, or `git checkout -- .`. Conventional Commits (`feat:`, `fix:`, `docs:`,
`measure:`, `test:`, `refactor:`, `chore:`). Commit after each logical unit rather than
batching. Never name the agent or model in a commit message. Never commit secrets; the Kaggle
CLI reads credentials from `/home/will/Documents/Github/Kaggle/.env`.

**Code.** Save code to a file before running it; no inline `-c` snippets for anything
non-trivial. Google-style docstrings on every function and class, Sphinx/Napoleon compatible.
Google style guide generally. Output filenames are `snake_case` with a date suffix, for example
`report_2026_07_28.csv`.

**Modelling.** Never apply SMOTE or any oversampling without explicit approval; raise the
tradeoff first.

**Kaggle CLI.** `set -a && . /home/will/Documents/Github/Kaggle/.env && set +a` then
`uvx --from kaggle==2.2.2 kaggle ...`.

**SUBMITTING IS CODE-COMPETITION ONLY, and the obvious form returns 400.**
`kaggle competitions submit -f <local path>` fails with a bare `400 Client Error` that names
no cause, whatever the description length. The scored artefact is a KERNEL, so the call needs
the kernel, its version, and the OUTPUT FILE NAME:

```
kaggle competitions submit -c rogii-wellbore-geology-prediction \
  -k wguesdon/<kernel-slug> -v <version> -f submission.csv -m "..."
```

`-f submission.csv` is the name of the file the kernel WROTE, not a path on disk. This cost
three failed attempts on 2026-08-03 to rediscover because it was recorded only in
`archive/AGENTS.md` and `archive/SESSION_SUMMARY.md`.

**Machine.** 16 cores, 30 GB. Leave a core or two free; a 15-worker pool plus another job puts
load above 25 and starves both.

**Claude Code continuation.** Claude Code can pick up this checkout without chat history. Run
`make claude-handoff` after `make preflight`; it writes a dated brief under `scratchpad/` with
the current axes, ledger tail, AWS trials, and recent commits. The brief is a supplement to this
file, not a replacement. A continuation agent must poll existing AWS jobs before launching more,
must not run the broad `make ensemble`, and must update `SESSION_SUMMARY.md` before ending.

## What we have

Updated 2026-08-05. CV is pooled corrected+ramped for the 2026-08-04/05 rows and the frozen
leave-one-fold-out number for older ones; the two differ by about +0.006 and are labelled where it
matters.

| submission | what | CV | public LB |
|---|---|---:|---:|
| **SELECTED** `55244616` | 9-seed psr4 base + estimate-space datum averaging | **7.1593** | 6.818 |
| **SELECTED** `55244617` | 9-seed psr4 base, SINGLE path, no multipath | 7.2113 | **6.618** |
| `55234752` | 3-seed base + estimate-space datum averaging | 7.2038 | 6.891 |
| `55231323` | + 3-seed average of the psr4 base | 7.2732 | 6.731 |
| `55223770` | + post-processing retune (0.88 / deg 6 / 0.78) | 7.37230 | 6.992 |
| `55221568` | setK-keeponly + psr4 compound + ramp | 7.5228 | 6.888 |
| `55218515` | + prediction-start resampling x2 | 7.68599 | 7.064 |
| `55216428` | + shape-supervised sequence loss | 7.82033 | 7.616 |
| `55210028` | setK-keeponly + ramp | 7.93299 | 7.848 |
| `55209717` | setK-keeponly + trust datum | 7.9772 | 7.796 |
| `55184703` | the 2026-08-02 best | 8.1326 | 7.581 |

**The deployed pipeline, as of 2026-08-04.** Three constants moved from the version below.

```
positive Ridge (alpha 1.0), cross-fit GroupKFold(5) by well, over 7 base OOFs
  -> last_known + 0.88*stack + 0.12*pf_selector          (W_STACK was 0.76)
  -> savgol(17, 3) per well
  -> anchored robust degree-6 U-projection, blend 0.78   (was degree 4, blend 0.70)
  -> trust-datum per-well correction, bank REBUILT per path (rule 13)
  -> ramp: pred + shift*(lambda + mu*(md - md.min())/5000), frozen per path
```

`harness/gate.py` still carries the OLD cell (0.76 / degree 4 / 0.70) and reproduces the historic
records with it. Do not change it: it is the anchor every banked number was measured against.
`scripts/verify_postproc_retune_2026_08_04.py` parameterises the cell and is asserted to reproduce
`Data.postproc` at 0.0 ft before it is used.

**Where the LB stands, downloaded 2026-08-05 09:54 UTC at 6,152 teams.** Gold top 22 at 5.518,
silver top 307 at 6.350, bronze top 615 at **6.400**. Our **6.618 is rank 1,824**, with 1,175 teams
in [6.405, 6.618). Inverting the REFITTED slope in rule 1 (1.1711, not the older 1.674), bronze
needs CV about **6.92**, not the 7.15 the earlier slope implied.

**THE PUBLIC AND PRIVATE WELLS ARE DISJOINT.** The leaderboard banner states the board is
"approximately 26% of the test data" and the final result is based on "the OTHER 74%", quoted by
Pavel in discussion 731550 on 2026-08-05 and uncontested. **Our 6.618 contributes nothing to the
private score.** Everything the board said about any component was measured on wells that will not
be scored.

**Private-draw noise, measured two ways and agreeing.** A rank-4 competitor published a well-subset
bootstrap on his own model: at 200 wells the 68 percent band is +/-0.415 ft and the 90 percent band
+/-0.664, around a population of 5.1362. Our own bootstrap of the shipped path gives +/-0.596 and
+/-0.972 around 7.1593. As a FRACTION of level those are 8.09 and 8.32 percent — the ratio of levels
(1.394) matches the ratio of half-widths (1.434). **Our relative draw variance already equals a
rank-4 solution's; our band is wider only because our error is bigger.** Shakeup exposure is driven
by well-level error concentration, not by stack diversity: the worst 1 percent of wells carry 18.7
percent of our squared error and the worst 5 percent carry 43.5.

**SELECTION IS MADE AND FINAL, 2026-08-05: `55244616` (public 6.818) and `55244617` (6.618).**
The owner selected them on the web UI; the CLI exposes no command for it, so it is always an owner
action. They are the SAME nine seeds with and without estimate-space datum averaging, which is the
one component whose sign is disputed: CV says it gains 0.05-0.07 at p_worse 0.0053 over 773
cross-fitted wells, the public board says it costs about 0.18 ft at two separate seed counts.

**The pair is NOT a variance hedge and must not be described as one.** Their per-well squared errors
correlate **0.9973**, so they diversify the well draw essentially not at all: E[min] 7.1336 against
E[multipath] 7.1363, a hedge worth -0.0027 ft, P(multipath better) 0.877. What it hedges is the
multipath block being SYSTEMATICALLY wrong on unseen wells, a model risk no well-bootstrap can see
because the bootstrap holds both models' per-well errors fixed and resamples only which wells
appear. `reports/probe_stack_diversity_shakeup_2026_08_05.json`.

**The earlier selection of `55221568` + `55223770` is superseded.** Both are dominated on both axes.

The pair was chosen under the objective Kaggle actually scores, and the rule this repo used before
was answering a different question. Kaggle scores each selected submission on the private set
independently and takes the BETTER one, so the objective is `E[min(RMSE_A, RMSE_B)]` over the
private draw, not `RMSE((A+B)/2)`. The 2026-08-01 criterion, lowest error correlation plus "the
only pair whose half average beats both members", is the figure of merit for a BLEND.
`scripts/probe_pick_pair_maxof2_2026_08_04.py` simulates 4,000 well-cluster bootstraps of 200 wells,
the host's stated private size, and the decorrelation rule INVERTS under it: the most decorrelated
candidate available (`geom_ramp`, err-corr 0.892) buys the LEAST (-0.0010) because at 8.139 it wins
only 1.3 percent of draws, while the closest competitor at err-corr 0.976 buys the most. A partner
helps only by winning draws, so closeness in quality dominates decorrelation.

**The second pick is worth at most 0.008 ft.** Every pairing with the best single lands between
-0.0010 and -0.0079. So the second slot is nearly free and nearly worthless on modelling grounds,
and it should be spent on the risk that actually exists: a kernel failing on the rerun. That is why
both picks must have RETURNED a public number, which is the only proof a kernel executes on hidden
data.

**Do not select `55216428`.** The shared dataset slug `rogii-seqalt-gru-models` was versioned twice
on 2026-08-03, so it now holds `psr2`'s checkpoints. A rerun of `55218515` loads exactly what it was
verified against; a rerun of `55216428` would load weights it was not. Both current picks read
`rogii-seqalt-gru-cmp4e160`, which has a single version. **Never re-version a slug that a selected
submission reads.**

What does hold, from CV rather than the board: the correction TRANSFERS across pipelines, giving
-0.1069 on the geom arm with frozen constants, and correcting both picks leaves their error
correlation at 0.9263 against 0.9305, so a corrected pair keeps the decorrelation that chose the
original pair while lifting both members.

**Selection was settled as of 2026-08-01 and both picks are scored.** `55157342` is the seven base
portable Ridge in which the rate coupled LightGBM replaces the older rate posterior LightGBM; it is
the best honest CV we own and carries no geom prior. `54791000` is retained as the hedge, not as a
primary: its CV edge is largely field leakage, field blocked CV retains 2% of it. See the two-pick
endgame section for the decorrelation measurement that chose this pair over the alternatives.

Rank 2377 of 5714. Bronze is LB 6.479, silver 6.444, gold 5.899.

**The deployed pipeline.** Every CV number in this repo is measured on exactly this:

```
positive Ridge (alpha 1.0), cross-fit over GroupKFold(5) by well, over the base OOFs
  -> last_known + 0.76*stack + 0.24*pf_selector
  -> savgol(17, 3) per well
  -> anchored robust degree-4 U-projection, blend 0.70, C 3.0
```

`harness/gate.py` implements it and reproduces 8.2826, 8.7300 and 8.9641 as its KNOWN-WINNER
arm. Import it; do not reimplement it.

## What actually moved CV, 2026-08-03 to 04: 8.2251 -> 7.2732

Five changes, -0.95 ft. **Every one came from noticing a component was optimising or consuming the
wrong quantity. None came from more capacity, more bases, or more blending**, and recombination
over all 60 banked bases is capped at 7.9537 in sample, so none of it could have.

| change | worth | what was wrong |
|---|---:|---|
| shape-supervised sequence loss | -0.11 | `compute_loss` weighted absolute TVT at 1.0, and 58.6 percent of absolute TVT is the per-well datum `src/trust_datum.py` corrects downstream, so most of the gradient went to a quantity another stage fixes |
| `gr_filter` 50 -> 9 | -0.07 | the horizontal GR was savgol-smoothed at 50 ft while the typewell it is cross-attended against sits at 1 ft unsmoothed, and that cross-attention IS the matcher |
| **virtual prediction-start resampling** | **-0.25** | the prefix/eval split is derived purely from which rows carry `TVT_input`, and train wells carry `TVT` everywhere, so the boundary of a TRAINING well can be re-cut anywhere. 773 fixed samples were an arbitrary choice, not a constraint |
| post-processing retune | -0.14 | the degree-4 blend-0.70 U-projection was MIS-CONDITIONING the path for the datum estimator |
| 3-seed averaging | -0.10 | see below; it pays about fifteen times its usual value here |

**Prediction-start resampling, the mechanism.** Cuts are drawn per epoch from the MEASURED real
known-fraction distribution (median 0.2600, p10 0.1967, p90 0.3464) — matching the real
distribution is what the synthetic-well bank failed to do. A well sees a median 61 distinct cuts
over 80 epochs. Dose-response against the within-well-rolled null: psr0 control -0.0861, psr1
-0.1874, psr2 -0.2439, psr4 -0.3278. An adversarial leak audit returned NO LEAK: rebuilding a
validation well's inputs from the raw CSV with `TVT` STRIPPED reproduces all ten tensors and the
checkpoint output at 0.0, one well per fold. The truncation hypothesis was refuted in the strongest
available direction — the 9 wells overflowing the 9,216-row future capacity improve MORE than the
rest, -3.69 against -1.66.

**Why seed averaging pays fifteen times what it paid the CNN family (-0.0102).** The three seeds
correlate **0.7706 to 0.8039** with each other, against 0.8893 to 0.9176 for same-recipe seeds
elsewhere here. Resampling draws different cuts per seed, so it DECORRELATES the seeds as well as
augmenting them, and augmentation and averaging compound rather than overlap. This is a reusable
fact about this kind of augmentation, not a fact about this competition.

**The retune's mechanism, because it explains an anomaly that looked like a bug.** The correction's
worth had FALLEN as the path improved, -0.2052 to -0.1079, which is backwards from the emission
dose-response curve. It was not stale `trust_datum` constants. Under the new cell the correction
gets STRONGER, -0.1079 to -0.1678, and the per-well datum correlation rises 0.2298 to 0.2777, which
lands on the 0.28 identifiability cap the GR channel was independently closed at four ways. So the
retune recovers a known ceiling rather than inventing headroom, and its gain runs through rho rather
than through error reduction: shape moves only 4.9747 to 4.9439.

## Saturated or closed on this axis, do not re-propose

* **The resampling axis, both knobs.** `psr2@160` -0.3603, `psr4@160` -0.3755, `psr8@80` -0.3744,
  `psr4@240` -0.3753 on real-minus-rolled-null: a 0.015 ft span against a 0.051 ft seed spread.
  `psr16`, width 160 and `gr25` are flat or worse. The epochs-versus-cuts confound resolves as
  BOTH saturated.
* **Cross-recipe averaging.** Nine banked psr arms in nested pools; best is -0.0204 against the
  seed-3 average, below the 0.05 floor. The diagnostic: mean pairwise error correlation is FLAT at
  0.787 to 0.801 across every pool, including pools mixing psr2/psr4/psr8, three schedule lengths,
  two widths and two smoothing levels. **Different recipes do not decorrelate further than
  different seeds of one recipe.** The diversity is the training draw, not the recipe, so the
  marginal GPU hour belongs to another SEED.
* **The datum aggregator.** Four robust consensus estimators against the incumbent
  isolation-weighted mean: median 0.2323, vote 0.1756, KDE mode 0.1697, mixture MAP 0.1509, summed
  profiles 0.0626, all against 0.2772, and cross-fitted constants push every one back toward a
  mean in 5 of 5 folds. The motivating argument is recorded as WRONG: rho is scale-invariant, so
  dilution cannot attenuate it. The mean earns its keep by 1/sqrt(N) averaging, 0.1011 at one
  window to 0.2759 at all ~25, already saturated.
* **Re-selecting the post-processing cell nested by well is DESTRUCTIVE**, +0.109 against always
  using the frozen cell, which sits rank 6 of 37 and within 0.019 ft of the argmin. Freeze it.
* **Synthetic pre-training is blocked, not closed.** Every generated bank roughens its own template
  well's anchored degree-1 `U` residual by 1.67x to 1.73x on 80 to 83 percent of wells, corrupting
  the exact term shape supervision won on. Unblock condition: a bank whose paired ratio against its
  own template is at or below 1.25. The LEAK worry does not hold and was measured: `data/synth`
  passes Gate B at slope R^2 -0.006 against the real 0.060.

## How to make a claim here

This project has produced at least five confidently wrong results, each caught by a control it
had not run. The rules below are what caught them.

1. **Rank by CV. Use the board to price a target and to prove a kernel executes, never to rank.**
   REVISED 2026-08-04, and the previous version of this rule was wrong in a way worth
   understanding. It said the CV-to-LB correlation is *negative* across five scored submissions.
   Those five differed only in correction-stack constants and spanned 0.20 ft of CV, against a
   board whose sd on a DIFFERENCE is 0.1613 ft — the spread was smaller than the instrument, so
   the sign was noise being read as signal.

   **REFITTED 2026-08-04 evening on NINE paired points, and the slope has fallen twice.** It read
   1.674 at six points, 1.5818 at seven, and now:

   ```
   board = 1.1711 * CV - 1.7009     r = 0.8912,  residual sd 0.2119 ft
   ```

   A foot of CV now buys 1.17 ft of board, not 1.67. **Treat the slope as unstable and do not
   quote it to three digits.** It has moved 30 percent as four points were added, which is what a
   regression over a 0.93 ft span with 0.21 ft of residual noise does. Consequences: bronze at
   6.408 implies CV about **6.92**, harder than the 7.15 the earlier slope implied, and any
   target priced off an older slope is optimistic.
   `scripts/probe_cv_to_lb_refit_2026_08_04_pm.py`.

   **The paired form is the only board test with resolving power, and it has now been used in
   anger.** `55234752` is `55231323` plus exactly one block, estimate-space datum averaging. CV
   said -0.0694; the fit predicted -0.0897 of board; the board delivered **+0.1600**. That is a
   +0.2497 ft discrepancy, **+1.55 sd** of the board's 0.1613 ft difference noise, and the
   leave-one-out residual of `55234752` against the other eight points is +1.10 sd, smaller than
   four of them. The pre-registered port-bug test therefore returns NOT DISTINGUISHABLE FROM BOARD
   NOISE, and the kernel log independently confirms the block executes: both Ridges load their
   correct coefficients and the two matcher passes produce genuinely different shifts, 1.069 ft
   and 0.829 ft. **A block whose CV gain is real at p_worse 0.005 can still land 1.5 sd the wrong
   way on 52 wells. That is the instrument, not the model.**

   What this does and does not license. It DOES let you price a target: bronze at 6.409 needs CV
   about 7.15, and Chris Deotte's rank-100 6.112 needs about 6.97. It does NOT let you select on
   the board, because the residual sd of 0.192 ft is comparable to the gaps between our own
   candidates and the absolute level carries sd 1.52 ft on 52 wells. A returned board number is
   also the only proof a kernel EXECUTES on hidden data, which is why a submission that has not
   returned one must never be selected.
2. **A base earns its place by DECORRELATING, not by standalone RMSE.** Gate on blend-add and
   err-corr against the deployed blend. Judging on standalone RMSE caused at least three wrong
   kills here.
3. **Ship a control in the same run, and pick the right one.**
   - For a **base** entering the Ridge: a within-well-rolled copy, seeded from
     `int.from_bytes(well_id.encode())`, never `abs(hash())` (Python's hash is salted per
     process, so an `abs(hash())` control is not reproducible). `harness/gate.py` does this
     automatically. It turned `[UNTRIED]` #19 from a -0.13 headline into a -0.0034 wash.
   - For a **per-well estimator**: the cross-fitted **global scalar**, not a permutation. A
     permuted per-well factor is actively destructive (+0.39), so every arm scores a spurious
     KEEP against it. Measured 2026-07-28.
   - For a **PF branch replacement**: the deployed filter **rebuilt through the identical
     reconstruction**, never the banked array, because a rebuild also changes the seed count
     and re-derives the beam and selector bins. `scripts/gate_pf_swap_2026_07_28.py`.
4. **Cross-fit everything, including the link.** Any fit is cross-fitted over the same
   `GroupKFold(5)` by well that produces the stack OOF, and any feature *selection* happens
   INSIDE the training fold. On 2026-07-28 three arms fitted a link in-sample and reported
   gains that do not exist: isotonic-on-`N` claimed -0.356 and is +0.005 cross-fitted;
   degree-5 claimed -0.617 and diverges to +118. Isotonic looks safe and is not.
5. **Report an interval, not a point.** Well-cluster bootstrap over the 773 wells, plus the
   five per-fold deltas. A gain carried by one fold of 155 wells is a much weaker claim than a
   broad one.
6. **Report the honest number even when it is a wash.** A clean kill with a reason is a result
   and belongs in a `measure:` commit. Most of this repo's value is its negative results.
7. **A bar must specify MAGNITUDE, not just direction and a control.** On 2026-08-02 the
   neighbour curvature field passed all three of its pre-registered bars — shape below the
   incumbent, real minus shuffled -0.0538 past the 0.05 floor, and the field carrying real weight
   in 5 of 5 folds — and was worth **0.0035 ft**. The bars asked whether the effect was real and
   never how big. Every pre-registration must name the smallest gain worth having, and this CV
   cannot resolve a per-well correction below ~0.10 ft.
8. **A selection function must never see the value it is selecting on.** A learned window
   reliability model reached out-of-fold AUC 0.826 and DESTROYED the estimator, taking the
   per-well datum correlation from 0.2718 to 0.1472, because `argmax` was a feature and windows
   whose shift is near zero really are likelier to be right when the truth clusters near zero. It
   learned "small shift is reliable" and selected away the signal.
9. **The proxy is not the objective.** The same model, made position-blind, still ranked
   individual windows far better than the hand-designed statistic at AUC 0.8116 and still produced
   a WORSE per-well estimate, 8.1296 against 8.0435. An aggregate needs component errors that are
   unbiased and roughly independent, which is not the same as components that are individually
   most often correct. Optimise the thing you are actually going to report.
10. **Compare like with like, especially in a ratio.** A structure-function argument on 2026-08-02
    compared the error in TVT space against the truth in U space. Since `TVT = U - Z` with `Z`
    known exactly, the error is identical in both spaces while the truth's short-lag variation in
    U includes the survey trajectory we already know, so the denominator was inflated and the
    conclusion inverted. It cost most of a session and a wrong strategic pivot.
11. **Do not summarise a profile or variogram by averaging bins past its range.** Two bars on
    2026-08-02 were mis-specified this way, one reporting a real spatial structure as absent
    because the third bin lay beyond a ~5,000 ft range. Read per bin, against per-bin controls.
12. **Read the DATA DESCRIPTION before mining the forum or public notebooks.** On 2026-08-02 a
    submission was spent confirming that the train/test well overlap is inert on the scored set,
    which the official dataset description states outright: the visible `test/` folder holds "only
    a few instances from the training set as example data" and they "will be replaced with the
    actual test data", about 200 wells. The repo had already inferred it twice.

13. **Judge a base set CORRECTED, never uncorrected.** On 2026-08-03 setK beat setK-keeponly
    uncorrected, 8.1570 against 8.1824, and LOST to it corrected, 7.9811 against 7.9772, because
    the datum correction is worth -0.1759 on one path and -0.2052 on the other. The correction's
    value depends on the shape of the path it is built around, so an uncorrected ranking can
    invert. Rebuild each candidate's own profile bank; borrowing the incumbent's estimates a shift
    for one trajectory and applies it to another.

13a. **Improving the RAW stack systematically degrades the datum correction, and usually by more
    than it gains.** Measured three independent times on 2026-08-03, with the correction's value
    tracking how much the new base displaces `cnn_1d_v1_avg3` in the Ridge:

    | change | uncorrected gain | corrected gain | correction worth |
    |---|---:|---:|---:|
    | shipped `setK-keeponly` | — | — | -0.2052 |
    | window-evidence base | -0.0216 | -0.0039 WORSE | — |
    | `seqalt_gru_v1_s42` added | -0.0599 | -0.0165 | -0.1617 |
    | `seqalt_gru_v1_s42` swapped for the CNN | -0.0648 | -0.0010 | -0.1414 |
    | `gbdtdiv_lgb_huber4` | -0.0618 | **+0.0026** | -0.1408 |

    The mechanism is measured, not assumed: the full-fit Ridge gives the GRU 0.1936 and collapses
    `cnn_1d_v1_avg3` from 0.1836 to 0.0223, and the correction weakens in proportion. The CNN's
    contribution to the path SHAPE is what lets the trust-gated typewell matcher localise. A base
    that wins raw error by taking the CNN's weight destroys the thing the correction depends on.
    So: never judge a base on the raw stack, and expect the inversion specifically when the
    candidate competes with `cnn_1d_v1_avg3`.

13b. **Anything capturing per-well LINEAR structure competes with the ramp rather than adding to
    it.** The ramp, `est*(0.2 + 1.7*x)`, and a sequence base that models per-well drift reach the
    same signal by different routes. Taking both keeps only the weaker: 7.9341 with the GRU plus
    ramp against 7.9325 for the ramp alone. Price any such proposal against the RAMPED path.

13c. **A cheaply trained slot injects more CV noise than any lever moves it, and the distinction
    between two kinds of comparison decides what that invalidates.** Measured 2026-08-03 over four
    seeds per slot:

    | slot | recipe | seed spread | 4-seed bag | bag minus mean-single |
    |---|---|---:|---:|---:|
    | `lgbdivmed` | 5000 trees, full rows | **0.006** | 8.0412 | -0.0010 |
    | ratecoupled | row-step 4, 1200 trees | **0.055** | 8.0417 | -0.0216 |

    0.055 ft is wider than the W_STACK point estimate (0.044), five times seed averaging itself
    (0.010) and twenty times the fitter arms. But it is variance across RETRAININGS of one recipe,
    so it invalidates only comparisons that resample a seed. Comparisons between BANKED artefacts —
    ramp versus no ramp, one base set versus another, post-processing changes — hold every seed
    fixed and are unaffected. What it does bound is how well an absolute CV predicts a fresh fit,
    and therefore the private set. Quote it whenever an absolute number is used as a forecast.

    A retraction is attached: on n=1 this repo briefly recorded that the deployed ratecoupled seed
    was "a lucky draw". With all four seeds it is SECOND of four, above the seed mean by about
    0.015 ft. Overstated from one sample and corrected here.

14. **In-sample |rho| is the honest univariate ceiling; a cross-fitted linear fit is NOT.** On
    2026-08-03 a screen reported null features at cross-fitted -0.187 when their in-sample rho was
    +0.002. That is structural: with total covariance near zero, the training-fold covariance is
    the NEGATIVE of the validation-fold covariance, so an out-of-fold fit drags a null feature to
    about -0.15 by construction. Report both, and compute a permutation multiple-testing floor for
    the number of features actually screened. Two screens this session used floors of 0.1124 over
    132 features and 0.1342 over 217.

15. **Nest feature selection inside the fold or pay about 40 percent of the headline.** The slope
    screen gave -0.0534 when features were ranked on all 773 wells and the Ridge then cross-fitted,
    and -0.0298 with selection nested. Same data, same model, same folds.

16. **A per-well estimate must be shrunk by a cross-fitted global scalar before it is applied.**
    Applying raw ridge coefficients from the trajectory probe made CV WORSE by up to +0.2673 ft: an
    estimate at correlation 0.05 with order-one amplitude injects variance instead of removing it.
    `SHIPPED_LAMBDA = 0.875` in `src/trust_datum.py` is that scalar for the datum, and refitting
    even it per fold cost +0.0115 against leaving it frozen.

## The numbers that govern strategy

**Recombination is capped.** An unconstrained least-squares fit of all 60 banked bases to the
truth, in sample with hindsight, gives 7.9537. Evaluated honestly it gives 8.43, worse than
three bases. No amount of stacking, weighting, shrinking or base-selecting over what exists
goes below 8. `reports/axis_regate_pilot_leak_2026_07_26.txt`.

**Bronze needs a better base model, not a better blend.** Calibrated against five scored CV/LB
pairs (no-geom offset -1.221, spread -0.976 to -1.388), bronze is roughly CV 7.7, which sits
below the in-sample ceiling of everything we own. Treat that extrapolation as an order of
magnitude only; those points have negative CV-to-LB correlation.

**This CV cannot resolve a per-well correction below ~0.10 ft.** Under the curvature weighting
the pooled loss applies, the 773 wells have a Kish effective N of 205 on the no-geom axis and
158 on Pick-2's, and every per-well arm measured on 2026-07-28 had a cluster-bootstrap sd near
0.05 ft. Four arms landed under that floor and none shipped.

**A competitor's single pure-physics model scores CV 6.85** on our exact pooled metric, against
our five-base stack at 8.6082 and our own PF at 10.3611. The deficit is the forward model.

**The public frontier is REAL modelling, confirmed 2026-08-02.** The top public notebook,
`tamerlanomralinov/hahaha-det-agi`, scores 6.42, above bronze. It carries the
`_gold_contact_candidate` train/test lookup, but the data description says the visible `test/`
wells are examples replaced at rerun by ~200 actual test wells, so that lookup is INERT on the
scored set and its 6.42 is honest. There is no shortcut being exploited by the people ahead of us,
and the gap is a real modelling gap.

**Where the error actually is, measured over all 3,783,989 rows.** Datum 6.2943 at 58.6 percent of
the variance, now solved to weighted correlation 0.27 and worth -0.18 ft; linear shape 3.7795, a
per-well dip residual that every observable predicts at only 0.196; non-linear shape 3.7081, of
which the quadratic slice is reachable from neighbours but worth 0.0035 ft. Content below 400 ft of
MD is 0.57 percent of the total, so short-scale proposals target almost nothing.

**The error is a RAMP, and the datum and slope are ONE defect.** Measured 2026-08-03: the per-well
constant and the per-well slope correlate at **+0.7640**, 58.4 percent shared variance. The
prediction anchors at `last_known` and drifts away roughly linearly. So the datum's 56 percent and
the slope's 22 percent of squared error are not additive prizes, and a method that attacks one is
attacking most of the other. A corollary that is free: the deployed correction applies its per-well
estimate in the wrong SHAPE, as a constant, and re-shaping it as `est*(lambda + mu*x)` is worth
-0.0132 with `mu >= 0.8` in all five folds.

**rho against the per-well datum is the headline metric for datum work, and it converts exactly.**
The residual datum term is `6.2943*sqrt(1-rho^2)` and the shape terms are fixed at 28.03 ft^2, so
rho 0.2718 -> CV 8.04, 0.50 -> 7.60, 0.70 -> 6.94, 0.90 -> 5.96, 1.00 -> 5.2948. Quote rho, not a
CV delta, when reporting a datum estimator.

**The datum channel is CAPPED near rho 0.28 and the cap is geometric, not architectural.** Three
independent attacks converged on 2026-08-03. Ten matching statistics: none beats the incumbent
cross-correlation at 0.2762. A best-window oracle: 0.9094 on real profiles against 0.8925 on ROLLED
ones, so believable selection headroom is +0.0168. Learned pooling over 31 evidence features: 0.2141
against the hand-built 0.2802. The mechanism is identifiability: hit-within-1-ft runs 7.0 / 12.6 /
24.4 percent from low to high TVT travel across a window with the rolled control flat at 5.8, so a
window where the well barely moves sees one bed and cannot localise against a log whose beds repeat.
About a sixth of windows carry literally zero signal. **Crossing formation MANUFACTURES the
evidence; it does not smear it.**

**A FOURTH attack, 2026-08-03, and it also fails: the AGGREGATOR is not the problem.** The
readout collapses trusted windows to one number per well with an isolation-weighted MEAN, which
looks like the wrong estimator for a mixture that is mostly outliers. Four robust alternatives
were built and cross-fitted, and every one LOST: incumbent mean rho 0.2772 (bootstrap
[0.212, 0.345]), median 0.2323, +/-2 ft vote 0.1756, KDE mode 0.1697, mixture MAP 0.1509, summed
raw profiles then argmax 0.0626. Cross-fitted constant selection is the tell: the KDE bandwidth
picked 3.0 ft in all five folds and the mixture picked `pi = 0.5` in all five, i.e. every fold
pushed its estimator back toward a mean. Replicated on a second, independently built profile bank
(incumbent 0.2872, best consensus 0.2241). `reports/probe_datum_consensus_2026_08_03.json`.

The reasoning that motivated it is also wrong and is recorded so it is not repeated: **rho is
scale-invariant, so diluting an estimate toward zero cannot attenuate its correlation with the
truth.** Attenuation requires added noise, not shrinkage. What the mean actually buys is
averaging: rho against the number of windows averaged runs 0.1011 (k=1), 0.1792 (k=4), 0.2309
(k=8), 0.2600 (k=16), 0.2753 (k=32), 0.2759 (all) — a 1/sqrt(N) curve that has already saturated
at our median 25 windows per well.

**CORRECTED 2026-08-03: there is NO ~13 ft alias lattice, and the line below said there was.** Two
independent tests on the banked profiles, neither eyeballed. (1) The histogram of window peak minus
true datum falls monotonically, and at 12-14 ft the real/rolled ratio is **0.73**, i.e. DEPLETED
relative to a control carrying no datum at all; the apparent fall-off is the +/-16 ft search-span
envelope, which the rolled arm reproduces exactly. (2) Reading competing peaks directly off each
profile, the distance from the global peak to the best rival outside 4 ft is FLAT: 1160 / 1097 /
1248 / 1268 / 1144 / 1100 windows per 2 ft bin from 4 ft to 16 ft, and no 0.5 ft bin holds more
than 2 percent. The real structure is a genuine spike inside +/-4 ft, real/rolled ratio 1.9, on a
near-uniform background at an 11 percent inlier rate. With 25 windows per well the inlier pile
never clears the background pile, which is why a mode has nothing to lock onto and a mean still
gets its 1/sqrt(N). The cap is real; the lattice explanation for it was not.

**Consequence for strategy, stated plainly.** Pooled 5.444 requires rho about 0.97 on the per-well
datum. Our own best-window ORACLE reaches 0.9094 on real profiles and 0.8925 on ROLLED ones, so
almost all of that is the free score from choosing among candidates rather than signal. The teams
ahead of us are therefore NOT reading this channel better. Either their evidence is different or
their error decomposition is not our 6.12 datum and 5.21 shape.

**Global alignment is closed with an explicit reopening bar: stack shape RMS below ~0.6 ft.** The
lateral emission is sound, prefix-calibrated typewell localising within 1 ft on 83.6 percent of
wells at contrast 0.667 against a rolled 6.4 percent at NEGATIVE contrast. But tracing rho against
shape error with the datum held fixed puts the crossover with the gated-window estimator at about
0.6 ft, and we are at 5.10. The gate wins because it SELECTS the minority of windows whose local
shape error is small; a global fit has no equivalent and one bad stretch biases its one coefficient.

**Reported CVs near 5 are not comparable without knowing the protocol.** Our UNCHANGED model scores
4.5162 on the first 30 percent of the eval span, 5.0674 on the first 40, and **5.2948 under a
row-wise holdout** where the well's own labels pin the per-well constant — identical to the oracle
datum ceiling, because that is exactly what such a holdout hands over. This is calibration for forum
numbers, never reassurance: our own 2.9303 ft of datum prize is real and we hold 6.2 percent.

## Why the gap is what it is

Our own oracles, on our own OOF: a perfect per-well **constant datum** scores **5.31**; datum
plus slope scores **3.74**. The wells are solvable and two thirds of the tail variance is one
number per well, its vertical datum.

We predict that number by regressing per-row tabular features. Every sub-5.5 team instead
matches the horizontal well's GR log against its typewell GR log to *locate* the well
vertically, and the datum falls out of the alignment. Tucker Arrants reaches LB 5.444 with a
single model and no neighbour data.

Our matcher does not work. From GR alone it extracts drift at correlation ~0.06 and runs away.
The objective's capture radius is ~1 ft (template correlation 0.728 at 0 ft, 0.326 at 1 ft,
noise beyond 2 ft), so a proposal 2 ft off reads the objective's own noise floor. Roughly 40
documented attempts have failed here.

**SETTLED 2026-08-02, read this before proposing any matcher, decoder or reference work.** Six
statements, each measured with a control, that between them explain every result on this axis:

1. **The emission is a sound DISCRIMINATOR over a low-dimensional family and an unsound OBJECTIVE
   for free path search.** One parameter, a rigid shift, and the truth wins on 76 percent of
   wells. Two parameters and there is no signal either way, twice. A Viterbi over 25 nodes at 65
   cells, about `65^25` paths, and the truth loses on 153 of 153. The objective's FORM is not what
   breaks; the DIMENSION OF THE SEARCH is.
2. **Freedom that can MOVE the implied TVT is harmful; freedom orthogonal to TVT is necessary.**
   Per-node TVT gradients let a decoder explain GR at a wrong depth by bending geometry. The
   window emission's per-window affine GAIN refit cannot shift a peak at all, only absorb real
   tool and facies variation, and removing it costs everything: correlation -0.0441 and CV 8.2466,
   worse than doing nothing.
3. **CONTRAST between the truth and nearby WRONG offsets is the figure of merit for a reference
   log, not oracle R^2 at the truth.** They demonstrably come apart: the well's own prefix beats
   the typewell on oracle R^2, 0.4560 to 0.4147, and loses on window hit rate, 0.143-0.204 against
   0.191-0.261. Following this lesson found a 2.5 ft boxcar smoothing across a 1-2 ft correlation
   length, whose removal doubled the gain.
4. **The estimator wants MAXIMUM resolution on both logs.** Reference unsmoothed at a 0.5 ft grid,
   query raw. Smoothing either destroys the discriminating content with no compensating noise
   reduction, because the window R^2 already averages hundreds of rows.
5. **The information is there; the deficit is IDENTIFIABILITY.** The Cramer-Rao bound on TVT from
   GR is 0.252 ft at a 400 ft window against 3.781 ft of truth variation at that lag, a fifteenfold
   margin, discounted honestly for the residual's own 24 ft decorrelation length. **The clause
   below about a "~13 ft alias lattice" is RETRACTED, measured 2026-08-03; competing peaks are
   spread near-uniformly from 4 to 16 ft and the 12-14 ft band is DEPLETED against its rolled
   control at a real/rolled ratio of 0.73. The identifiability deficit is real, its cause is an
   11 percent inlier rate against a near-uniform background, and it is not periodic.** The correct peak
   is sharp. Choosing WHICH peak is the problem, with a ~13 ft alias lattice against a ~6.3 ft
   datum error.
6. **The train/test overlap is INERT on the scored set and the public frontier is REAL
   modelling.** The data description says the visible `test/` wells are examples replaced at rerun
   by ~200 actual test wells. So `hahaha-det-agi`'s public 6.42 is honest modelling, not a leak,
   and the gap between it and us is a real gap. There is no shortcut to buy.

**CORRECTED 2026-08-02. The emission is NOT the problem, and this file said otherwise for two
weeks.** The line below about a ~10% inlier rate, and `QUEUE.md`'s "the deficit is the
observable, not the decoder", are both refuted. Scored at the TRUE path with the row set held
fixed, the supplied typewell explains a median 0.43 of eval GR variance and its datum argmax
lands within **1 ft of the truth on 76 percent of wells**, contrast 0.3688 CI [0.3051, 0.4253],
against 0.0012 for a rolled typewell and 0.0381 for a foreign one. That is the exact inverse of
the 90-of-90 Viterbi certificate and the first positive emission certificate this campaign has
produced.

What the emission cannot do is give a DIRECTION from a wrong path, and the reason is now a
measured curve rather than a guess. Localisation decays monotonically with the shape error fed
in: 76.4% within 1 ft at 0 ft of shape error, 54.0% at 1.23 ft, 34.4% at 2.46 ft, and **18.4% at
our 4.91 ft**, with the rolled control flat at 0.001 to 0.007 throughout. **Any mechanism that
delivers a path at 1 to 2 ft of shape error makes the datum, 6.2943 of our 8.2251, nearly
free.** That is the governing number for this axis; read
`notes/emission_certificate_2026_08_02.md` before proposing any matcher work.

**Still open, one thread.** The 2026-07-27 audit repaired `[UNTRIED]` #4's emission to sit
inside the capture radius and its alias log likelihood ratio went from -0.023 to
**+0.3003 +/- 0.0465 bits** against a rolled null at -0.997, Wilson disjoint. The direct E2
model, its fixed rate posterior, and its rate coupled state model are now measured. Only the
rate coupled residual earns a place, through LightGBM rather than as a direct prediction. The
remaining high upside direct test is a fold trained boundary conditioned local pair emission,
then a whole well run only if its fold zero truth versus alias margin clears its control.
`notes/literature_boundary_emission_2026_08_01.md`, `LEDGER.tsv`.

## The polarisation identity

Verified independently to 3.7e-14, `reports/verify_polarisation_2026_07_28.txt`. With
`r = pred - last_known`, `t = y - last_known`, `e = pred - y`, and per-well squared norms
`den = <r,r>`, `S = <t,t>`, `N = <e,e>`, the per-well scale minimising the deployed loss is

```
c_w = <r,t>/<r,r> = (den + S - N) / (2*den)
```

`den` is observed, so the whole per-well correction is governed by two unknown MAGNITUDES with
no sign anywhere. Only one binds: `S/den` has Spearman +0.692/+0.706 against `c_w`, `N/den` has
+0.023/+0.005. Every uncertainty feature this repo owns (`pfgap`, `div_spread`, `dense_std`,
`dtw_stoch_std`) predicts `N`.

`S` was then built and priced. Correlation on `log(S/den)` of 1.000 buys -1.345/-1.421 against
the global scalar; 0.714 buys -0.72; 0.478 buys -0.25. **A ledger KEEP needs about 0.45. We
reach 0.316**, and only from the stack's own residual shape. Alignment features reach -0.021,
geometry -0.063, neighbour +0.002. The axis is priced and shut; see `QUEUE.md`.

## The loop

```
edit QUEUE.md -> train -> make gate BASE=<name> -> LEDGER.tsv -> make ensemble
```

`make help` lists every maintained entry point. `scripts/` holds 320 files and most are one-off
probes kept for provenance, so prefer the targets below over writing a new script.

| what | how |
|---|---|
| verify the checkout | `make preflight` |
| gate a new base | `make gate BASE=<name>`, then never hand-edit `LEDGER.tsv` |
| what is gateable | `make gate-list` |
| honest best combination | `make ensemble` (SLOW, hours; it is a nested greedy) |
| sweep PF variants | `make pf-sweep` (~45 min, all 773 wells) |
| carry a PF variant to CV | `make pf-rebuild CFG=X` then `make pf-gate CFG=X` |

**Train locally:** `scripts/train_new_bases_2026_07_26.py`, the working GBDT recipe (median
objective, 5-fold GroupKFold by well, ~170 s per fold at row-step 4 / 1200 trees).

**Train on AWS:** the launchers in `aws/` work. `launch_cnn_sdf.py`, `launch_cnn_1d.py`,
`launch_lgb.py`, `launch_cat.py`, `launch_xgb.py`, `launch_realmlp.py`. Features are already in
S3; do not re-upload. **Verify every launcher flag against `--help` before pasting a queued
command.** On 2026-07-26 all three queued lines trained a different model from the one whose
number justified the queue entry, because launcher defaults are always transmitted and override
the entry point's own defaults.

**Collect a CNN job by hand;** `aws/pull_predictions.py` 404s on them. It reads
`output/output.tar.gz` but CNN entry points write to `/opt/ml/model`, so the artifact is
`output/model.tar.gz`. `scripts/build_cnn_5fold_base.py` then needs two DISJOINT CSVs and saves
to `<out-name>.npy` while `gate.py` looks for `oof_<base>.npy`, so pass the `oof_` prefix.

## Campaign continuity, required while the deadline is open

**The objective is a GOLD medal, public LB 5.899.** Not an improvement, not a respectable
finish. `harness/forward_target.json` carries it, with CV 6.0 as the deliberately stricter
working proxy. Do not convert CV to LB to argue success: that mapping rests on five points whose
correlation is negative.

**Stopping and deferring are the same failure, and both are enforced by script.** These are NOT
admissible reasons to stop work, to defer an experiment, or to spend the remaining time writing
documentation instead of running something:

> not enough time remaining; too close to the deadline; this is post-deadline work; this is a
> multi-day build; the remaining ideas are too speculative; we have plateaued; every axis is
> closed; the expected gain is too small; further work is unlikely to succeed; the picks are
> already locked so the score is fixed.

Only four things end the work: **the objective is reached, the deadline passes, the owner says
stop, or a named external blocker prevents the next bounded experiment** and you can state the
attempt that hit it.

**A large idea is never deferred, it is decomposed** until its first bounded experiment fits the
time actually available, and that experiment is run. `harness/breadth_gate.py` rejects the axis
registry outright if an open axis's `next_action` contains deferral language such as
"post-deadline" or "multi-day", because an open axis with a deferred action is early stopping
wearing a different hat: the registry looks healthy while nothing is scheduled.

**When every axis closes, open a new one.** A closed axis is a result, not an ending. The correct
response to "the search is exhausted" is a new axis with a first experiment, not a report.

A clean negative retires its measured mechanism. It does not end the campaign or license a
plateau claim.

1. Keep one bounded high upside forward model experiment and one inexpensive verification or
   deployment task in `QUEUE.md`. Finish the next eligible item after recording a wash.
2. A failed test must state its exact scope, its control, and the next distinct mechanism. Do
   not revive the same family by changing constants.
3. After two material failures, or after two hours of experimentation, mine fresh Kaggle forum
   and public notebook evidence. Then seek an independent scientific or engineering source.
   Record only mechanisms that change an observable, a target, or an inference family.
4. Use an independent deputy or Claude Code for a red team read of each new forward model idea.
   The main loop verifies the deputy's claimed measurement before it is banked.
5. Continue until the target is reached, the deadline closes, the user stops the work, or an
   actual external blocker prevents the next bounded experiment. Difficulty and a run of washes
   are not blockers.
6. Update `QUEUE.md`, `LEDGER.tsv`, and `SESSION_SUMMARY.md` after each material result. Give
   the user an honest interval update and name the next test.

`harness/forward_axes.json` is the persistent record of distinct forward model and observation
routes. Unlike a generic tabular breadth quota, it does not reward irrelevant model variants.
Run `make breadth-status` at each session start. `make lock-check BEST_CV=<value>` rejects a
campaign completion statement while CV is above 6 and the deadline remains open. An override
requires a written reason and appends it to `harness/campaign_override_log.tsv`.

## Traps, each of which has cost at least half a session

**PIN BLAS THREADS IN EVERY WORKER POOL.** A 13-worker `ProcessPoolExecutor` doing SVDs drove
load average to **137** on a 16-core machine, roughly nine times oversubscribed, and cost about
40 minutes before it was noticed. Every parallel script must be launched with
`OMP_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 MKL_NUM_THREADS=1`, and 12 workers is the practical
maximum here.

**AND THE `0.0 ft` ANCHOR BARS ONLY HOLD AT ONE THREAD.** Measured 2026-08-04. The GRU-free
six-base path reproduces its banked array at exactly `0.0` under `OMP_NUM_THREADS=1` and at
`4.366e-11` under 2 or 4 threads, because the thread count changes the reduction order inside the
positive Ridge solve. Every `!= 0.0` gate in `scripts/emit_*_constants_*.py` and the port
verifiers therefore ABORTS on a correct checkout that was merely run unpinned, and the failure
message says the base set is wrong, which it is not. Pin to one thread before reading any anchor,
and do not widen a tolerance to make one pass. `4e-11` is twelve orders of magnitude below the
0.002 ft micro-comparison noise floor, so it is a reproducibility statement and never a modelling
one.

**A BANKED `pred_*.npy` IS AN ANCHOR, NOT A SCRATCH FILE.** A builder that recomputes an existing
path under a new name must not also rewrite the old one. On 2026-08-04 a nine-seed ladder script
re-emitted `pred_psr4avg3.npy` from its own `n=3` point; the values were the same base average but
reached by a different float32 accumulation order, which moved the Ridge solution by `4e-11` ft and
broke the shipped kernel's constant derivation until the canonical file was regenerated.

**AWS LAUNCHER DEFAULTS ARE TRANSMITTED AND OVERRIDE THE ENTRY POINT.** Every launcher builds its
hyperparameter dict unconditionally, so a default you never typed still wins. Two are armed right
now. `launch_cnn_1d.py` defaults to `--drift-mode direct --w-local 0.0 --w-smooth 2.0 --epochs 60`,
which is the `cnn_1d_direct` WASH arm at blend-add -0.0044, NOT the `resid / 5.0 / 0.0 / 80` recipe
behind every banked KEEP at -0.0314; any queued line must carry those four flags explicitly.
`launch_cnn_sdf.py` defaults `--pf-prior auto` and `pf_candidates_n773_s64_full64.npz` sits inside
the mounted training channel, so a default SDF launch silently trains WITH the PF prior.

**A REAL DEFECT, unfixed:** `--h-future 768` at `h_step 48 / nn_scale 4` gives 12 ft bins covering
9,216 ft, but 9 of 773 wells have longer eval regions and the builder pads and truncates rather
than failing. That is the likely source of the recorded "partial well coverage despite full
length", and it means every banked `cnn_1d` base is silently blind on those wells' tails.

**SUBSET SCREENS OVER-PROMISE, second documented instance.** A 40-well smoke of the curvature-free
shape search read real-minus-rolled **-0.1238**; at 773 wells it read **-0.0234**, five times
smaller, and a 60-well rerun already showed the decay at -0.0529. Quote this beside
`cnn_1d_direct`, which screened at -0.1394 and gated -0.0010.

**THE ERROR IS LONG-SCALE, and where it lives is now measured.** Over all 3,783,989 rows the
deployed 8.2251 decomposes as a per-well DATUM of 6.2943 at 58.6 percent of the variance, a LINEAR
shape half of 3.7795, and a NON-LINEAR half of 3.7081. Content below 400 ft of MD is 0.6189 ft,
just **0.57 percent** of the total. Any proposal aimed at short-scale structure is aimed at almost
nothing; the datum and the long-wavelength drift are where the error is.

**The OOF grid is EVAL ROWS ONLY.** `meta_train.parquet` has 3,783,989 rows, not one per CSV
row. Well `000d7d20` has 5278 CSV rows, 1442 known and 3836 eval, and its grid rows carry
`row_index` starting at 1442. Any per-well array must be gathered at `row_index`. Truncating a
full-length array onto the grid writes known-region values into eval slots and still returns a
plausible float; it scored 152 against a banked 10.36 before the known-winner arm caught it.
`make preflight` checks this invariant.

**Well-subset screens over-promise, not just fold 0.** `cnn_1d_direct` screened at -0.1394 and
gated WASH. The PF's ROBUST200 variant read -1.13 on 30 wells and -0.398 on 773. Same code,
same variant, a third of the effect. Use a subset only to decide where to spend GPU.

**Width is poison at 773 wells, even with an in-fold screen.** The 1446-column per-well matrix
reaches R2 -0.073 and -0.233 where 16 hand-named columns reach +0.078; adding 378 alignment
columns to those 16 cuts correlation from 0.316 to 0.112. Start narrow and add deliberately.

**Seeds are a distribution, not a number.** A hardcoded `random_state=42` in a LightGBM arm was
the minimum of 12 draws, worth 0.011 ft of pure optimism. Bag over seeds before quoting.

**Weighted least squares takes `sqrt(w)` on the rows.** Scaling by `w` minimises `sum w^2 (.)^2`
and silently solves a different problem from the control sitting next to it.

**Verify every kernel port against the banked OOF before shipping.** Both CNN codebases build a
model INPUT from the label and degrade silently when it is absent. The CNN-1D one emitted pure
carry-forward and is fixed; the SDF one is why the SDF cannot ship.
`scripts/verify_cnn1d_inference_2026_07_28.py` is the pattern to copy.

**The released `test/` directory is 3 TRAIN wells, and their labels are one file lookup away.**
`000d7d20`, `00bbac68`, `00e12e8b` are MD-identical to train wells of the same id, and the train
copies carry `TVT` on 100% of the 14,151 hidden rows. Contact reconstruction off those files scores
**0.0053 ft**. Two consequences. First, the public leaderboard is NOT these wells, or the whole 8.0
family of public notebooks would score 0.005 instead of 8.0, so `test/` is a development sample and
the scored set is different wells. Second, the formation contact columns
(`ANCC ASTNU ASTNL EGFDU EGFDL BUDA`) exist only in train horizontal files and each one is the
target reparameterised: an oracle on CV, absent at test, inadmissible at inference. The exact
identity, measured over all 773 wells in
`reports/probe_formation_observable_oracle_2026_08_01.json`, is

```
TVT + Z - surface_f = C_w        constant per well, for ALL SIX surfaces
```

to a median spread of **0.05 ft** and a max of 0.08 ft. So `U = TVT + Z = surface_f + C_w`: the
projection variable this repo already uses IS the structural surface plus a per-well datum. That is
why the U-projection works and why the `plane_fit` and `dense_ancc` groups, which interpolate those
surfaces from offset wells, are the deployed route. It also means the whole task decomposes as
"estimate the surface shape along the well, and take the datum from the known prefix", and our
residual is exactly the part of the surface that offset wells cannot predict. `scripts/public_lb_oracle.py`, `scripts/probe_test_train_overlap_2026_08_01.py`,
`notes/public_6213_teardown_2026_08_01.md`. The public 6.213 notebook's headline mechanism is this
lookup and nothing more.

**A leak filter must check how a base was TRAINED, never its name.** `cnn_surface_twin` has no
"geom" in its name and consumes the geom prior as an input channel (`aws/src/train_cnn_sdf.py`,
`"geom-candidate": "geom_k16"`). Admitting it to a no-geom pool bought a fake -0.58. Ten of the
banked bases are inadmissible and three more have partial well coverage despite full length;
`harness/gate.py` holds the list with a reason each.

## Vocabulary that will mislead you if you skip it

- **"Field-safe" means "no geom prior". It does NOT mean neighbour-free.** Every base, including
  Pick-2's own three, trains on the `plane_fit` and `dense_ancc` groups, which fit spatial
  planes and dense correlations across *other* wells: 30 such columns per divergence base. A
  genuinely typewell-blocked retrain costs **+0.31 ft**. Read every "field-safe" number as
  "no-geom, with ~0.31 ft of residual field dependence still in it".
- **The geom prior's CV edge is within field leakage.** Field blocked CV retains 2% of it; a
  random-block control of the same held-out size retains 117%. It also *reverses* by 0.233 ft
  on public LB. That is why Pick-2, not Pick-1, is the honest primary.
- **`pf_selector` is not a base.** It enters the blend at the fixed 0.24 weight downstream of
  the Ridge, and is on `gate.py`'s INADMISSIBLE list for that reason.

## Two-pick endgame

Both selected picks are submitted and scored, so both already survived a hidden rerun. No
deadline failure mode touches them unless a new version is saved.

- **SETTLED 2026-08-01. The two selected picks are `55157342`, titled
  `rogii ratecoupled gbdt v1, Version 1`, and `54791000`, titled
  `rogii pf multidiv proj geom, Version 2`.** `55132115` was deselected. Both selected
  submissions are scored and have survived a rerun, so no deadline failure mode touches them
  unless a new version is saved.
- That pair was measured, not assumed, in `reports/probe_pick_pair_decorrelation_2026_08_01.json`.
  All three candidate CVs reproduce their records: 8.2251, 8.2826, 8.3811. `55157342` and
  `55132115` are near duplicates at error correlation **0.9866**, differing by one base, and their
  average is worse than the better one alone, so selecting both would spend two picks on one bet.
  `55157342` with `54791000` is the most decorrelated pair at **0.9305** and the only pair whose
  half average beats both members, by 0.1159. The two are also different kinds of bet: `55157342`
  is the best honest CV with no geom prior, `54791000` is the geom arm whose CV edge is largely
  field leakage and is therefore a hedge rather than a primary.
- New candidates must be **submitted by 2026-08-03** so they return a score with room for one
  retry. Never select a submission that has not returned a public number.
- A new submission displaces a selected pick only if it beats cross fit CV **8.2251** and has
  returned a normal score. That is now the bar, since `55157342` is selected.
- **Never select `54853374` or `54851870`**, the 2026-07-20 pair at 26.939.
- A forked public notebook is not selectable without a 773-well CV of our own (owner decision).
- **The one CV/LB pair that is a controlled experiment went the wrong way.**
  `rogii pf multidiv proj realmlp v1` is Pick-2's pipeline plus exactly one base. CV 8.9490 ->
  8.8144 (-0.135), LB **7.666 -> 7.862 (+0.196)**. Quote it whenever someone expects a CV gain
  to appear on the board.
- On **2026-08-05** run `notes/POST_DEADLINE_HARVEST_RUNBOOK.md`. Every sub-6 team declines to
  disclose while the competition is live; that expires at the deadline.

## Provenance

Until 2026-07-26 the read path was nine documents totalling ~7,000 lines. All of it is in
`archive/`, unchanged and in git history, as a grep target rather than a read path. The rewrite
happened because the project spent twenty-plus sessions optimising inside a family whose ceiling
was one `lstsq` call away and never measured, while the documentation about why progress was
slow kept growing.

Train models, control them properly, ensemble what decorrelates, and write down what failed.
