# SESSION 2026-08-04 — LIVE HANDOFF. Read this, then AGENTS.md, then QUEUE.md.

**Deadline 2026-08-05 23:59 UTC. Today's quota is SPENT (`55244616`, `55244617` at 16:00 UTC).
Five submissions arrive at 00:00 UTC on 2026-08-05.**

---

# 2026-08-05 22:25 UTC — CLOSING BLOCK. One retraction matters more than everything else here.

**Selection is FINAL and unchanged: `55244616` + `55244617`.** Both scored, both survived a rerun.
Three submissions remain unspent and expire at 23:59 with nothing worth spending them on.

## THE HEADLINE: the submission-latency constant was wrong by ~35x

`QUEUE.md` said "THE SUBMISSION CUTOFF IS ~15:30 UTC, NOT 23:59", on an observed 5h20m-7h30m
latency. **Measured tonight end to end: submission `55281768` entered 22:08 UTC on the final
evening, the busiest hour of the competition, and returned a public score at 22:18. TEN MINUTES.**

Latency is dominated by **the notebook's own runtime**, not a shared queue. Ours run 255-405 s.
Topic 733099 carried both readings side by side, msg 3509012 "full 200 wells take only 4-5 mins, and
full submission kaggle too" against msg 3509011's 8.5 hours, and I recorded the slow one as
confirmation and dismissed the fast one as a competitor quirk. That was backwards, and it was
written into the harvest note at 21:30 as "retroactively confirms that the ~15:30 cutoff was right".

**What it cost.** `bag4_add` (CV 7.135123, E[min] -0.0279 against the selected pair) was ruled out at
11:05 because its ~5 hour retrain "costs the whole remaining window". Under the true latency it
fitted with roughly seven hours to spare. It was still under the 0.05 ft floor and might have been
declined on merit; the point is the decision was never made. **Measure submission latency on day
one of the next competition.** Retraction is in `QUEUE.md` where it gets read, not only in notes.

Deotte's rule in 732947 is untouched and was always the correct half: only a submission that
completes and has a public score before the deadline is selectable.

## The selection was re-challenged and survived

`reports/price_final_selection_emin_2026_08_05.json` ranks the selected pair **fourth** among
selectable pairs; `55244616 + 55234752` is nominally 0.0080 ft better on E[min]. Tested properly in
`scripts/probe_pair_swap_stability_2026_08_05.py`:

* Test A, 200 RNG seeds: not Monte Carlo noise, the challenger wins 200/200 at sd 0.00049.
* Test B, 400 outer resamples of the 773 wells: **-0.0104 ft, CI95 [-0.0417, +0.0093], P(helps)
  0.807.** Re-run at `k_wells=148` (the private set is ~148 of ~200, not 200): -0.011, P 0.792.
* Test C: the board prefers the incumbent by 0.273 ft, z +1.69, and the multipath lever has now
  moved the board the wrong way twice, +0.160 and +0.200, on CV gains of -0.069 and -0.052.
* Test D: the swap takes multipath exposure from **1 of 2 picks to 2 of 2**, discarding the only
  single-path hedge, which is the stated reason `55244617` exists.

**Verdict: do not swap.** A hundredth of a foot, a CI through zero, one chance in five of active
harm, for the loss of the only uncorrelated-code pick.

## The well-as-image question, asked and answered by submission

Tucker Arrants (LB 5.078) disclosed in topic 733099 that his inference is "~200 forward passes per
model", one per well, each well as an image. We did **not** miss it: `src/cnn_sdf` is a faithful
hengck23 port, a 2D U-Net over a (typewell TVT x horizontal column) image emitting a dense SDF.

So we submitted it standalone to find out. **`55281768` scored 16.204.**

| arm | 3 released wells | public LB |
|---|---:|---:|
| deployed stack | 3.59 | 6.618 |
| CNN-SDF surface, standalone, 4 folds | 11.29 | **16.204** |
| absolute gap | 7.70 ft | **9.59 ft** |

The gap widens. **The family does not transfer better than our stack**, and the "item one for next
campaign" call I made at 21:30 is downgraded on this evidence. Its value was always what we already
had it as: a decorrelated stack column at -0.0174, err-corr 0.460. Caveats: 4 folds not 5 (no fold-0
checkpoint for the geom-free run), standalone rather than stacked, our port not his. The
`correspondence_loss` re-check remains formally UNVERIFIED (killed on drift-corr with "ensemble win"
in its own kill text) and is still worth one GPU run, but it is no longer the headline.

Reproducible without Kaggle: `scripts/build_sdf_submission_2026_08_05.py`.

## The 6.546 public notebook, checked and dismissed

`farhanabidtech786/rogii-best-score-wellbore-geology-prediction`. Beats our 6.618 by 0.072 ft, which
is z 0.45 against the board's own 0.161 ft difference noise, p 0.66. It is an aggregator mounting
seven external artifact datasets and **every one of its ingredients scores better than the blend**
(pilkwang 5.952, needless090 6.426, fleongg 6.429, thbdh5765 6.486, ravaghi 6.505). Zero executed
outputs, so no CV of any kind. All nine profiles ship `run_guarded_overlap_override=True`, the
train/test label lookup we tore down on 2026-08-01 and proved inert on the scored set.

## Loose thread for next time

The competition's train wells ship **per-well `.png` files** we have never touched. Noticed in a
kernel log tonight. Given the disclosure being chased is literally "treat each well as an image",
that is worth pulling first.

---

# 2026-08-05 17:25 UTC — THE FIRST SUBSTANTIVE DISCLOSURE, and it corroborates us rather than beating us

Topic **732999**, msg 3508839, 2026-08-05 14:11 UTC, 3 votes. A competitor at **CV approximately
equal to LB at 8.0 ft** posted a 4,272-character account of their method and asked for advice. They
are BELOW us, we hold CV 7.1593 and public 6.618, so nothing here is a lever. Its value is that an
independent party working a completely different way reached four of our conclusions.

**What it corroborates, each against our own measurement.**

* *"the median deviation is very low at 3-4 ft, yet it rises due to a few catastrophic estimation
  failures"*. Our own: the worst 1 percent of wells carry 18.7 percent of squared error and the worst
  5 percent carry 43.5.
* *"TVT = g - Z + datum. Therefore, T' = g' - Z' = constant - Z'"*. That is our identity
  `TVT = surface + C_w - Z` from the `structural_surface_transfer` axis, derived independently.
* *"Perfectly matching the typewell GR (GV) and horizontal GR (GH) is meaningless. Indeed, the TVT is
  not matched in this way either. However ... discrepancies exist in the form of a thick-tailed
  t-distribution."* Our emission certificate: the truth loses 153 of 153 against the DP optimum, and
  the rolled typewell arm is indistinguishable. They then build the t-distribution emission we
  already closed, since per-row heavy tails are blind to `corr(d_i, d_j)` by construction.
* **The failure mode they cannot fix is the one we registered as the real open problem.** Verbatim:
  *"when the drill penetrates deeper into the strata ... the prediction for the TVT gradient deviates
  significantly, and unless there is a strong reduction in GR emission sufficient to offset that
  deviation, there is no reason to be drawn toward the true path."* That is "the objective prefers a
  wrong track", arrived at from the other side.

**Their conclusion is rank 4's conclusion, from a different direction and a different score.**
Verbatim: *"there is no longer any benefit from a single method; benefits are found only through
ensemble work. This may imply that there is no definitive lever to infer the true TVT of the hard
well, and that only weak determinants causing trade-offs remain."* **Two independent sources now say
breadth rather than mechanism.** Rank 4 said it from 5.0 CV on 135 models; this competitor says it
from 8.0.

**One NEW number and one DISCREPANCY worth checking after the close.**

* *"The length over which the bedding plane slope g' remains constant appears to be approximately
  300 ft."* Ours are 400 to 800 ft for honest local linearity and a ~170 ft rate correlation length.
  Neither contradicts it; nobody has measured this exact quantity here.
* *"Neighboring bedding planes can serve as good bedding plane priors and are good estimators of the
  g' described above."* **This is neighbour pooling, and they report CV approximately equal to LB at
  8.0 while our `geom_k16` neighbour edge is 98 percent within-field leakage and REVERSES by 0.233 ft
  on the board.** Either their prior is built field-safely in a way ours is not, or their CV/LB
  agreement is coincidental at that error level. It is the one claim here that touches a closure of
  ours, so check it against whatever they publish.

Topic **733015**, "This competition broke me", is banter with one open question nobody answered:
the theoretical ceiling of DP or particle filtering without GPU-trained models.

---

# 2026-08-05 11:05 UTC — SHIP NOTHING. The answer is measured, and the ceiling is the reason.

Preflight green, 8.2826 / 8.7300 / 8.9641. Live state re-verified against the Kaggle API rather than
against this file: `55244616` public 6.818 and `55244617` public 6.618 are both `COMPLETE` with
returned scores, **zero of today's five submissions are used**, and `rogii-seqalt-gru-psr4avg9` shows
a single upload at 2026-08-04 14:49. Nothing was pushed, re-versioned or submitted.

**The decision is NO, and it rests on a ceiling rather than on a survey.** The adversarial verifier's
selection-free arm `adv_bag4_and_gbdtball` takes ALL 25 `gbdtdiv` arms, so there is no family left to
choose, and on the deployed multipath path it reads **-0.0375 ramped, CI95 [-0.0909, +0.0144],
p_worse 0.081, 3 of 5 folds**. The -0.0536 headline for `bag4_and_gbdtbag` was the `lgb`-family pick
made on full data; **choosing the family is worth 0.016 of it.** So the most the entire GBDT-bag
family can be worth, with nothing withheld for portability, is -0.0375, and it misses the 0.05 ft
floor on magnitude, on the CI and on fold count at once. **No portable subset of it can exceed the
whole.** The shippable half stays at -0.0116.

**A whole-repo sweep confirms nothing was missed.** Every report JSON from 08-04 and 08-05 was scanned
for any delta past -0.05: 166 hits, all resolved. Two were absent from this file and both looked live.

* `typewell_blocked_selection` at pooled -0.0520 to -0.0577, 4-5 of 5 folds, is **not a candidate**.
  It is the multipath-minus-single delta between the two ALREADY-LOCKED picks, re-measured under
  typewell-blocked cross-fitting. It VALIDATES the selection: the ordering survives the 98.1 percent
  sibling leak at -0.0519 by-well and -0.0519 by-typewell, p_worse 0.017 and 0.015. **It also prices
  the leak: the deployed path scores 7.18744 under typewell-blocked downstream cross-fitting against
  7.15932 by-well, so 0.0281 ft of our CV is stacking-level typewell leak.**
* `gbdtbag_lgb_swap` at -0.0698 ramped against the single-path comparator and -0.0465 against the
  deployed path is real and is the same unshippable family.

**The eighth-column screen is exhausted and its top arm is a control failure.** 234 candidates; best
uncorrected -0.0392 and best corrected-and-ramped -0.0451 at 5 of 5 folds, both
`gbdtdiv_lgb_grraw_2026_08_03`. It is the max of 234 chosen on full data, so rule 15 applies; it
already failed its rolled null on 2026-08-03 at real-minus-null **-0.0238 against a seed-noise sd of
0.0264**, recorded in `notes/gbdt_diversity_2026_08_03.md` as failing rule 3; and it has no saved fold
models. Arms ranked 21 to 234 cannot rescue it either, since they sit at `delta_uncorrected` >= -0.0002
and lifting that to -0.05 corrected needs a correction differential of 0.05 ft where the largest ever
measured here is 0.029.

**Portability was checked rather than assumed.** The `gbdtdiv` family has `oof_*.npy` and
`summary_*.json` in `predictions/` and nothing else: no checkpoints, no Kaggle dataset,
`aws/src/train_gbdt_div.py` ran on AWS. Shipping it means retraining 13 to 25 LightGBM arms AND
building each one's features test-side. The cheap escape was already priced at +0.0098 uncorrected.

No insurance case either: QUEUE section 5 holds a slot against a port bug, and both picks returned
normally, so there is no bug to fix.

## External literature read the same session. One source CLOSED, one attribution CORRECTED.

`notes/external_literature_2026_08_05_read.md` has the full record with provenance.

* **The Deep Hierarchical Graph Correlator is closed by its own numbers.** It was the runbook's best
  external hope, the one mechanism attacking the rho 0.28 cap with a learned kernel rather than
  another statistic. On 89 pairs it reads Pearson 0.91 against **cross-correlation's 0.90**, and the
  paper's own Wilcoxon test puts that difference at **p = 0.15**, while DTW beats it at p = 1.6e-12.
  It buys an 8.2x speedup over DTW and nothing else. **A learned similarity is not a better
  similarity.**
* **TST3D publishes no accuracy figure at all**, only runtime. It is not a benchmark we failed. Its
  algorithm is free in patent `WO2015070022A1`: relative-L1 panel error rather than NCC, one degree
  of freedom per hinge with azimuth frozen, heel-to-toe commitment, an operation tree with rollback.
* **`negatives.md` N3 is misattributed and half of it is dissolved.** Crossref confirms
  **SPE-202046-MS is the HOST COMPANY's paper** — Denisenko, Kuvaev, Uvarov, Kushmantzev, Toporov,
  all OOO ROGII EUROPE — not a Mitkus paper. The same four inventors hold **`US11480045B2`**, which
  discloses a composite objective `K = T^pow1 × SC^pow5 / (SQ^pow2 × (∑Δa)^pow3 ...)`: Pearson
  correlation times a self-correlation term, divided by an RMS deviation term and by dip-roughness
  penalties, searched over overlapping segments restarting at 10-60 percent of the previous one.
* **The convergence across TST3D, Rogii and Chevron is a design observation, NOT a lever.** All three
  restrict freedom to about one degree of freedom per segment, penalise adjacent-segment change,
  anchor to a pre-drill prior and commit sequentially. Our own certificate already ran a decoder with
  a quadratic transition penalty and the truth lost **153 of 153**, with the rolled arm
  indistinguishable. The diagnosis there is dimensional, not about the penalty: 1 parameter carries
  signal, 2 already carries none. A composite objective reweights an objective whose information
  content we measured; it does not add identifiability. **Do not write this up as a refutation.**
* **One scope note for `kill_audit.md`, not an overturn.** `within_well_gr_self_consistency` was
  killed as a **standalone path selector**. The host uses self-correlation as a **multiplicative
  factor inside a composite**, which our measurement does not bind.
* **Durable method lesson: for any paywalled petroleum-engineering paper, look for the patent.** Three
  of four algorithms were recovered in full from free patent text. `api.crossref.org` is unblocked and
  is the right tool for attribution. OnePetro is 403 plus CAPTCHA on every route including
  `r.jina.ai`; MDPI needs `r.jina.ai`; Google Patents full-text pages render directly while its search
  pages do not.

---

# 2026-08-04 EVENING. `permutation_aligned_weight_fusion` is CLOSED. Nothing shipped, nothing changed.

Git Re-Basin activation matching was run against its pre-registered 11.1 ft bar. **The bar is missed
by 6.9 ft and the axis closes.** No model, dataset, kernel or submission was touched.

**The mandatory control passed first, and it is the part worth keeping.** Random permutations of all
19 permutable unit sets of the bidirectional GRU — three gate blocks moved as one unit, each
direction separate, the recurrent matrix permuted on both sides, the GroupNorm stem restricted to
its legal `S_12 wr S_8` wreath product — leave the output unchanged to **1.819e-12 ft** over 12
draws that move at least 97.6 percent of units, against a 1e-6 ft bar. Measured in float64 on
purpose: the float32 deviation is 9.766e-04 ft, which is exactly one ulp at the 12,193 ft output
scale. Three deliberate mis-applications of the same machinery move the output 12.97, 181.5 and
34.24 ft, so the control has teeth. The algebra is banked in
`scripts/perm_align_gru_2026_08_04.py` and is reusable.

**The measurement, on the identical 24 fold-0 wells and 115,921 rows.** Alignment takes the
interpolation midpoint **86.3525 -> 17.9837 ft [13.5494, 22.1211]**, removing 90.2 percent of the
barrier and missing the 11.1 ft bar with the whole interval above it. Solving the permutations on
24 DISJOINT fold-0 wells, the honest version, gives 21.4227 [16.6813, 26.6529]. Endpoints reproduce
exactly at 10.5622 and 10.8508 and every aligned single reproduces its unaligned RMSE. Priced beyond
the bar for the record: the nine-way ALIGNED weight average is **25.1135 ft [17.6828, 32.9102]**
against the nine-way PREDICTION average's 10.3656 and a naive weight average's 102.5044.

**Two controls carry the interpretation, and both refute what the headline alone suggests.** The
mean matched activation correlation of 0.5374 looks like discovered structure; the fresh-init null
says it is not. Two UNTRAINED networks matched by the identical machinery reach **0.7750**, HIGHER
than the trained pair, and their aligned weight cosine is 0.1105 against the trained pair's 0.1152.
A linear assignment over 96 units manufactures that much correlation from nothing, and the trained
seeds have LESS unit-level correspondence than random inits. Permutation is not the obstruction:
these seeds learn genuinely different feature bases. Separately, the merged midpoint keeps 0.909 of
the endpoints' mean per-unit activation scale, so REPAIR-style variance rescaling has at most 9
percent of scale to recover and cannot close a 7.42 ft barrier. **Do not re-propose weight
averaging, permutation or activation matching, or REPAIR for these checkpoints.**

Reports: `reports/axis_permutation_aligned_weight_fusion_2026_08_04.json` plus the four probes it
assembles. Successor axis opened: **`inference_time_prefix_resampling`**, the one legal source of
new ensemble members that needs no retraining, with its two bars in `harness/forward_axes.json`.
`make breadth-status BEST_CV=7.159` is green at one open axis.

---

# CLOSING SNAPSHOT, 2026-08-05 09:54 UTC — final sweeps done, nothing actionable

**Board: 6,152 teams. Gold 5.518, silver 6.350, bronze 6.400. Us 6.618 at rank 1,824**, with 1,175
teams in [6.405, 6.618). Selection is locked on the 6.618 and 6.818 kernels.

## THE PUBLIC AND PRIVATE WELLS ARE DISJOINT — settled by the official banner

The leaderboard banner reads: *"This leaderboard is calculated with approximately 26 percent of the
test data. The final results will be based on the OTHER 74 percent."* Quoted by Pavel in 731550 on
2026-08-05 00:32, correcting Tucker Arrants, who did not contest it.

**Our 6.618 contributes NOTHING to the private score. There is no partial credit.** Everything the
public board told us about the multipath block — the +0.160 and +0.200 — was measured on 52 wells
that will not be scored. That does not make the signal meaningless, but it does mean the private
result is an independent draw and the CV-versus-board disagreement was never resolvable by
submitting more.

## AN INDEPENDENT MEASUREMENT OF PRIVATE-DRAW NOISE, and it validates the pair

Tucker Arrants ran a well-subset bootstrap on his own model (731550, 2026-08-05 01:05):

```
 50 wells   68% [4.248, 5.861]   90% [3.843, 6.762]    population 5.1362
200 wells   68% [4.711, 5.542]   90% [4.490, 5.818]    population 5.1362
```

At the host's stated 200-well private size that is **+/-0.42 ft at 68 percent and +/-0.66 ft at 90
percent**. Our two locked picks differ by 0.20 ft on public, which sits deep inside even the 68
percent band. An independent party has now corroborated this repo's own finding that the second
pick is worth at most 0.008 ft on modelling grounds. **The pair is fine and the split is the right
use of it.**

## RANK 4 DISCLOSED HIS SETUP, and it is breadth rather than a mechanism

Tucker Arrants, 731550, 2026-08-05 00:12: *"I am picking my best CV, blended across around 135
different models (fold, seed, architecture) and then the same one with a type of TTA that is a wash
on CV but has consistently helped all my models on the LB."* And at 2026-08-04 23:18: *"most of the
move from 8 down to 5 CV was met with noisy LB results and the gap has now settled at 5.0 CV and
5.5 LB."*

**He started where we started, at CV 8, and got to 5 on blending breadth across fold, seed AND
ARCHITECTURE — not on one mechanism.** That is the single most useful strategic fact of the
campaign and it arrived too late to use. It also reframes our own encoder-diversity result: we
measured that a different architecture is the only thing that breaks the 0.78 correlation wall
(cross-family 0.6066) and closed it because ONE transformer at the GRU's hyperparameters was 16
percent worse standalone. At 135 members the arithmetic is different. **This is the first thing to
build for next time.**

His TTA claim is worth nothing to us twice over: it is an explicitly board-only effect on 52 wells
his own bootstrap says cannot resolve it, and our TTA axis is closed on all five arms.

## Two more external CV/LB pairs, both saying the board cannot rank

Pavel, 2026-08-04 22:51: *"I first achieved ~6.5 LB with ~8.0 CV. Then as I was gradually moving
toward ~5.0 CV I have now, the LB score kept randomly fluctuating between 6.2 and 6.8."* He sits at
public 6.220, rank 144. **A team roughly two feet better on CV is 0.4 ft better on the board.** More
reason to treat the refitted 1.1711 slope as measuring board noise rather than transfer.

## Process, confirmed by Chris Deotte

732947, 2026-08-05 09:22: *"Only a submission that completes and has a public LB score (before
competition deadline) is eligible (as one of your final subs) and possible to select."* This
independently confirms both the ~15:30 UTC practical cutoff and this repo's standing rule that a
pick must have returned a number.

## The 14,151-row hypothesis: source-level CONFIRMED, runtime still open, one inference WITHDRAWN

The guards are real and live in the pulled version. Cell 85 raises `RuntimeError` unless the pre-Q
submission's SHA matches a hardcoded literal or its stats match
`{rows 14151, tvt_min 11589.0386, tvt_max 12240.0161, tvt_mean 11904.2379, tvt_std 277.3705}` to
1e-8; then unless exactly one branch row is "applied"; then unless that row's well is `00e12e8b`
with shift exactly 2.0 and `moved_rows` exactly 4301. Cell 93 raises unless all three layers hold
exactly 14,151 rows. None is inside a `try`.

Supporting: in the sibling `yaroslavkholmirzayev/rogii-contact-and-u-restore`, the identical
cell-85 raise is **commented out and replaced with `print("Wertiba")`** — an author in this lineage
deliberately defanged that exact guard, which is evidence it is known inside the lineage to be a
rerun hazard.

**WITHDRAWN:** the argument that the guards must pass at rerun because evgendvorkin's team shows
6.405 with a last-submission timestamp matching the guarded version. Leaderboard `Score` is a team's
BEST while `LastSubmissionDate` is the most recent SCORED submission, and they need not be the same
one — our own row at 08-04 07:14 read 6.888 against a timestamp belonging to `55223770`, which
scored 6.992. The timestamp certifies that some submission completed, not that the guarded one did.

Unresolvable before the close and it changes nothing we can act on.

---

# 2026-08-05 — SELECTION MADE, and the last leads measured

**SELECTED by the owner: the submissions scoring public 6.618 and 6.818.** These are the nine-seed
single-path and nine-seed multipath kernels, submitted within three seconds of each other at
16:00 UTC on 2026-08-04. They are the same nine seeds WITH and WITHOUT estimate-space datum
averaging, which is the one component whose sign is genuinely disputed: CV says it gains 0.05-0.07
at p_worse 0.0053 over 773 cross-fitted wells, the board says it costs about 0.18 ft at two
separate seed counts. Splitting the pair covers both readings for 0.008 ft of expected score.

**THE REAL SUBMISSION CUTOFF IS ABOUT 15:30 UTC, NOT 23:59.** Observed queue latency on 2026-08-04
ran 5h20m to 7h30m over three submissions, and a submission must RETURN a public score to be
selectable at all. Anything sent after roughly 16:30 UTC will not report in time.

## The last real lead, and why it does not ship

The final sweep found `bag4_and_gbdtbag` at ramped **7.10575** against the deployed 7.15932 —
**-0.05357**, CI95 [-0.1070, **+0.0004**], p_worse **0.026**, 4 of 5 folds, control -0.3616. The
first arm all campaign to clear the 0.05 ft floor with a CI essentially clear of zero. It has two
halves and each was measured separately rather than assumed:

| half | what it is | corrected + ramped | shippable |
|---|---|---:|---|
| `zz_ratecoupled_bag4` | 4 seeds of the ratecoupled GBDT, same recipe and features | **-0.0116** CI [-0.0454, +0.0223] | YES, cheaply |
| `zz_gbdtbag_lgb` | 28 GBDT families, each with its own feature construction | carries the rest | NO |

**The gain lives entirely in the half that cannot ship.** Porting 28 families, several reading
artefacts with no test-side counterpart, is not a six-hour job.

**The obvious escape was tested and fails.** The kernel already computes `lgbdivmed` and
`xgbdivmed`, so a bag restricted to the twelve members sharing that divergence frame would have
been nearly free. It is WORSE uncorrected, 7.38236 against 7.37225. No bank was rebuilt for it:
turning a +0.010 raw loss into a -0.05 corrected win needs the correction to be worth 0.06 ft more
on the candidate path, and the largest such differential ever measured here is 0.029.

## Encoder diversity: the one thing that broke the correlation wall, and it still lost

Eight transformer arms at the compound recipe. **Cross-family pairwise error correlation 0.6066
against 0.7825 within the nine GRU seeds** — the only break in the 0.78 wall this campaign
produced, and no schedule change ever moved it at all. It still fails on quality: the transformer
is 16 to 19 percent worse standalone and cannot be admitted by averaging (+0.0562 equal weight, no
interior optimum in the sweep) NOR by its own Ridge column (weight 0.0261, +0.0134 worse).

**lr was never the deficit** and the dose-response proves it: standalone 11.99 at the untuned
1e-3, 12.56 at 3e-4, 13.11 at 1e-4, 12.78 at 3e-4 with depth 4. Best of eight is 11.78 against a
10.35 bar. The encoder-diversity IDEA is the right one for a session with time to tune; this
architecture at this scale is not the vehicle.

## Also closed 2026-08-05

* **The rate-HMM structural repair.** Both defects are real — the E2 node gap is 200 ft at p05, p50
  and p95, so `rate_sigma/dr = 5.657` on every node and the one-cell band truncates a kernel 5.7x
  wider than itself. Repairing it makes things WORSE on every arm, all five folds: band-only
  +5.0795, both +7.6516, best variant +4.2103. **The band limit is a load-bearing REGULARISER, not
  a bug:** at a 200 ft node the declared kernel is 0.0283, the truncation realises 0.00407 and the
  true increment sd is 0.01502, so the incumbent is 3.69x too tight and the repair 1.88x too loose
  — and too loose lets the rate chase emission aliases, which is settled statement 1 in `AGENTS.md`
  reappearing exactly where it predicts. The OU term is separately misspecified: **89 percent of the
  rate's variance is BETWEEN wells**, so reverting to a global zero pulls every well toward flat and
  that arm diverges to non-finite.
* **The top public notebook**, honest CV 10.4451. See the fork-wall section below.

---

# THE ONLY DECISION THAT STILL MATTERS: WHICH TWO TO SELECT

Selection is a WEB-UI action. The Kaggle CLI exposes no command for it, so it is always the
owner's. **Never select a submission that has not returned a public number** — a returned score is
the only proof a kernel executes on hidden data.

Everything below is decided by `E[min]` over 4,000 well-cluster bootstraps of 200 wells, the
objective Kaggle actually scores, from `reports/probe_pick_pair_maxof2_2026_08_04_afternoon.json`.
Kaggle scores each pick independently and keeps the BETTER one, so the objective is
`E[min(RMSE_A, RMSE_B)]`, never the half average.

## BOTH RETURNED, and the controlled experiment is decisive about the disputed block

```
                     single-path            multipath
  3 seeds   55231323 CV 7.2732 / 6.731   55234752 CV 7.2038 / 6.891   multipath +0.160 board
  9 seeds   55244617 CV 7.2113 / 6.618   55244616 CV 7.1593 / 6.818   multipath +0.200 board
```

**`55244617` at 6.618 is the best board score this project has produced.** And at BOTH seed counts
the multipath block costs about 0.16-0.20 ft of board while CV says it gains 0.05-0.07. The
discrepancy is +1.50 and +1.62 sd of the board's 0.1613 ft difference noise.

**These are NOT two independent replications and must not be read as one 2-sigma result.** The
public set is a FIXED ~52 wells, so both comparisons score the same wells and share whatever those
wells happen to favour. It is one measurement at about 1.5 sd, repeated on the same sample.

Rule 1 still says rank by CV, and CV is 773 wells cross-fitted with p_worse 0.0053 against the
board's 52. But the disagreement is now consistent, in the same direction, at two seed counts, and
it is exactly the kind of thing the second pick exists to hedge.

## Recommended pair, and it is better motivated than before

**`55244616` + `55244617`.** One is the best CV we own (7.1593), the other is the best board we
own (6.618), and they differ by EXACTLY the disputed block at matched seed count. E[min] 7.1155
against 7.1075 for the two-multipath pair, so the hedge costs 0.008 ft of expected score and buys
insurance on the only component whose sign is in dispute. Taking two multipath kernels would
concentrate the bet on precisely the thing the board argues against twice.

## If `55244616` and `55244617` both returned normally

| pick pair | E[min] | reading |
|---|---:|---|
| `55244616` + `55244617` | 7.1155 | **RECOMMENDED.** Same nine seeds with and without the multipath block. Costs 0.008 ft against the pair below and hedges the one open doubt. |
| `55234752` + `55244616` | 7.1075 | Lowest E[min], but BOTH carry the multipath block, which is the thing the board is mildly negative on. |
| `55234752` + `55223770` | 7.1604 | Fallback if either new submission failed to return. |

**Why the recommended pair is not the lowest-E[min] pair.** The second pick is worth at most
-0.0108 ft on modelling grounds, so it should be spent on the risk that actually exists. That risk
is no longer "does the kernel execute" — all four execute. It is "is the multipath block helping on
hidden data", where CV says yes at p_worse 0.005 and the board says maybe not at +1.55 sd the other
way. `55244616` and `55244617` are the same nine seeds WITH and WITHOUT that block, so their board
difference is a second, independent read on it. **Read it before selecting**, and note it is one
more 52-well comparison at sd 0.1613, so it can inform the pick but cannot settle the question.

## If only one of them returned

Select the one that returned plus `55234752` if the returner is `55244617` (keeps one of each
kind), or plus `55223770` if the returner is `55244616` (`55223770` is the best non-multipath
alternative with a returned score).

## If neither returned

`55231323` + `55223770`, E[min] 7.2176. This is the best pair among submissions that have returned
a public score, and it is what the max-of-two objective picks independently of the CV-and-board
reasoning that first recommended it. Two routes, same answer.

## Do not select

`55216428` — the shared slug `rogii-seqalt-gru-models` was versioned twice on 2026-08-03, so a
rerun would load weights it was not verified against. `54853374` and `54851870`, the 2026-07-20
pair at 26.939.

## On 2026-08-05 after the close

Run `notes/POST_DEADLINE_HARVEST_RUNBOOK.md`. Every sub-6 team has declined to disclose while the
competition is live; that expires at the deadline.

---

# 2026-08-04 AFTERNOON. Nothing left clears 0.10 ft, and that is now measured rather than argued.

**Preflight green: 8.2826 / 8.7300 / 8.9641.** Seven levers were measured this afternoon. One is
real and below the floor, one is destructive, four are washes, and one deputy claim was refuted.

| lever | cross-fitted delta | CI95 | verdict |
|---|---:|---|---|
| **9 psr4 seeds (3 -> 9)** | **-0.0438** | [-0.1097, +0.0255] | real, below the 0.05 floor |
| post-processing cell re-freeze | +0.0689 | [+0.0164, +0.1240] | DESTRUCTIVE, 5 of 5 folds worse |
| ramp re-freeze | -0.0087 | [-0.0303, +0.0137] | wash |
| quadratic ramp term | +0.0053 | [-0.0039, +0.0146] | mild negative |
| blend as a 3rd datum substrate | +0.0082 | [-0.0265, +0.0442] | REFUTED, see below |
| diversity-selected 9-member pool | +0.0305 | [-0.0164, +0.0799] | wrong sign, 2 of 5 folds |
| weight averaging / SWA | n/a | | seeds are not mode-connected |

**The nine-seed path is BUILT, VERIFIED and UNSUBMITTED.** Pooled ramped **7.203837 -> 7.159317**.
Kernel `notebooks/pf_multidiv_setk_multipath_psr4avg9_trustdatum`, dataset staged as the NEW slug
`rogii-seqalt-gru-psr4avg9` (45 models load through the shipped loader, all 12 staging checks
pass), port verified at mean **8.518e-04 ft** with ZERO rows over the 0.01 bar and correlation
1.00000000, constants derived through the same checked path with all three anchors at 0.0 ft, and
every structural integrity bar passing. Runtime is **5.8 percent of the limit at 300 wells**.

It needs no inference code change at all: `load_fold_models` recurses from the dataset root and
`predict_well_drift` averages whatever it is handed, so nine seeds is a DATASET change. That is
what separates it from `55234752`, which introduced a second Ridge, projection and matcher pass.
**Not uploaded, not pushed, not submitted — owner's call.** It does not clear the bar; it exists so
the option does.

**The seed axis is now closed with the curve validated past its support.** The pre-landing forecast
was -0.034 [-0.0498, -0.0188] and the realisation is -0.0438, a miss in the conservative direction,
so the saturation curve extrapolated correctly beyond n<=6. Sub-additivity with the multipath lever
is confirmed and small: -0.0563 alone, -0.0445 on top of it, interaction +0.0117. The six new draws
decorrelate exactly as the shipped three do, mean pairwise error correlation 0.7755 against 0.7875.
The registered worry that the ensemble and the correction are substitutes does NOT continue past
three seeds: rho goes 0.2490 -> 0.2551 and the correction gets STRONGER, -0.1198 -> -0.1272.

**The post-processing result is the one to carry forward.** The frozen cell `(0.88, 6, 0.78)` sits
rank 6 of 80 and 0.0088 ft from the in-sample argmin, and re-selecting it costs **+0.0689 ft**
cross-fitted, worse in 5 of 5 folds, with the five folds picking five different cells. That 0.0088
is optimism, not headroom. The 2026-08-03 retune is still worth -0.114 ft on the current path, so
it ported cleanly through a structural change to the correction; it simply cannot be improved again.

**A deputy claim was refuted before it was banked, and it is rule 9 in its purest form.** An audit
reported the pre-projection blend is far more decorrelated from the shipped path (0.5377) than the
GRU-free sub-path (0.7218), read rho 0.2491 -> 0.2660 on 260 wells, and converted that to about
-0.023 ft through a ratio borrowed from a different arm. Re-measured on all 773 wells in feet
through the deployed ramp: **rho DOES rise, 0.2840 -> 0.2869, and the CV does not follow.** Adding
it as a third substrate is +0.0082; replacing the GRU-free path with it is +0.0505; it makes the
nine-seed path worse. The subset also over-promised in the documented direction, +0.0169 of rho on
260 wells against +0.0029 on 773.

**`augmentation_induced_ensemble_diversity` is CLOSED, all three mechanisms, each for its own
reason.** Weight averaging is dead because the seeds are not mode-connected: `set_seed(seed)` then
a fresh `Seq1DNetAlt` per fold means nine independent inits, cross-seed weight cosine **0.049**,
which is BELOW two fresh random inits at 0.095, and the 9-way weight average scores **102.50 ft**
against the prediction average's 10.37. Snapshot ensembling is structurally impossible: `train_fold`
writes one best-validation state after the epoch loop and a repo-wide search finds zero
epoch-indexed weights. Diversity-encouraging selection is a clean negative at +0.0305, and both
easy explanations are refuted — exhaustive enumeration of all `C(13,9)=715` subsets reproduces the
greedy path at 0.0 ft, and the pairwise error-correlation matrix transfers train-to-validation at
Spearman 0.915-0.952. The frontier is now a number: **0.998 ft of CV per unit of member pairwise
error correlation**, so clearing 0.05 ft by decorrelation alone needs members driven from 0.7825
to about 0.73, and even a hindsight oracle reaches only 7.3781 against the fixed nine's 7.3723.
Successor axis opened: `permutation_aligned_weight_fusion`.

## THE FORK WALL, MEASURED FROM THE INSIDE — and one unresolved hazard in it

`reports/physv48_honest_cv_2026_08_04.json`. The most-voted notebook in the competition,
`evgendvorkin/rogii-physics-lb-7-872-v48` at public 6.361, is the source of the 1,146-team wall in
[6.36, 6.50). Its predictions had never been priced on our protocol. They are now.

**Protocol equivalence at zero compute:** its own ablation baseline is 15.91 and our carry-forward
over the same 3,783,989 rows is **15.9099**. The same instrument to 0.0001 ft, so every CV it
prints reads directly against our ledger.

| its component | CV on our metric | vs our 7.159 |
|---|---:|---|
| physics selector | **10.4451** measured here | +3.29 |
| ML Ridge OOF | 10.42 self-reported | +3.26 |
| learned trajectory | 9.21 self-reported | +2.05 |

**Its 6.361 is not produced by its forward model.** Every component is 2 to 3 ft above our deployed
path, and the board number comes from blending three mediocre components plus post-hoc per-well
machinery. This confirms the "honest CV near 10" reading from the notebook's own arithmetic rather
than from inference, and it is the strongest evidence yet that the wall is not what its public rank
suggests.

**An unresolved hazard, recorded as a hypothesis and NOT as a fact.** The published v48 carries an
unguarded top-level block ("Q0522") that applies a flat +0.522 ft to well `00e12e8b` and `raise`s
unless the submission has exactly **14,151 rows**. Our own audit confirms 14,151 rows is the
VISIBLE three-well sample — all four of our kernels produce exactly that interactively. A scored
rerun against ~200 real wells cannot match it, so a fork submitting v48 unmodified would die on the
rerun rather than merely regress.

**Why this is not yet established.** A kernel that raised would never have returned a public score
at all, and 6.361 is displayed, so either the badge is from an earlier version or the guard does not
fire the way it reads. The agent notes the author's last submission (2026-08-03 20:18) PREDATES the
currently published version and his team sits at public **6.405, rank 612**, not 6.361 — which
favours the stale-badge reading. **Do not plan around the wall collapsing.** It is recorded because
it is checkable after the deadline, not because it is actionable now.

**Do not ship its selector, and it is not a rule-2 mis-kill.** Blend-add **+0.0476** against a
+0.0063 rolled control, despite an err-corr of 0.7042 against our blend that is genuinely
decorrelated and well inside the bar. The mechanism is its **0.9616** err-corr against
`pf_selector`, which the deployed path already carries at weight 0.12 downstream of the Ridge — the
same signal entering twice. Its selector IS `src/pf_frontier.py`, every constant verbatim
identical, both descending from `needless090/lb-8-860-rogii-sel15-256seeds`.

**A permanent external blocker worth recording:** 40 percent of its final prediction is third-party
boosters fitted on the same 773 train wells, so nobody outside those dataset authors can honestly
cross-validate the full path. That is why the selector was priced and the full path was not.

## THE SELECTION PATH CLEARS 0.10 ft EVEN THOUGH NO SINGLE LEVER DOES

`scripts/probe_pick_pair_maxof2_2026_08_04.py` was ranking a STALE MENU. It carried twelve
candidates and was missing the four newest paths, **including both currently recommended picks**,
so every earlier run chose from a menu that excluded the best submissions we own. Extended and
re-run over 4,000 well-cluster bootstraps of 200 wells, the host's stated private size.

Expected private RMSE, singles:

| candidate | E[RMSE] | | candidate | E[RMSE] |
|---|---:|---|---|---:|
| `NEW_avg9_multipath` | **7.1182** | | `55231323` | 7.2295 |
| `55234752` | 7.1640 | | `55223770` | 7.3293 |
| `NEW_avg9_single` | 7.1698 | | `55221568` | 7.4638 |

Under `E[min]`, which is what Kaggle actually scores:

```
55231323 + 55223770               7.2176   best pair among RETURNED submissions, i.e. today
55234752 + 55223770               7.1604   once 55234752 returns
55234752 + NEW_avg9_multipath     7.1075   best available at all
```

**7.2176 -> 7.1075 is -0.1101 ft of expected private RMSE**, and it splits almost evenly into
`55234752` becoming selectable (-0.057) and shipping the nine-seed kernel (-0.053). Two routes now
agree on the pick pair: among submissions that have returned a public score, the max-of-two
objective independently selects `55231323 + 55223770`, which is what CV-and-board reasoning already
recommended. The second pick itself is still worth almost nothing, -0.0108 at best, consistent with
the earlier -0.008.

**SUBMITTED 2026-08-04, both remaining slots, owner approved.** Dataset
`wguesdon/rogii-seqalt-gru-psr4avg9` uploaded as a NEW slug, so no already-submitted kernel is
repointed and both current picks still read what they were verified against.

| kernel | ramped CV | why |
|---|---:|---|
| `rogii-setk-multipath-psr4avg9` v1 | **7.159317** | best CV we own |
| `rogii-setk-gruramp-psr4avg9` v1 | 7.211347 | no second matcher pass; its whole inference lineage already returned 6.731 |

The single-path one exists because the seed lever is worth MORE without multipath (-0.0563 against
-0.0445, the two being sub-additive) and because it is the candidate to reach for if the multipath
inference code misbehaves on the rerun.

## 55234752 RETURNED 6.891, AND THE PRE-REGISTERED PORT-BUG TEST CLEARS IT

Worse than `55231323`'s 6.731 despite a better CV, 7.2038 against 7.2732. The handoff registered
this test in advance, so it was run rather than eyeballed.
`scripts/probe_cv_to_lb_refit_2026_08_04_pm.py`.

```
leave-one-out residual of 55234752 against the other eight   +0.237 ft = +1.10 sd
  four other points have LARGER residuals: -0.242, -0.236, -0.221, +0.258
paired against 55231323, which is 55234752 minus exactly one block
  CV delta          -0.0694
  board PREDICTED   -0.0897
  board OBSERVED    +0.1600
  discrepancy       +0.2497 = +1.55 sd of the board's 0.1613 ft difference noise
```

**VERDICT: NOT DISTINGUISHABLE FROM BOARD NOISE.** The kernel log rules out a gross port bug
independently: both Ridges load their correct coefficients and the two matcher passes produce
genuinely DIFFERENT shifts, 1.069 ft and 0.829 ft, so the second path is real rather than a copy.
A block whose CV gain is real at p_worse 0.005 can still land 1.5 sd the wrong way on 52 wells.

**THE SLOPE IS UNSTABLE AND HAS FALLEN TWICE: 1.674 at six points, 1.5818 at seven, 1.1711 at
nine**, r 0.8912, residual sd 0.2119. A foot of CV buys 1.17 ft of board, not 1.67. **Bronze at
6.408 now implies CV about 6.92, not 7.15**, so every target priced off an older slope was
optimistic. Do not quote this slope to three digits.

## The `w_global` axis is CLOSED, and it went the opposite way from the hypothesis

Five arms at the compound recipe, three seeds at `w_global 0.0` and two at `0.1`, against the
shipped `0.25`. Every comparison holds the SEEDS FIXED so the 0.051 ft spread cancels.

| matched seeds | `w_global` | uncorrected CV | delta |
|---|---|---:|---:|
| 42, 7, 1337 | **0.25** | **7.42651** | — |
| 42, 7, 1337 | 0.0 | 7.51009 | +0.0836 |
| 42, 7 | **0.25** | **7.46014** | — |
| 42, 7 | 0.1 | 7.56852 | +0.1084 |
| 42, 7 | 0.0 | 7.55069 | +0.0906 |

All five new single-seed arms also lose to their matched 0.25 counterpart. The hypothesis was that
with the datum corrected TWICE downstream, gradient spent on absolute TVT upstream is waste. It is
refuted for a reason consistent with everything else here: **the downstream correction only reaches
rho 0.28, so the base still has to carry most of its own datum.** `w_global 0.25` is an INTERIOR
OPTIMUM, not a point on a slope toward zero, and the earlier 1.0 -> 0.25 gain does not extrapolate.
The cell was genuinely untried — `w_global 0` had only ever been observed jointly with `w_shape` 10
or 20, both independently known-bad — and it now loses on its own.
`reports/screen_wglobal_axis_2026_08_04.json`.

**RAN AND CLOSED: five `w_global` arms at the compound recipe**, launched 13:42 UTC,
`ml.g5.2xlarge`,
about 113 min each (the 06:21 batch ran 07:51 to 09:44 BST; the 60 min figure was a misread of CreationTime). Three seeds at `w_global 0.0` and two at `0.1`, against the shipped `0.25`
which has nine banked seeds as its comparator. The axis was last swept at `psr0 / gr50`, before
the two levers worth -0.25 and -0.07 landed, and `w_global 0.25` was inherited into the current
recipe rather than re-derived for it. `w_global 0` looks closed and is not: it has only ever been
observed jointly with `w_shape` 10 or 20, both independently known-bad. Collect with
`scripts/collect_gru_fleet_2026_08_03.py --since 2026-08-04-13-42`; the five tuples are registered.

## Two reproducibility traps found the hard way, now in AGENTS.md

* **The `0.0 ft` anchor bars only hold at ONE BLAS thread.** The GRU-free path reproduces its
  banked array at exactly 0.0 under `OMP_NUM_THREADS=1` and at **4.366e-11** under 2 or 4, because
  thread count changes the reduction order in the positive Ridge solve. Every `!= 0.0` gate then
  aborts on a correct checkout with a message blaming the base set.
* **A banked `pred_*.npy` is an anchor, not a scratch file.** The nine-seed ladder rewrote
  `pred_psr4avg3.npy` from its own `n=3` point — same average, different float32 accumulation
  order — which moved the Ridge solution by 4e-11 ft and broke the shipped kernel's constant
  derivation until the canonical file was regenerated at one thread.

## Corrections to this file's own strategic read, from fresh board and forum evidence

* **The fork wall is 1,146 teams in [6.36, 6.50), not 1,391**, and it GREW over 24 hours from
  1,097. The 1,391 is a mis-stated band; [6.36, 6.55) gives 1,364. `AGENTS.md`'s bands are right.
* **It is not a pile of identical scores.** In [6.20, 6.80) the most duplicated single score is
  6.467 with 41 teams, and 41 is the TOTAL across every score shared by 20 or more teams. The band
  is a near-continuum at 12-17 teams per 0.001 bin. These are tuned forks, not copies, so "they
  collapse together on the rerun" is weaker than the count suggests.
* **RETRACTED: the public-minus-CV gap is not a diagnostic of honesty.** GG Ayo published the first
  external paired CV/LB ladder (discussion 732455): CV 6.205 -> 5.982 against LB 8.115 -> 7.377, so
  his board is WORSE than his CV by +1.21 to +1.91. Cyrus reports -1.82. Ours is -0.54. Across four
  teams the offset spans **3.7 ft**. The inference that our -0.63 gap against the fork lineage's
  -3.6 predicts a favourable shakeup does not hold. Our own within-pipeline fit, slope 1.5818 at
  r 0.9382, is untouched and stays valid for pricing our own targets; only the cross-team use dies.
* **Tucker Arrants disclosed his protocol and it closes the escape hatch.** Discussion 731550,
  2026-08-03: he confirms a RANDOM, non-spatial split, the same family as our `GroupKFold(5)` by
  well, not the row-wise holdout that produces our 5.2948. So his CV 5.13 is comparable, and it
  sits BELOW our perfect-per-well-constant-datum oracle of **5.31**. His SHAPE is better than ours,
  not only his datum. Not actionable in the time remaining; it reprices the gap.
* Board at 13:38 UTC, 6,118 teams: gold 5.521, silver 6.356, bronze 6.408, us 6.731 at rank 1,830.
  The public frontier's last 24 hours of motion was seed count: `evgendvorkin/rogii-physics` at
  6.361 diffs against our 08-03 copy in exactly two places, `range(4)` -> `range(16)` and one
  multiplier.

## State

Cross-fitted CV **8.2251 -> 7.204** over two days. Public **7.581 -> 6.731**.

| submission | what | CV | public |
|---|---|---:|---:|
| `55234752` | + estimate-space datum averaging | **7.204** pooled | PENDING |
| `55231323` | + 3-seed average of the psr4 base | 7.2732 | **6.731** |
| **SELECTED** `55223770` | + post-processing retune | 7.37230 | 6.992 |
| **SELECTED** `55221568` | setK-keeponly + psr4 compound + ramp | 7.5228 | 6.888 |
| `55218515` | + prediction-start resampling x2 | 7.68599 | 7.064 |

### THE ONE ACTION THAT IS FREE AND IS NOT DONE

**Swap `55221568` out for `55231323`.** `55231323` is better on BOTH axes: CV 7.2732 against
7.5228, and public 6.731 against 6.888. Selection is a WEB-UI action; the Kaggle CLI has no command
for it, so it is always the owner's. If `55234752` returns a normal score it should displace
`55223770` as well, putting both picks on the two best CVs we own.

The pair criterion is in `AGENTS.md`: Kaggle scores each selected submission independently and
takes the BETTER one, so the objective is `E[min]` over the private draw, not the half average. The
second pick is worth at most 0.008 ft on modelling grounds, so it should hedge a kernel failing on
the rerun. Both picks must have RETURNED a public number, which is the only proof a kernel executes
on hidden data.

## What is running or banked and NOT yet used

* **Six psr4 seeds finished on AWS** (`2026-08-04-06-21-*`, all Completed) and are UNCOLLECTED.
  Collect with `scripts/collect_gru_fleet_2026_08_03.py --since 2026-08-04-06-21`; their tuples are
  already in `ARM_BY_TUPLE`. Folding them in takes the base from 3 to 9 seeds.
  **Priced before they landed: worth -0.034 ft ramped, CI95 [-0.0498, -0.0188].** Below the 0.05 ft
  floor and a fifth of the board's 0.1613 ft sd on a difference, so it will not be visible on the
  leaderboard. Marginals: 4th -0.0133, 5th -0.0080, 6th -0.0054, 7th -0.0038, 8th -0.0029, 9th
  -0.0023. `reports/probe_seed_saturation_2026_08_04.json`.
* Everything else measured this session is closed or below floor. See QUEUE.md.

## 2026-08-04 afternoon: the last open axis is CLOSED, and a successor is open

`augmentation_induced_ensemble_diversity` asked whether ensemble diversity can be deepened at FIXED
member count from artefacts already on disk, at no training cost. Three mechanisms, all dead, three
different reasons. `reports/axis_ensemble_diversity_fixed_members_2026_08_04.json`.

**1. Weight averaging / SWA is not an operation that exists for this bank.** The nine psr4
replicates share no initialisation: `aws/src/train_seq_alt.py` calls `set_seed(seed)` with the arm's
own seed and then builds a fresh model inside every fold. Cross-seed weight cosine over 307,394
parameters is **0.049**, the same as one seed at an adjacent fold (0.048) and BELOW two FRESH random
inits (0.095). The linear interpolation midpoint between two seeds scores **86.35 ft** against
10.56 at the endpoints and **14.63 for carry-forward**, and the 9-way weight average is 102.50
against the 9-way prediction average's 10.37. Not mode connected, so nothing to average. Replicated
on fold 3: barrier +82.20 ft, midpoint 91.17 against endpoints 8.97 and 10.00, 9-way weight average
102.63 against a prediction average of 9.11. No BatchNorm confound, checked rather than assumed:
GroupNorm/LayerNorm only, zero running buffers in the checkpoints, so no recalibration pass was
skipped. `reports/probe_ckpt_basin_swa_2026_08_04.json`,
`reports/probe_ckpt_basin_swa_fold3_2026_08_04.json`.

**2. Snapshot ensembling is STRUCTURALLY IMPOSSIBLE from disk, not merely unattractive.**
`train_fold` holds one best-validation state in memory and writes it once after the epoch loop. All
46 GRU checkpoint directories hold exactly `cnn_1d_fold0..4.pt`; a repo-wide search finds **zero**
epoch-indexed weights. Reopening needs a retrain with periodic saves.

**3. Diversity-encouraging SELECTION is a clean negative, judged corrected with a rebuilt bank.**
45-arm pool, subset chosen inside each outer training fold to minimise mean pairwise error
correlation under a pre-registered 1.05x quality floor, at the comparator's own member count of
nine. **Ramped +0.0305 ft against the fixed nine seeds**, CI95 [-0.0164, +0.0799], 2 of 5 folds
improved; per-fold deltas -0.012, +0.092, +0.003, +0.104, -0.042. **All sixteen cells of the
(k, floor) surface lose**, +0.027 to +0.246 ft, as do the three same-recipe seed-pool cells, and
the pre-registered cell is the best of the nineteen, so no untried setting is hiding a win. No rule-13 inversion: the correction is worth -0.1272 on the
comparator and -0.1223 on the candidate. `reports/probe_diversity_selection_2026_08_04.json`,
`reports/score_divsel_corrected_2026_08_04.json`.

**Both easy explanations for that null are refuted, which is what makes it worth having.** The
search did not fail: exhaustive enumeration of all `C(13,9) = 715` admissible subsets per fold
reproduces the greedy PATH byte for byte, max absolute difference 0.0 ft, so the corrected verdict below covers both. The statistic is not noise: the
pairwise error-correlation matrix transfers at Spearman **0.915 to 0.952** train to validation, and
standalone RMSE at 0.856 to 0.910. The rule is not even under-powered at decorrelating: at a loose
floor it reaches mean pairwise error correlation **0.6818** against the fixed nine's 0.7825.

**The frontier is measured, not asserted, and decorrelation is sold at 3.4 times its value.**
Loosening the quality floor from 1.05 to 1.25 at nine members is the clean read, since member count
is held fixed and only the admissible pool changes. It buys mean pairwise error correlation
**0.7831 -> 0.7284**, and the equicorrelated-average identity calibrated on the observed 3-to-9 step
prices that 0.0547 drop at **-0.0546 ft**. The arm is **+0.1302 ft worse**. The slope is
**0.998 ft of CV per unit of member pairwise error correlation**, so clearing the 0.05 ft shipping
floor by decorrelation alone needs the members driven from 0.7825 to about **0.73** — reachable on
this bank only at several times that cost in accuracy. Diversity and quality point the SAME way on a
bank whose best recipe is also its most decorrelated. Inside the nine same-recipe seeds,
min-correlation selection LOSES to random at every member count, which is what selecting on a
near-degenerate statistic looks like.

This is a reusable fact about ensembling a saturated bank, not a fact about this competition:
measure the correlation spread and price it with the member-count step you already have, before
building any selection machinery.

**The axis's own substitution worry is retired.** rho fell 0.2777 to 0.2490 from one seed to three
and that looked like the ensemble eating the correction. It does not continue: 3 to 9 takes rho
**0.2490 -> 0.2551** and the correction gets STRONGER, -0.1198 -> -0.1272, so the ramped gain
-0.0550 is essentially the whole uncorrected -0.0543. The nine-seed realisation also beat its own
pre-landing forecast of -0.034, in the conservative direction.

**Successor axis open: `permutation_aligned_weight_fusion`.** The barrier closes NAIVE
co-ordinate-wise averaging, not weight-space fusion; hidden-unit permutation symmetry is precisely
why independent inits look unrelated. First experiment, its mandatory identity control and its
11.1 ft bar are written out in `harness/forward_axes.json`. Nothing was submitted and no AWS job was
launched for any of this.

## The CV-to-board fit, seven points now

```
board = 1.5818 * CV - 4.8324      r = 0.9382,  residual sd 0.1755 ft
```
A foot of CV buys 1.58 ft of board. Bronze (6.409, rank 610 of 6,109) implies CV **7.107**, i.e.
-0.17 from 7.2732. Chris Deotte's rank-100 6.112 implies CV 6.919, i.e. -0.35.
`scripts/probe_cv_to_lb_target_2026_08_04.py`.

**But the public rank badly understates our position.** 1,391 teams sit in a public-notebook fork
wall at [6.36, 6.50), and two independent parties put that lineage's honest CV near 10 against its
LB 6.4, a public-minus-CV gap of about -3.6 ft where ours is -0.63. Public rank 1,871 is roughly
266 real teams plus a wall that should collapse on the private rerun. Rank 4 (public 5.078) put it
plainly: "public leaderboard ranking is irrelevant. Get your CV as low as possible and pray you
survive the shakeup."

## What moved CV, and the shape they share

| change | worth | what was wrong |
|---|---:|---|
| shape-supervised sequence loss | -0.11 | 58.6 percent of absolute TVT is the datum `trust_datum` corrects downstream |
| `gr_filter` 50 -> 9 | -0.07 | the query log was smoothed at 50 ft against a 1 ft reference |
| virtual prediction-start resampling | -0.25 | 773 fixed training samples were an arbitrary choice, not a constraint |
| post-processing retune | -0.14 | the degree-4 projection mis-conditioned the path for the datum matcher |
| 3-seed averaging | -0.10 | resampling decorrelates seeds (0.77-0.80 vs the usual 0.89-0.92) |
| estimate-space datum averaging | -0.06 | the pipeline's smoothing fights its own matcher |

**Every one came from a component optimising or consuming the wrong quantity. None came from more
capacity, more bases or more blending**, and recombination over all 60 banked bases is capped at
7.9537 in sample, so none of it could have.

## Traps this session added, each of which cost something

* **A base inert in one fit is not inert in another.** `cnn_1d_v1_avg3` is exactly 0.0 in the
  seven-base fit and in all five of its folds, which made a prune look free. In the GRU-free
  six-base fit it carries 0.18359, nonzero in every fold, and deleting it moves that path by up to
  7.554 ft. The second matcher substrate depends on it.
* **Never re-version a dataset slug a selected submission reads.** `rogii-seqalt-gru-models` was
  versioned twice on 2026-08-03, which is why `55216428` must never be selected: a rerun would load
  weights it was not verified against.
* **The Kaggle daily quota is UTC.** A submission at 23:52 local on the 3rd 400s against the 3rd's
  spent quota; the same call at 00:00:25 UTC succeeded.
* **Submitting is code-competition only.** `kaggle competitions submit -f <local path>` returns a
  bare 400. The form is `-k <kernel> -v <version> -f submission.csv`, where `-f` is the name the
  kernel WROTE.
* About **0.0035 ft of the shipped CV is L-BFGS-B convergence state**, not model, and an
  information-free perturbation moves corrected CV by 0.002. That is the noise floor for
  micro-comparisons; do not read anything smaller.

---

# SESSION 2026-08-03 EVENING — three levers on the sequence base, 7.9330 -> 7.6860

**SUBMITTED `55216428`**, kernel `wguesdon/rogii-setk-gruramp-v1` v1, `setK-keeponly` plus the
shape-supervised GRU plus the ramp, frozen CV **7.82033** against the incumbent `55210028`'s
7.93299 under the identical leave-one-fold-out rule. Owner approved. Three of five daily
submissions used; `55209717` and `55210028` are still PENDING after seven hours, which matches the
forum report that the queue runs long near the deadline.

**Submitting is `-k <kernel> -v <version> -f submission.csv`, NOT `-f <local path>`.** This is a
CODE competition and a file submission returns 400. That fact lived only in `archive/` and cost
three failed attempts to rediscover. It is now in `AGENTS.md`.

## The three levers, all on the GRU sequence base, all screened against the rolled null

| lever | arm | real minus null | shape gain | standalone |
|---|---|---:|---:|---:|
| loss aimed at the mean-removed residual | `wshape2g025` | -0.1143 | -0.0920 | 12.2202 |
| query GR smoothing 50 ft -> 9 ft | `gr9` | -0.1630 | -0.1361 | 11.9706 |
| **virtual prediction-start resampling** | **`psr2`** | **-0.2439** | **-0.2160** | **10.7860** |
| the resampling CONTROL | `psr0` | -0.0861 | -0.0772 | 12.4912 |
| every seed, width, depth, lookahead arm | 10 arms | -0.007 to -0.041 | | |

**Prediction-start resampling is the session's largest lever and it uses no new data.**
`build_seq1d_sample` derives the split purely from which rows carry `TVT_input`, and train wells
carry `TVT` everywhere, so the prefix/eval boundary of a TRAINING well can be re-cut anywhere.
Every cut is real geology with a real GR log; only the question changes. Cuts are drawn from the
MEASURED real known-fraction distribution (median 0.2600, p10 0.1967, p90 0.3464), which is the
mistake the synthetic bank made and was free to avoid. A well sees a median 61 distinct cuts over
80 epochs. Dose-response monotone over three points; the control reproduces the arm it was built
from within seed noise.

Corrected with its own rebuilt bank: **7.70285** against the shipped 7.97724. Ramp frozen at
`LAMBDA 0.3, MU 1.0`: **7.68599** leave-one-fold-out.

**Query smoothing is a SECOND, independent lever.** The horizontal log was savgol-smoothed at 50 ft
while the typewell it is matched against sits at 1 ft unsmoothed, and the cross-attention between
them IS the learned matcher. `gr9` and `gr25` are flat against each other, so it is the 50 ft
setting that was wrong rather than a fine-resolution effect. Finer BINS are strongly negative
(`gr9step24` -0.0292, `gr9step12` +0.0007): sequence length is the cost and 12 ft bins stay.

**The shape axis has an interior optimum and the lever is de-weighting absolute TVT, not adding
shape weight.** At fixed `w_shape`, dropping `w_global` 1.0 -> 0.25 helps in both available pairs;
at fixed `w_global`, raising `w_shape` past 2 hurts monotonically out to 20. But the three seed
replicates of `wshape5g025` read -0.0963, -0.0670 and -0.0452, a spread of 0.051 against an 0.018
gap between the two best arms, so the WITHIN-axis ranking is not resolvable at one seed each
(rule 13c). The group difference is: shape arms mean -0.0695, plain arms mean -0.0244.

## The leak audit, because a gain this size from a dataset change looks like one

`reports/audit_ps_resample_leak_2026_08_03.json`. **VERDICT: NO LEAK.** The decisive check
reconstructs a validation well's input tensors from the raw CSV with `TVT` STRIPPED and compares
against the full-label build: all ten tensors identical at max abs diff **0.0**, and the
checkpoint's output identical, one validation well per fold. Resample records are drawn from train
ids only, the known-fraction pool excludes validation ids, validation samples are built at the true
`TVT_input` boundary, and both OOF arrays are complete, NaN free and aligned to all 3,783,989 meta
rows by `(well_id, row_index)`.

**The audit's own per-length-bin table is internally impossible and was not used.** It reported
psr2 worse in all five quintiles while better overall, which pooled RMSE over an exhaustive
partition of wells forbids, and its per-bin values sat near 1.8 against an overall 10.8.
Recomputed in `scripts/check_psr2_by_length_2026_08_03.py`, psr2 wins EVERY quintile by -1.34 to
-2.64 and the row-weighted reconstruction reproduces 12.4912 and 10.7860 exactly. The truncation
hypothesis dies in the strongest direction available: the 9 wells that OVERFLOW the 9,216-row
future capacity improve MORE than the rest, -3.69 against -1.66.

The one honest caveat the audit leaves standing: the OOF is evaluated at TRUE cuts, so the gain is
measured at the real prediction-start distribution, but nothing here establishes that the hidden
test set's cut distribution matches the training draw.

## Running at the time of writing

Eight AWS arms launched 17:26 and 17:32 UTC: five compound arms crossing all three levers
(`psr2 + gr9 + w2g025`, `psr4`, `psr8`, `w5` variant, and an `epochs 160` control for the
cuts-versus-steps confound), plus three seed replicates of `wshape2g025` so the shape axis winner
becomes estimable rather than a single draw. A kernel port of `setK-keeponly + psr2 + ramp` and a
multi-base screen over the banked sequence arms are both in progress.

---

# SESSION 2026-08-03 CONTINUATION (afternoon) — the loss was aimed at the wrong term

Resumed 14:52. `make preflight` green: 8.2826, 8.7300, 8.9641 all reproduce.

## Headline: cross-fitted CV 7.9330 -> 7.8188, from one change to the sequence loss

**`compute_loss` weighted absolute TVT at 1.0, and 58.6 percent of absolute TVT is the per-well
datum that `src/trust_datum.py` already corrects downstream.** Most of the model's gradient was
spent on a quantity another component of the pipeline fixes, while the 28.03 ft^2 of mean-removed
SHAPE that nothing else can touch got the remainder. `w_shape` was added on 2026-08-02 and never
carried to a 5-fold gate on any family.

Eight GRU arms had finished on AWS at 13:23-13:25 and were never collected. All eight collected,
banked at full 3,783,989-row coverage with zero NaN, and screened on the shippable `setK-keeponly`
path with the rolled null beside each. Seeds, width, depth and lookahead land between -0.0071 and
-0.0405 real-minus-null, a 0.033 ft band consistent with the measured seed noise of sd 0.0264. The
two shape-supervised arms land at **-0.0837** and **-0.0963**, and have the LOWEST err-corr against
`cnn_1d_v1_avg3` in the fleet, which is what rule 13a says should protect the datum correction.

| path | uncorrected | corrected | ramped, cross-fitted |
|---|---:|---:|---:|
| shipped `setK-keeponly` | 8.18244 | 7.97724 | 7.9330 |
| **+ `seqalt_gru_wshape5g025`** | **8.03205** | **7.85499** | **7.81881** |

The correction is worth -0.1771 on the new path against -0.2052 on the shipped one, so rule 13a's
tax is present and small: this base returns 81 percent of its raw gain after correction where the
plain GRU returned 27 percent and `gbdtdiv_lgb_huber4` returned a net loss. The ramp still composes,
`mu` positive in 5 of 5 folds, and it wants `mu 1.0-1.2` where the shipped path wants 1.7, so the
shape-trained base has absorbed part of the ramp and the ramp keeps the rest.
`notes/shape_supervision_2026_08_03.md`.

## The honest single-model target is 5.13, and it is not reachable through the datum

Fresh forum and notebook mining, 153 discussions and 33 kernels pulled today. Tucker Arrants,
discussion 731550, states it outright: "a model whose overall CV is 5.13", on a protocol he
confirmed on request as pooled per-row, GroupKFold by well, per-well data only, non-tabular. Live
public LB: shu01 4.608, Tucker 5.078, us 7.581, bronze cut 6.412 with 2,007 teams packed into the
1.17 ft between it and us.

**The datum channel cannot deliver it, and that is now settled four ways rather than three.** A
fourth attack tested whether the rho-0.28 cap is an artefact of aggregating window evidence with a
MEAN. It is not. Median 0.2323, +/-2 ft vote 0.1756, KDE mode 0.1697, mixture MAP 0.1509, summed
raw profiles 0.0626, all against the incumbent mean at 0.2772, and cross-fitted constants pushed
every estimator back toward a mean in all five folds. Replicated on a second profile bank. The
motivating argument is recorded as wrong: rho is scale-invariant so dilution cannot attenuate it,
and what the mean actually buys is 1/sqrt(N) averaging, 0.1011 at one window to 0.2759 at all ~25.

**A canonical fact in `AGENTS.md` was also wrong and is retracted: there is no ~13 ft alias
lattice.** Real/rolled at 12-14 ft is 0.73, i.e. DEPLETED, and competing peaks read straight off
the profiles are flat from 4 to 16 ft. The real structure is a spike inside +/-4 ft at ratio 1.9 on
a near-uniform background at an 11 percent inlier rate.

Pooled 5.444 needs rho about 0.97. Our own best-window ORACLE reaches 0.9094 on real profiles and
0.8925 on ROLLED ones carrying no datum at all. **The teams ahead are not reading this channel
better.**

## Two forum claims tested and refuted, both with numbers

**"U = TVT + Z is linear in MD at R^2 0.99, so the hidden region is one parameter once
`last_known` pins the intercept."** The R^2 replicates at median 0.9923 and means nothing: `U`
swings 90-170 ft along a well, so the median anchored-linear residual is 4.9973 ft and only 59.9
percent of wells reach R^2 0.99 at all. A PERFECT one-parameter slope oracle scores 7.5872, against
the 7.8188 we already hold. The ladder nobody had priced, all anchored at `last_known`:

| anchored family in U | free params | pooled RMSE |
|---|---:|---:|
| line | 1 | 7.5872 |
| line, anchor released | 2 | 5.9639 |
| **quadratic** | **2** | **4.2792** |
| cubic | 3 | 2.9794 |
| degree 4 | 4 | 2.2584 |

The anchor is worth more than a degree of freedom once curvature is allowed.

**"The label is a sparse StarSteer dip annotation, piecewise-linear with ~15 control points, so
every L2 smoothness prior here is structurally wrong."** Priced by exact dynamic programming over a
40-point knot grid, the piecewise family LOSES at equal degrees of freedom: 3 dof gives 3.7612
against the polynomial's 2.9794. The stronger claim that stretches are flat is refuted directly:
`dU/dMD` over 50 ft has excess kurtosis **-0.78**, flatter than Gaussian rather than spiked, and
0.5 percent of rows fall below 0.001 ft/ft. Do not rebuild the estimators around a TV prior.

That probe found something it was not looking for: **passing the TRUE labels through our own
post-processing costs 1.5497 ft.** savgol(17,3) is nearly free at 0.0264; the anchored robust
degree-4 U-projection at blend 0.70 does all of it. That is 19.8 percent of our current CV and it
explains why `probe_setk_postproc_retune` found `PROJ_BLEND` flat at -0.0021: the variance the
projection removes and the signal it destroys are balanced AT OUR CURRENT ERROR LEVEL, so the term
becomes binding only as the model improves.

## Also closed this session

The decomposed Ridge, fitting separate positive-Ridge weights to the per-well level and to the
mean-removed remainder, is a WASH at -0.0075 raw and moves SHAPE the wrong way (+0.0131). The tied
control reproduces the deployed Ridge at 0.00e+00, so the null is the estimator's.

## Running at the time of writing

* Eight AWS arms sweeping the shape-supervision axis, launched 14:23 UTC. Five break the confound
  `wshape5g025` left (it moved w_shape 2->5 AND w_global 1.0->0.25 at once) and push to `w_global 0`;
  three replicate the current best on two seeds and one width.
* Four AWS arms wiring `synth-frac` into `train_seq_alt.py`, with a `synth-frac 0` control. The repo
  built a leak-proof forward simulator and 3,000 synthetic wells in July and plumbed the
  augmentation only into `train_cnn_sdf`, which cannot ship; there is no gated result anywhere.
  Tucker Arrants' one public statement about training is "Pre-training on synthetic wells gave a
  decent boost", and hengck23's controlled three-arm experiment says only the arm with a MODELLED
  RESIDUAL NOISE transfers.
* A kernel port of `setK-keeponly + seqalt_gru_wshape5g025 + ramp`, generated and verified but NOT
  uploaded, pushed or submitted. All three are the owner's call.
* The host's own conditional tip, that the lateral's pre-PS GR beats the typewell as a reference
  when the well travels NEGATIVE in TVT. The repo closed the prefix reference UNCONDITIONALLY; the
  host's version specifies exactly when it should work.

---

# SESSION 2026-08-03 — 2 days to deadline

New candidates due **2026-08-03**. Selection locks **2026-08-04**. Deadline **2026-08-05**.

## Final state

| ref | path | cross-fitted CV | public |
|---|---|---:|---|
| **`55210028`** | setK-keeponly + **RAMP** | **7.9330** | pending |
| `55209717` | setK-keeponly | 7.9772 | pending |
| `55192950` | geom pick + correction | 8.1757 | 7.642 |
| `55190292` | seven-base + fine datum | 8.0435 | 7.747 |
| `55184703` | seven-base + coarse datum | 8.1326 | 7.581 |

**Session movement 8.2251 -> 7.9330, -0.292 ft.** Both new candidates were live-verified on Kaggle
before their slots were spent. **Recommended picks: `55210028` plus `55192950`**, which keeps the
pair decorrelated at err-corr 0.9263. Selection is a WEB-UI action, see below.

Measured but not shipped: setK with the window-evidence base at 7.9811 (no inference path, fold
models never saved); setK-keeponly plus the GRU at 7.9608, which does not compose with the ramp;
the geom pick plus its own ramp at 8.1429, a WASH at -0.0328 against the 0.05 bar to displace a
pick, and needing its own constants rather than the four verified lines.

**SELECTION IS A WEB-UI ACTION AND THE OWNER MUST DO IT.** The Kaggle CLI exposes no command to
choose final submissions; `kaggle competitions` has list, submit, files, leaderboard and host-side
commands only. Whoever picks up this campaign must hand the owner the two refs and ask them to
select in the browser before the lock. Do not assume a script can do it.

**The RAMP is the session's second real gain and it needs no new information.** The per-well constant
and slope correlate at +0.7640, so the error is a ramp and the deployed correction has been applying
its per-well estimate in the wrong SHAPE, as a constant. Re-shaping it as
`est * (lambda + mu * x)` with `x = (md - md.min()) / 5000.0` over the eval rows, frozen at
`lambda 0.2, mu 1.7`, gives 7.9372 nested against 7.9772. The control that decides it: a pure
RESCALE with no ramp gives 7.9839, WORSE than shipped, so the shape is doing the work rather than
the magnitude. `md` is fully known at test time, so the coordinate is legal.

**The ramp REPLICATES on an independent stack, which is why it should be believed.** On the geom
path the nested ramp gives 8.1429 against 8.1757 with the mu=0 comparator at 8.1686, so the shape
effect beyond rescaling is -0.0258 there. `feet_from_anchor` wins all five folds on both stacks and
per-fold mu is strongly positive on both. The transfer is DIRECTIONAL, not quantitative: the blend's
frozen constants applied to the geom path give only -0.0156 against the -0.0328 it gets from its own,
because it prefers mu about 1.2 where the blend froze 1.7. The ramp is a property of the ERROR; its
magnitude is stack-specific.

**Why every base-model family failed, stated as a law.** See `AGENTS.md` rules 13a and 13b.
Improving the RAW stack systematically degrades the datum correction, usually by more than it gains,
because the correction's strength tracks `cnn_1d_v1_avg3`'s weight in the Ridge and any strong new
base takes that weight. Three independent measurements: the window-evidence base, the GRU and
`gbdtdiv_lgb_huber4` gained -0.0216, -0.0648 and -0.0618 uncorrected and delivered -0.0039 worse,
-0.0010 and +0.0026 corrected. The two things that DID land avoided this trap: a base that
decorrelates without displacing the CNN (`untried19_ratehmm_public_sg5`, standalone 27.0097 at
err-corr 0.118), and a shape fix to the correction itself.

**The GRU is a KEEP whose gain is mostly eaten by rule 13.** `seqalt_gru_v1_s42`, `Seq1DNet` with
only its dilated-TCN encoder swapped for a bidirectional GRU, beats every banked CNN arm on all four
gate columns, and a single seed beats a three-seed average. But the correction is worth -0.2052 on
the incumbent path, -0.1617 with the GRU ADDED and -0.1414 with it SWAPPED for `cnn_1d_v1_avg3`.
Monotone: the more the GRU displaces the CNN, the worse the trust-gated matcher localises against
the resulting path shape. Swapped it delivers -0.0010 corrected and LOSES the cross-fitted set
choice at +0.0183; added it delivers -0.0165, chosen 5 of 5 folds. Its five fold checkpoints exist
and load, backbone mechanically confirmed, so the family IS shippable whenever a variant earns it.

## Headline: cross-fitted CV 8.0435 -> 7.9811, the first sub-8 result

`setK`, six bases, every one a LEDGER KEEP, with the trust-datum correction rebuilt around the new
path rather than borrowed. The set choice is itself cross-fitted, made on training-fold wells and
scored on held-out wells, and picks setK in 5 of 5 folds.
`reports/score_setk_corrected_2026_08_03.json`, `predictions/pred_setk_trustdatum_2026_08_03.npy`.

```
lgbmede2ratecoupledwindowevidencefeature_2026_08_03   KEEP today, standalone 8.4788, best single
realmlp_v1_s42
cnn_1d_v1_avg3
untried19_lam0.01_sg5
untried19_ratehmm_posterior_sg5
untried19_ratehmm_public_sg5                          KEEP today, REAL-NULL -0.0627
```

Uncorrected 8.1570 against the SEVEN's 8.2251; the correction is worth -0.1759 on this path.
**Not yet ported to a kernel.** That port is the critical path and submission needs owner approval.

## The governing measurement of the session

`scripts/probe_datum_ceiling_2026_08_03.py`. Per-well polynomial fits to our OWN error:

| degree | 0 | 1 | 2 | 3 | 4 |
|---|---:|---:|---:|---:|---:|
| residual RMS | 5.2948 | 3.7081 | 2.8808 | 2.3510 | 1.9950 |

Degree 0 reproduces the oracle datum ceiling exactly. **94 percent of our squared error is smooth
per-well trajectory error**: five coefficients per well over 773 wells, not 3.78 M independent
rows. Everything we build predicts per ROW and has the trajectory imposed afterwards by savgol and
the degree-4 U-projection.

CV is a pure function of rho against the per-well datum, since the residual datum term is
`6.2943*sqrt(1-rho^2)` and the shape terms are fixed at 28.03 ft^2: rho 0.2718 -> 8.04 (today),
0.50 -> 7.60, 0.70 -> 6.94, 0.90 -> 5.96, 1.00 -> 5.2948. **rho is the headline metric for any
datum work.** The datum prize is 2.9303 ft and the deployed correction holds 6.2 percent of it.

## Reading forum reports of "CV 5 from a single model"

`scripts/probe_cv_protocol_comparability_2026_08_03.py`. Our UNCHANGED model scores 4.5162 on the
first 30 percent of the eval span, 5.0674 on the first 40, and **5.2948 under a row-wise holdout
where the well's own labels pin the per-well constant** — the same number as the oracle datum
ceiling, because that is exactly what a row-wise holdout hands over. A reported 5 is therefore
consistent with a protocol difference and does not on its own establish a better model. It is
equally consistent with a competitor who genuinely solved the datum. Both readings imply the same
action, so the ambiguity costs nothing. Do NOT use the protocol reading as reassurance.

## Nine axes closed on 2026-08-03, each with its control

| axis | result | note |
|---|---|---|
| cross-well spatial datum | rho 0.0935 | also killed the 30.7 percent "short-range structure" |
| 132-observable datum screen | nothing beats 0.2718 | floor 0.1124; prefix dip extrapolation dead at \|rho\| <= 0.027 |
| per-well dip from window-shift slopes | rho 0.0500 | oracle on the same parameterisation worth -0.9286 |
| global monotone alignment | closed | crossover at ~0.6 ft shape RMS; we are at 5.10 |
| reference log, 4 candidates | all <= typewell control 0.2752 | pooled basin reference not deployable |
| window readout and pooling | headroom +0.0168 rho | best-window oracle 0.9094 real vs 0.8925 ROLLED |
| ledger resurrection, 26 bases | nothing clears | 38 of 38 verdicts reproduce; zero invalid kills |
| cnn_1d capacity, 8 fold-0 arms | family converged | width and lookahead unimodal, peaks at deployed 96 and 768 |
| WarpMatch-XC v2 | kill VALID | 23.6053 on the KEEP recipe against 23.3652 off it |

**Reopening bar for alignment, explicit and testable: stack shape RMS below about 0.6 ft.** The
lateral emission is sound — prefix-calibrated typewell localises within 1 ft on 83.6 percent of
wells at contrast 0.667, rolled control 6.4 percent at NEGATIVE contrast. The gated-window
estimator wins only because it SELECTS the minority of windows whose local shape error is small;
a global fit has no equivalent and one bad stretch biases its single coefficient.

## Three claims refuted on 2026-08-03, recorded so they stay refuted

1. **"The window matcher stalls because local dip smears the peak."** The opposite. Holding window
   length at 400 ft and bucketing by TVT travel across the window, hit-within-1-ft runs
   7.0 / 12.6 / 24.4 percent low to high, rolled control flat at 5.8. Crossing formation
   MANUFACTURES the evidence; a window that barely moves in TVT sees one bed and cannot localise
   against a log whose beds repeat. This killed a shift-plus-slope rebuild before it was built.
2. **"The datum carries ~31 percent spatial structure within 2,500 ft."** Artifact of pair-weighted
   versus well-weighted variance. Direct pair correlation is inside [-0.011, +0.076] at 0-2,500 ft.
   This retracts the 2026-08-02 self-correction below, which had already flagged the bar as
   mis-specified but kept the 31 percent.
3. **"Several CNN bases may have been killed under the wrong launcher recipe."** All 20 SageMaker
   records carry `resid/5.0/0.0/80`. The launcher-defaults trap is real but PROSPECTIVE; it never
   contaminated the ledger. Anyone launching must still pass the four flags explicitly.

## Calibration to apply to every small margin

Blend-add seed noise on fold 0 is **sd 0.0264 ft**, range [-0.0876, -0.0166] over six banked v1
seeds. A margin under about 0.05 ft measured on a subset is a redraw. This is why the floor exists.

## Marker legality, checked and clear

Train `horizontal_well.csv` carries `ANCC ASTNU ASTNL EGFDU EGFDL BUDA` and the train typewell
carries `Geology`; **none exist at test.** The derived features are nonetheless legal:
`dense_ancc`, `pf_ancc*` and `plane_fit tvtF_*` are fully finite and non-degenerate in the TEST
parquets, because they are surfaces fitted from TRAIN wells' markers and evaluated at the test
well's position, and train labels are available at inference. No leakage, no silent failure. A
pooled basin-wide reference log is NOT deployable for the same reason the features are: it would
need a marker datum for the test well itself.

## The train/test overlap exploit is DEAD, settled by the board

`55194906` returned **7.747**, identical to `55190292`. `55194906` is `55190292` plus the guarded
overlap override and nothing else, so an identical score means the override was a **strict no-op on
the scored set**: no recoverable overlap is present there. That submission was spent to determine
exactly this and it did. Do not build on the overlap; do not expect it on the private rerun.

## Board against CV, and why selection still follows CV

| submission | what | cross-fit CV | public |
|---|---|---:|---:|
| `55184703` | coarse datum correction | 8.1326 | **7.581** |
| `55192950` | geom pick + correction, frozen constants | 8.1757 | 7.642 |
| `55157342` SELECTED | untouched | 8.2251 | 7.726 |
| `55190292` | FINE datum correction, best CV | **8.0435** | 7.747 |
| `55194906` | `55190292` + inert override | 8.0435 | 7.747 |
| `54791000` SELECTED | untouched geom | 8.2826 | 7.899 |

CV and the board are INVERTED across these six. The spread from 7.581 to 7.747 is 0.166 ft against
a measured board sampling sd of **0.1613** on ~52 wells, which is 1.03 sigma. A board this size
cannot rank these models, so rule 1 stands: trust CV, never the public LB.

**Both currently selected picks are our two WORST on CV.** The CV-optimal pair available today is
`55190292` at 8.0435 plus `55192950` at 8.1757, which stay decorrelated at err-corr 0.9263. If the
setK port lands and verifies, it supersedes `55190292` at 7.9811. Selection locks 2026-08-04 and is
the owner's decision.

## Open at end of 2026-08-03

* **Port setK to a kernel and verify it.** The generator inlines `src/trust_datum.py` with a
  round-trip check; port verification must hold the 8.9e-16 standard set on 2026-08-02.
  Submission requires owner approval and has not been requested.
* Running: per-well SLOPE observable screen (the largest unexamined term, 22 percent of squared
  error, never screened), per-well trajectory-coefficient model, matching statistic sweep
  stratified by travel bucket, GBDT family diversity, new neural families, full-capacity retrain
  and seed averaging.
* Selection unchanged pending a scored candidate. Picks remain `55157342` and `54791000`.

---

# SESSION 2026-08-02 — 3 days to deadline

## Claude Code session, 2026-08-02

Preflight green: 8.2826, 8.7300, 8.9641 and DEPLOY7 8.2251 all reproduce.

**Headline: the GR emission is SOUND, and `AGENTS.md` said the opposite for two weeks.** The
recorded framing was "the deficit is the observable, not the decoder" and "the GR emission is
information limited at ~10% inlier". Both are refuted. Scored at the TRUE path with the row set
held fixed, the supplied typewell explains a median 0.43 of eval GR variance and its datum
argmax lands within **1 ft of the truth on 76 percent of wells**, contrast 0.3688 CI
[0.3051, 0.4253], against 0.0012 for a rolled typewell and 0.0381 for a foreign one. That is the
exact inverse of the 90-of-90 Viterbi certificate. `AGENTS.md` and `QUEUE.md` are corrected.

**The gap is now ONE number.** Localisation decays monotonically with the shape error fed in:
76.4 percent within 1 ft at 0 ft, 54.0 at 1.23, 34.4 at 2.46, **18.4 at our 4.91**, with the
rolled control flat at 0.001 to 0.007 at every level. A path at 1 to 2 ft of shape error makes
the datum, 6.2943 of our 8.2251, nearly free. That curve is the governing result for any future
matcher work. `notes/emission_certificate_2026_08_02.md`.

**A per-well datum estimator that works, CV 8.2251 -> 8.1361, NOT YET PORTED.** The window-local
emission signal is sparse but concentrates on an observable: by quintile of peak isolation the
fraction of windows whose argmax lands within 2 ft of the true local offset runs 0.139 to 0.449
at L=800 and 0.159 to 0.487 at L=1600, while the rolled control stays flat near 0.11 across
every quintile. Gating on that and collapsing the trusted windows to one number per well gives a
per-well datum estimate with weighted correlation 0.1971 to 0.2323 against the true datum, where
every tabular datum regression tried today topped out at 0.110. Every choice is cross-fitted:
window-length set, trust metric, quantile, temperature, prior width and the global scalar. 5 of
5 folds improve, datum-gain CI [-0.2210, -0.0137]. Controls with the same 512-rule selection
budget: rolled typewell +0.0968, across-well shuffle +0.0479, cross-fitted global scalar
+0.0158, so real minus control is -0.1858, -0.1370 and -0.1048.

**This clears the `AGENTS.md` bar for displacing a selected pick (beat cross-fit CV 8.2251), and
it is the open action.** It reads only the well's own GR, MD, predicted path and supplied
typewell, so it is legal test-time post-processing and portable in pure numpy. It needs an
inference port verified exactly against the banked OOF, then owner approval for a submission.
New candidates must be submitted by **2026-08-03**.

**Five clean negatives, each with its control.** Global joint (datum, slope) search over the
certified emission, corr 0.099 with the CI spanning zero. Window-local MD-interpolated
correction, 8.2271 against 8.2251, real minus rolled -0.0032. Iterating the trust-gated
correction, +0.0126 with every control in band, because the first pass fixes the datum and
leaves the shape so the emission never sharpens. Log-likelihood aggregation across windows,
corr 0.0801 and real minus rolled the wrong sign. Per-well PATH SELECTION over the blend plus
the seven bases, where the oracle is 5.7307 but every selector is worse than a seeded random
pick, so the ceiling is menu spread rather than information; that also closes the registered
from-scratch beam decoder for within-well self-consistency.

**A void measurement is disclosed rather than buried.** The first datum certificate scored a
shift with a row set that depended on the shift, so it ranked shifts by how many rows stayed in
the reference's TVT support and never read the GR. The tell was exact: real and rolled agreed at
`frac_truth_beats_zero = 0.4320827943078913` to every digit. v2 fixes it and all reported
numbers use v2.

**The prefix is the better reference but cannot carry a search.** It spans a median 727.29 ft of
TVT against a 26.37 ft eval span and beats the typewell by +0.0377 oracle R^2, paired CI
[0.0235, 0.0514], on 64.1 percent of wells. But only 24.8 percent of wells have their eval
region fully inside its support and only 53 of 773 retain support across a +/-16 ft box.

**SUBMITTED 2026-08-02: `55184703`, kernel `wguesdon/rogii-trustdatum-v1` version 1.**
Owner approved. It is the selected pick `55157342`'s exact pipeline plus ONE block, the
trust-gated typewell datum correction, inserted after the U-projection and before the
submission write, so any score difference is attributable to the correction alone. The kernel
was generated by `scripts/build_trustdatum_kernel_2026_08_02.py`, which INLINES
`src/trust_datum.py` and round-trip checks that every module line survives, so the shipped code
is byte-identical to the code verified at 8.9e-16. Kernel completed in 356 s, corrected 3 of 3
wells, mean |shift| 1.289 ft, max 2.311 ft, 14,151 finite rows.

**A free controlled check, outside the CV protocol.** The 3 released test wells are train wells,
so the submission was scored against their real labels with the correction removed by
subtracting its own recorded per-well shifts. Pooled **2.7730 -> 2.6728, -0.1002**. Per well it
is +0.2228, **-0.4228**, +0.0352: one large hit and two small misses, which is the expected
signature of a correlation-0.2 estimator and agrees with the CV gain in sign and magnitude.
Three wells prove nothing on their own, but there is no sign error and no scale blow-up. This is
NOT a leaderboard oracle; `notes/public_lb_identity_2026_08_01.md` settled that the board is not
these wells.

**Selection is unchanged.** The two picks remain `55157342` and `54791000`. `55184703` beats the
`AGENTS.md` bar for displacing a pick on CV, but the bar also requires a returned public score,
and swapping a pick is the owner's decision. Selection locks 2026-08-04.

**End of session state, 2026-08-02.**

The inference port is BUILT AND VERIFIED, and unsubmitted. `src/trust_datum.py` is the single
implementation used by both the CV measurement and inference.
`scripts/verify_trust_datum_inference_2026_08_02.py` compares it against the banked measurement
over 40 wells and passes at max absolute difference **8.9e-16**. The first run differed by
5.7e-07; rather than widen the tolerance the cause was demonstrated to be the harness banking
its profiles as float32, and matching that storage precision collapses the difference to machine
precision. A correctness problem was fixed on the way: the CV probes gated windows at a
population QUANTILE, which is not deployable because a test well's correction would then depend
on which other wells were scored beside it. The threshold is now an absolute constant per window
length and the whole estimator was re-measured under it, giving **8.1326** rather than 8.1361.

Honest numbers, in order of conservatism. Fully cross-fitted rule: **8.1326**, gain -0.0925.
Frozen shipped configuration with cross-fitted shrinkage: 8.0870. At the frozen lambda 0.93:
8.0829. The last two carry the optimism of a rule chosen after seeing the OOF, so 8.1326 is the
number to quote and 8.0870 is what the artifact actually produces on this OOF.

The shape axis was then worked and is where the gap remains. The base bank CAN express the
per-well shape, in sample rank 3 reaching 2.481 ft and rank 5 reaching 1.833 against the
deployed 5.2948. But nothing we own can choose those coefficients legally: fitted on the first
half of a well by MD they fail on its second half (4.272 at best), and the emission has now
failed three distinct ways of supplying them, as a direction, as a path ranker and as a drift
estimator where degree 0 wins 5 folds of 5. The registered next experiment scores emission PEAK
HEIGHT over the bank's top singular directions, which is the one thing not yet tried.

**SUBMITTED `55194906`: fine datum correction PLUS the guarded overlap override.** Owner
approved after the 2026-08-02 forum and notebook mining found the current top public notebook,
`tamerlanomralinov/hahaha-det-agi` at public **6.42**, scoring partly through the same
`_gold_contact_candidate` train/test lookup this repo tore down on 2026-08-01.

**How to read the score, which is the point of the submission.** `55194906` is `55190292` plus one
guarded block and nothing else, and `55184703` returned 7.581 on the same pipeline family. So

    recoverable fraction f  ~  1 - (score_55194906 / score_no_override)^2

If the scored set carries no overlap the override is a strict no-op and the score is unchanged,
which settles the open question negatively at the cost of one submission. If f is near 0.40 the
score lands near 5.9, which is gold range.

**The guard, verified offline before submitting.** Identical MD arrays AND agreement with the test
file's own visible prefix to 0.05 ft over at least 50 rows. Recovery is exact, 0.000000 ft RMSE on
the hidden rows of all three released test wells; impostor ids and missing files are correctly
rejected; on our real submission it moved those rows from 2.4909 to 0.000000. The shipped kernel
reported 3 of 3 wells overridden with mean changes 2.2117, 2.0468 and 2.1013 ft, matching the
offline computation exactly.

**Caveat carried honestly.** This exploits an organiser data error rather than a modelling
insight, and whether the PRIVATE rerun set carries the same overlap cannot be determined from
here. It does not affect any CV number in this repo; the honest pipeline remains CV 8.0435.

**AWS: A/B fold-0 screen launched 2026-08-02 20:09, SHAPE supervision.**

| job | w_shape | everything else |
|---|---|---|
| `rogii-cnn-1d-f0-s42-2026-08-02-20-09-51-174` | **0.0**, the control | identical |
| `rogii-cnn-1d-f0-s42-2026-08-02-20-09-57-605` | **2.0** | identical |

Both carry `--drift-mode resid --w-local 5.0 --w-smooth 0.0 --epochs 80 --seed 42`
EXPLICITLY, because an audit found `launch_cnn_1d.py`'s transmitted defaults
(`direct / 0.0 / 2.0 / 60`) reproduce the `cnn_1d_direct` WASH rather than the recipe behind
every banked KEEP. That trap is still armed for anyone who launches without these flags.

`w_shape` is a new mean-removed TVT term: the residual left after a per-well datum correction.
Rationale: the datum is 58.6 percent of the error variance and is now handled downstream by
`src/trust_datum.py`, so `w_global` on absolute TVT spends most of its gradient on a quantity
already corrected. It is NOT a resolution change; the corrected scale budget puts under 0.6
percent of variance below 400 ft. Verified locally before launch: a pure datum error scores
shape 0.000, a pure shape error scores 5.129, demeaning is exact to 1.6e-08, and `w_shape=0`
reproduces the historic loss bit for bit.

Read it on SHAPE RMSE, not pooled RMSE. A datum-removed model has no datum, so its standalone
number will look bad by construction, and `AGENTS.md` records three wrong kills from judging on
standalone RMSE. If it clears, gate through `harness/gate.py` on blend-add and err-corr with the
within-well-rolled control. Collect from `output/model.tar.gz`, never `output.tar.gz`.

**FINAL STATE OF 2026-08-02.** Three submissions, one scored and best-ever; the error fully
decomposed; twelve axes closed on pre-registered bars.

| submission | what | cross-fit CV | public LB |
|---|---|---:|---:|
| `55184703` | blend + trust datum, 2.5 ft reference | 8.1326 | **7.581** |
| `55190292` | the same at FINE reference resolution | **8.0435** | pending |
| `55192950` | geom pick + the same correction, frozen constants | 8.1757 | pending |
| `55157342` SELECTED | untouched | 8.2251 | 7.726 |
| `54791000` SELECTED | untouched | 8.2826 | 7.899 |

**The error is now decomposed completely, which it never was before.**

| component of 8.2251 | RMS | status |
|---|---:|---|
| datum | 6.2943 | SOLVED to correlation 0.27, banked, shipped, -0.18 ft |
| shape, linear half | 3.7795 | dip residual; every observable reaches only 0.196 |
| shape, non-linear half | 3.7081 | surface variation with an ~800 ft MD correlation length |

**What the remaining error IS, measured not assumed.** The structure function shows our
prediction does not track the surface below about 400 ft at ALL: the error's variation at short
lags equals the truth's own (0.411 vs 0.388 at 25 ft, 3.615 vs 3.781 at 400 ft), because
savgol(17,3) and the degree-4 U-projection remove those scales by construction. The truth
saturates by 800 ft; our error climbs to 3,200 ft, which is the drift. So the residual is real
geology below the well scale, invisible to any method that summarises a window or a well by one
number, and only the foot-sampled GR log has the resolution to reach it.

**Three lessons that should govern future work here**, each measured:
1. The emission is a sound DISCRIMINATOR over a low-dimensional family and an unsound OBJECTIVE
   for free path search. One parameter, truth wins 76 percent; two, no signal; 65^25 paths, truth
   loses 153 of 153.
2. Freedom that can MOVE the implied TVT is harmful; freedom orthogonal to TVT is necessary.
   Per-node gradients break decoders; the per-window affine GAIN refit is load-bearing.
3. CONTRAST, not oracle R^2 at the truth, is the figure of merit for a reference log. Following
   that lesson found a 2.5 ft boxcar smoothing across a 1 to 2 ft correlation length, whose
   removal doubled the gain.

**Two self-corrections recorded rather than buried.** Part 6 called the datum "spatially white";
per-bin it carries about 31 percent structure within 2,500 ft, and both that bar and the dip's
were mis-specified by averaging bins beyond the variogram's range. The first datum certificate
was VOID, its row set depending on the shift, which showed as real and rolled agreeing to 16
digits.

**Two results that passed their bars and were NOT shipped**, because passing a bar is not the
same as mattering: confidence-weighted shrinkage at -0.0345 against a 0.05 floor, and the
curvature field at -0.0035 with bars that asked for direction and forgot magnitude.

**SCORED: `55184703` returns public 7.581, the best this project has produced.**

| submission | what | cross-fit CV | public LB |
|---|---|---:|---:|
| `54791000` SELECTED | geom arm, the hedge | 8.2826 | 7.899 |
| `55157342` SELECTED | seven base portable Ridge | 8.2251 | 7.726 |
| **`55184703`** | `55157342` + trust-gated datum, 2.5 ft reference | **8.1326** | **7.581** |
| `55190292` | the same at fine reference resolution | 8.0435 | pending |

**WITHDRAWN 2026-08-03, see Part 26. This was called the first CV-to-LB confirmation; it is a single point 0.90 sd from zero on a board whose resolving power is sd 0.1613 ft. The paragraph below is kept for provenance and its conclusion no longer stands.**
`AGENTS.md` records the CV/LB correlation as NEGATIVE across five scored submissions, and names
the single controlled experiment that existed as evidence against trusting CV: `rogii pf multidiv
proj realmlp v1` was Pick-2's pipeline plus exactly one base, CV 8.9490 -> 8.8144 (-0.135) and LB
7.666 -> 7.862 (**+0.196**), the wrong way. `55184703` is the same experimental form, one
pipeline plus one component, and it moved CV -0.0925 and LB **-0.145**, the right way and by MORE
than the CV predicted.

Do not over-read one point. What it does support is a distinction worth carrying: that earlier
reversal added a BASE to a stack, which changes a fitted blend, while this adds a physically
motivated post-processing correction that reads the test well's own GR. The second kind appears
to transfer.

**Consequence for selection, which is the owner's call.** `AGENTS.md` states the bar: "A new
submission displaces a selected pick only if it beats cross fit CV 8.2251 and has returned a
normal score." `55184703` satisfies both, at CV 8.1326 and a returned 7.581, and it beats BOTH
selected picks on the board. Selection locks 2026-08-04.

**SECOND SUBMISSION 2026-08-02: `55190292`, kernel `wguesdon/rogii-trustdatum-fine-v1`.**
Owner approved. Same pipeline and same single inserted block as `55184703`, differing only in the
reference resolution. `55184703` shipped a 2.5 ft boxcar; this ships none.

**The finding.** Part 8 established that CONTRAST, not oracle R^2 at the truth, is the figure of
merit for a reference log. Resolution controls contrast, and this estimator's resolution had
never been examined: `SMOOTH = 5` at a 0.5 ft grid was inherited from the probe the module grew
out of, while the recorded GR correlation length is 1 to 2 ft, so the reference was smoothed
across the entire informative scale. Over 773 wells the ROLLED arm is flat at 0.108 to 0.123 at
every resolution while the real top-trust hit rate at L=400 runs 0.379 at 0.5 ft, 0.337 at the
shipped 2.5 ft and 0.229 at 4.5 ft. Fully cross-fitted this moves correlation with the true datum
0.2000 to **0.2718** and CV 8.1326 to **8.0435**, five folds of five improving, against rolled
+0.0415, across-well shuffle +0.0208 and global scalar +0.0158.

On the 3 released wells, scored against real labels with the correction removed by subtracting
its own shifts, the fine kernel gives **2.7728 -> 2.4909, -0.2820**, against the coarse kernel's
-0.1002 on identical rows. The ratio tracks the CV ratio.

`ISOLATION_FT` was swept over 1 to 8 ft on the same banked profiles and is FLAT, correlation
0.269 to 0.284, well inside the bootstrap interval. It is not load-bearing and was deliberately
NOT retuned, since picking its best value on a flat surface is pure selection optimism.

**Total from the deployed protocol: 8.2251 -> 8.0435, -0.1816, fully cross-fitted.**

**Two more axes closed on their own pre-registered bars, both after the submission.**

*Peak height fails, and the emission is exhausted for SHAPE.* The registered experiment asked
the emission for the one thing it had not been asked for, since the dose-response curve says
peak HEIGHT tracks shape quality by a factor of seven even where peak LOCATION is uninformative.
Over the top-2 singular directions of the per-well base bank disagreement, with the datum
profiled out, shape RMS goes 5.3484 to 5.3230 while the ROLLED control reaches 5.3465. Real
minus rolled **-0.0234**, less than half the floor. Cross-fitted shrinkage lands at 0.10 to 0.15,
the estimator declaring its own coefficients noise, while the oracle in the same k=2 family is
3.0064. A 40-well smoke of this arm read -0.1238, five times the 773-well answer, and a 60-well
rerun already read -0.0529: a second documented instance of subset over-promise alongside
`cnn_1d_direct`. The emission has now refused per-well SHAPE four distinct ways and gives the
DATUM only.

*Field-scale pooling is dead.* Opened because `TVT + Z = surface_f + C_w` makes the target a
field and the trust-gated windows are roughly 20,000 localising constraints at surveyed
positions never pooled across wells. Its first step was a falsifier and it killed the axis in one
pass: the TRUE per-well datum has a structured fraction of **0.183** against a 0.20 bar, and no
quantity's short-range semivariogram sits below its position-shuffled interval, so even that is
not distinguishable from chance. The datum is a white per-well innovation in space. This explains
the earlier surface-transfer finding that an honest bias field is worse than its own shuffled
control.

**The live axis is `globally_calibrated_decoder`,** opened from this session's certificate rather
than a fresh source. The matcher was closed by a Viterbi certificate where the truth loses 90 of
90, read then as the observable being information limited. Today's certificate refutes that
reading, so 90-of-90 is a statement about the DECODER. The one concrete difference: the
certificate fits ONE global affine calibration to the whole eval region, while every DP decoder
here profiles slope and gain LOCALLY per window, which lets a decoder explain GR at a wrong TVT
by moving the calibration instead of the path. That is exactly the overfitting signature a
truth-loses-90-of-90 result has. First step reruns the existing exact Viterbi with the
calibration held global and re-runs the certificate. Price it on the CERTIFICATE, never on
standalone RMSE.

**Open actions.**
* Port the datum estimator to inference and verify it against the banked OOF exactly, following
  `scripts/verify_cnn1d_inference_2026_07_28.py`. Then ask the owner about a submission.
* `emission_shape_requirement` is the one open axis. Its first bounded step is running:
  `scripts/probe_shape_bank_ceiling_2026_08_02.py` asks whether the 163 banked base OOFs can
  express the per-well shape at all, with a pass bar of 2.5 ft oracle residual.
* AWS trial B `rogii-cnn-sdf-fall-s42-2026-08-01-16-26-31-932` is still uncollected and its
  ceiling is the measured -0.027 ft; the SDF cannot ship. Trial A `...16-26-07-490` is INVALID.

---

# SESSION 2026-07-28 — 8 days to deadline

## Resume update, 2026-08-01 evening, Claude Code continuation

Preflight green: 8.2826, 8.7300, 8.9641 all reproduce. Best honest CV remains **8.2251**.

**Public 6.213 teardown, closed.** The highest scoring public kernel title was pulled on 07-31 and
had never been analysed. Its distinctive mechanism is a train/test well id lookup, not a forward
model: `_gold_contact_candidate` opens `train/{wid}__horizontal_well.csv` for a *test* well id and
reads its `TVT`. All 3 released test wells are train wells, MD identical, labels on 100% of the
14,151 hidden rows, and contact reconstruction against them scores **0.0053 ft**. Since the public
frontier is 5.4 to 6.2 rather than 0.005, the released `test/` directory is a development sample and
the scored set is different wells. Already known here via `scripts/public_lb_oracle.py` but missing
from the read path; now in `AGENTS.md`. Full writeup `notes/public_6213_teardown_2026_08_01.md`,
audit `reports/probe_test_train_overlap_2026_08_01.json`. Contributes nothing to CV.

**Open question for the owner, needs a submission to settle.** Whether the *scored* test set also
contains train-overlapping well ids is unknown. If a fraction `f` of scored rows were recoverable,
pooled RMSE scales by `sqrt(1 - f)`; 8.0 to 6.213 needs `f = 0.40`. Consistent with the public
frontier, not evidence for it. Not tested; costs one submission and owner approval.

**E2 null calibration, Stage 1: clean negative, axis closed.** The report banked under this name was
a 1 well 2 decoy smoke at 6 s, never the queue's instrument. The real run is 90 fold zero wells at 16
decoys, 1590 s. `truth_optimal` is **0.000 on all 22 arms** against a pre registered bar of 0.20. No
paired well clustered bootstrap CI on `dp_node_err` lies below zero, and `frac_wells_better_than_raw`
never exceeds 0.50. Two pre registered controls independently refute the mechanism: kappa
monotonicity fails and the roll and lag decoy constructions disagree in sign. The null bias is real
at a median 2.29 nats per node over 1.17 M cells, so the wrong attractors are not the shape family
maximum. Do not retune anything here.
`notes/e2_null_calibration_result_2026_08_01.md`.

**Public sources exhausted, axis closed.** Physics 7.872 and blacklions were read in full and every
surfaced mechanism adversarially refuted, four for four with named prior art. Both score through the
same well id lookup as the 6.213. Neither carries a verifiable CV: physics 7.872 has `RUN_CV_REPORT`
off and zero stored outputs across 97 cells, and blacklions reaches 6.390 by adding a flat +0.522 ft
to one sample well then fitting a quadratic to returned board scores.
`notes/public_source_teardown_2026_08_01.md`.

**The public leaderboard is NOT the 3 sample wells.** Settled offline, no submission spent. Those 3
wells are train wells, so the deployed protocol can be scored on exactly their 14,151 eval rows.
Pick-1 scores **3.5932** there against a board return of **7.899**; the Pick-2 backbone scores
**4.1885** against **8.065**. Pooled numbers reproduce their records to 4 decimals, so no pipeline
difference explains a 4.31 ft gap, and cross fitting only strengthens it. There is no offline board
oracle, and the overlap lookup is inert on the graded set. The 3 sample wells are **4.58 ft easier**
than pooled with a spread of only 0.50 ft across three arms, which partly explains the CV to LB
offset. It does not license trusting the board.
`notes/public_lb_identity_2026_08_01.md`, `scripts/probe_public_lb_identity_2026_08_01.py`.

**Matching scale is closed, and it localises the deficit to the observable.** The E2 emission's
baseline was never swept; `WINDOWS = (400.0, 800.0)` is hardcoded and the inlier rate was rising
between them, so a long baseline was a plausible datum estimator worth 5.31 (datum) or 3.74 (datum
plus slope). It is not. Over 155 fold zero wells, lift over the rolled typewell null peaks at 800 ft
(+0.088) and decays monotonically to **exactly 0.000** at full eval span, with the standardised value
at the truth decaying 0.55 to 0.08. Median rows per window *rises* 287 to 877 across that sweep, so
this is model mismatch in the linear (slope, gain) family, not noise. A full span datum estimate is
53.60 ft RMSE against a null of 56.93, at chance on the 2 ft criterion.
`notes/emission_baseline_length_2026_08_01.md`.

**Four routes to the matcher failure are now closed with controls:** search (exact Viterbi, truth
loses 90 of 90, a certificate not a plateau), aliasing (Stage 0), candidate cell selection bias
(Stage 1), matching scale (above). Further decoder work on this emission is not justified. The one
untried direction is a different observable: the typewell `Geology` column is a categorical formation
label against TVT that the raw NCC never reads. That is now the lead axis.

**Full recipe retrain of the strongest base: a wash.** The declared training handicap in
`train_new_bases_2026_07_26.py` was never lifted for the E2 feature family, and the rate coupled
GBDT holds Ridge coefficient 0.685 of the seven, so it was the right slot. Row step 4 to 1 and 1200
to 2500 trees, five folds, 5361 s. Standalone 8.5161 to 8.5037, but the same slot swap moves cross
fit CV 8.2251 to **8.2260**, delta +0.0010, worse and inside noise. Halves disagree in sign
(-0.0285 / +0.0245) and only 2 of 5 folds improve. **Do not swap it in.**
`notes/ratecoupled_full_recipe_2026_08_01.md`.

The pilot read -0.1106 on fold 0, nine times the pooled gain that materialised; fold 0 is the
easiest of five. Third confirmation that standalone steps under ~0.05 ft are noise in this slot
(boundary 0.0126 better standalone, 0.0076 worse in stack; futureu 0.0175 for 0.0004).

**AWS trials resolved.** Trial A `...16-26-07-490` hit MaxRuntimeExceeded at 14,634 s and is
INVALID, never collect or gate it. Trial B `...16-26-31-932` completed at 10,155 s and is
uncollected. Whether it can ship depends on the unresolved SDF deployability contradiction between
`AGENTS.md` NOT QUEUED and `src/cnn_sdf/kernel_infer.py`.

**Data loss and recovery, 2026-08-01.** The gitignored `experiments.db` was deleted by a subagent. A
2026-07-17 snapshot was recovered from the UM890 checkout by rsync and integrity checked, then the
lost window was backfilled from the Kaggle submissions API and `LEDGER.tsv` via
`scripts/rebuild_experiments_db_2026_08_01.py`. kaggle_runs 17 to 25, experiments 30 to 63, rerun is
a no-op. Every reconstructed row is tagged in its notes. Exact hyperparameters, runtimes and feature
group lists for models trained 07-17 to 08-01 existed only in the DB and are permanently gone; all
scores, verdicts and decisions survive in `LEDGER.tsv` and the docs.

**Formation sequence observable is CLOSED, priced dead in one probe.** A PERFECT membership oracle
scores 25.3311 pooled, worse than carry forward 15.9107, because the median well's eval region sits
inside exactly ONE formation so membership is nearly constant along the lateral. The same probe
pinned the exact identity over all 773 wells: `TVT + Z - surface_f = C_w` per well, all six
surfaces, median spread 0.05 ft. So `U = TVT + Z` IS the structural surface plus a per well datum.
`reports/probe_formation_observable_oracle_2026_08_01.json`.

**Structural surface transfer is CLOSED.** Interpolating the surface (not TVT) from offset wells and
refitting the datum on the prefix reaches 24.68 ft and gates WASH, but at err-corr 0.327/0.297
against 0.87 to 0.95 for every other base, the most decorrelated family here. Half the error is the
datum: an oracle constant gives 12.3364, beating carry forward. That gap is `bias_prefix -
bias_eval`, and correcting it with an honest leave-one-well-out bias field is WORSE (29.9975) and
worse than its own shuffled control (25.8947), so the bias is unpredictable innovation.
`notes/surface_transfer_single_model_2026_08_01.md`.

**Next axis.** `independent_literature_forward_model` is the single open axis and `make
breadth-status` passes at 1 of minimum 1. Kaggle sources are spent, so rule 3 requires an
independent scientific or engineering source; price any candidate's standalone oracle against carry
forward 15.9107 before a 773 well build.

**Where the deficit now provably is.** Offset wells resolve the structural surface to ~25 ft against
90 to 170 ft of variation along a well, and the GR emission is information limited at ~10% inlier at
its best scale with four routes excluded by controls. Two independent lines therefore agree: a sub 6
single model cannot be reading the surface from neighbours, and cannot be reading it by NCC
correlation matching. It must extract the surface from the query well's own log by a mechanism not
yet identified.

**Open actions for the next session.**
* Collect AWS trial B `rogii-cnn-sdf-fall-s42-2026-08-01-16-26-31-932`, correspondence likelihood,
  Completed at 10,155 s. Pull `model.tar.gz`, never `output.tar.gz`. Trial A
  `...16-26-07-490` is INVALID (MaxRuntimeExceeded at 14,634 s) and the 16-25-14-819 job is invalid;
  never collect or gate either.
* Resolve the SDF deployability contradiction FIRST: `AGENTS.md` NOT QUEUED says the test time image
  cannot be reproduced, `src/cnn_sdf/kernel_infer.py` says its repair is exact at 0.000e+00. It
  decides whether trial B can ship or is CV only. The SDF's measured stack contribution was only
  -0.027 ft, so the ceiling is low either way.
* `predictions/backup_oof_lgbmede2ratecoupledfeature_2026_08_01.npy` is a byte identical copy of the
  deployed base, kept until the deadline. Do not delete it before 2026-08-05.

## Resume update, 2026-07-31

## Resume update, 2026-08-01

### Claude Code handoff, 2026-08-01

The project now has `make claude-handoff`. It writes a dated continuation brief under
`scratchpad/` with the current CV, active forward axes, AWS job notes, recent ledger rows, and
recent commits. Claude Code should run `make preflight` and then this target before taking work.
The brief repeats the no early stopping rule, the prohibition on the broad ensemble search, and
the required ledger, queue, and session updates.

### Boundary emission result, 2026-08-01

The literature led boundary conditioned E2 emission passed its pre decode fold zero test. Across
155 held out wells, adding the fixed 0.25 boundary score improves true versus nearest alias
margin by 0.00506, 95 percent interval [0.00400, 0.00611]. It also beats the typewell rolled
boundary control by 0.00580 and the matched magnitude control by 0.00511. These controls leave
raw E2 NCC unchanged.

The full 773 well rate posterior rebuild is materially stronger. Its raw arm reproduces the
banked 19.4995 standalone RMSE exactly. Boundary emission gives **17.5165**. Boundary only
rolled and matched magnitude controls give 19.6159 and 19.4676. The normal base gate records
`untried19_ratehmm_boundary_real_sg5` as a KEEP: NOGEOM blend add minus 0.2578 against a rolled
base control of minus 0.1396, with 4 of 5 LEAK folds improving and 0.252 NOGEOM error
correlation. The fixed seven base portable stack falls from 8.2251 to 8.2087 on its cross fit,
but the two held out halves average only minus 0.0127. Do not port the direct residual column.
Use it as one feature in a median LightGBM retrain and apply the normal gate. Do not tune the
fixed 0.25 boundary weight on this OOF.

The boundary feature retrain is now complete. `lgbmede2ratecoupledboundaryfeature_2026_08_01`
has standalone 8.5035 and passes its normal base gate. It replaces, rather than joins,
`lgbmede2ratecoupledfeature_2026_08_01` in the portable stack because the models are near
duplicates. That replacement raises cross fit CV from 8.2251 to 8.2327. The boundary path is
therefore tried, not open. Do not tune the score or port either boundary model. The next bounded
test is candidate specific E2 null calibration.

An aggressive public claim was also screened: nearest training well TVT transfer in XY, with a
held out well's known prefix used only for a datum correction. It is legal under GroupKFold but
not useful in this data split. The 150 ft arm transfers 5.6 percent of rows and scores 33.4466
RMSE. This is not De DQ's successful method, or it requires a spatial relation this direct
transfer does not capture.

Candidate specific E2 null calibration passed its Stage 0 falsifier. On 90 fold zero wells, only
41.9 percent of 2,960 exact E2 DP node errors fall within 2 ft of a 13 ft alias lattice. The
wrong attractors are broad enough that a candidate specific null correction can address them.
Next is the 90 well decoy instrument. It must quantify candidate null score range and improve
the objective truth optimal fraction against decoy controls before a full 773 well decoder.

### Harness update, 2026-08-01

The active instructions now require campaign continuity while the deadline is open. Each clean
negative closes only the mechanism measured. The queue must retain one high upside forward model
test and one inexpensive verification or deployment action. After two material failures or two
hours, mine the Kaggle forum and notebooks, then seek an independent technical source. Use a
deputy or Claude Code to red team each new forward model proposal. Update the queue, ledger and
session record after each material result. The user set a sub 6 CV target, so neither a plateau
nor a run of washes permits stopping.

The template is present on the UM890 checkout at
`/home/will/Documents/Github/Kaggle/Playground_Series/_harness_template`. It adds a script
enforced breadth check and a visible completion override log. ROGII now carries a tailored
version in `harness/forward_axes.json`, `harness/forward_target.json`, and
`harness/breadth_gate.py`. It tracks independent forward model routes instead of generic model
counts because ROGII has already measured a recombination ceiling below the target.

The user confirms the recommended two web selections are made. Pick 1 remains
`rogii pf multidiv proj geom, Version 2`, public score 7.899. Pick 2 is
`rogii pf multidiv proj med realmlp cnn e2 ratehmm, Version 1`, submission
`55132115`, public score 8.065. The second pick is selected for its lower
cross fit CV, not its public score.

The rate coupled E2 likelihood passed the next useful test as a feature, not
as a standalone prediction. `lgbmede2ratecoupledfeature_2026_08_01` adds the
accepted E2 residual, the rate posterior residual, and the rate coupled
residual to the ten group median LightGBM. It has standalone RMSE 8.5161.
The normal base gate is a KEEP, with NOGEOM blend add minus 0.4088 against a
rolled null minus 0.0481 and five of five LEAK folds improving.

Replacing the earlier E2 rate posterior LightGBM in the portable seven base
stack gives cross fit CV **8.2251** and full fit CV **8.1833**. This is a
0.1560 ft cross fit gain over the six base submitted rate posterior stack,
and a 0.1171 ft gain over the earlier seven base GBDT. Do not treat it as a
submission yet. Its five fold LightGBM artifact and its legal rate coupled
inference source must reproduce the OOF before a Kaggle port can be made.

The direct matcher itself remains a failed standalone base. The coupled HMM
has pooled RMSE 17.4909, better than the 19.4995 posterior but worse than
carry forward at 15.9107. Its reduced control is 24.8337 and rolled control is
49.4023. The observed rate likelihood is real, but it currently helps only
through the tabular model.

The next direct state experiment is a clean kill. It generates each local GR
template directly from the nine rate states at unit survey gain. This removes
the unequal shape bin maximum and makes the emission agree exactly with the
transition kinematics. Full 773 well RMSE is 33.9114. The reduced control is
36.8873 and the rolled control is 46.7203. It misses the required 17.4909
mechanism bar, so no constants from this decoder may be tuned.

Submission `55157342`, titled `rogii ratecoupled gbdt v1`, scored **7.726** on
2026-08-01. Its cross fit CV is 8.2251 versus 8.3811 for selected Pick 2
`55132115`, and it therefore satisfies the documented replacement rule.
**DONE 2026-08-01: the owner made this change. Selection is now settled.**

The cross fitted future `U = TVT + Z` model is a wash. It forecast 250, 500,
and 1000 ft future U values from strict GroupKFold models, then supplied them
to the median LightGBM. Its portable stack cross fit CV is 8.2247, just 0.0004
ft below the rate coupled GBDT at 8.2251. That difference is below the
protocol resolution and the models are substitutes. No port or second Ridge
column is justified.

`55132115` is submitted and pending. It is the first kernel with the E2 rate
posterior HMM readout. The uploaded kernel completed, emitted 14,151 finite test
rows, and uses the verified six base portable Ridge.

| Candidate | cross fit CV | full fit CV | public score |
|---|---:|---:|---|
| `55117845`, selected E2 stack | 8.5105 | 8.4752 | 7.975 |
| `55132115`, E2 rate posterior | **8.3811** | **8.3518** | pending |

The rate posterior base is a KEEP. Its NOGEOM blend add is -0.2146 versus a
rolled control of -0.1187. It improves every LEAK fold. Its legal kernel
inference reproduces its banked OOF to 8.5e-09 ft mean absolute TVT difference.

Keep selected Pick 1 `54791000` unchanged. If `55132115` receives a normal
score, it replaces selected Pick 2 `55117845` because its cross fit CV is 0.1294
ft lower. Do not change selection before the score returns.

The next bounded test is `lgbmede2ratefeature_2026_07_31`, a median LightGBM
with the accepted E2 and rate posterior residuals as its only two new features.
It ran across the full five fold OOF and had to pass the normal gate before any
kernel work.

### 2026-07-31 E2 conditioned LightGBM result

`lgbmede2ratefeature_2026_07_31` is a KEEP. It has standalone RMSE 8.7453,
NOGEOM blend add -0.2582 versus rolled control -0.0374, and improves all five
LEAK folds. The seven base stack gives cross fit CV **8.3422** and full fit CV
**8.3019**, an additional 0.0389 ft reduction from the rate posterior stack.
The base needs a legal test inference port before it can become a submission.

Deadline **2026-08-05**. New submissions in by **2026-08-03** to return a score with room for a
retry. **Selection locks 2026-08-04.**

---

## RESOLVED 2026-08-01 — selection is settled, no blocking user action remains

The owner selected **`55157342`** (`rogii ratecoupled gbdt v1, Version 1`) and **`54791000`**
(`rogii pf multidiv proj geom, Version 2`) on the Kaggle web UI. `55132115` was deselected. Both
selected submissions have returned a normal public score and have survived a rerun, so the silent
default-to-best-public-score failure mode no longer applies.

The pair was measured, not assumed: `reports/probe_pick_pair_decorrelation_2026_08_01.json`.
`55157342` and `55132115` are near duplicates at error correlation 0.9866 and their average is
worse than the better one alone, so selecting both would have spent two picks on one bet.
`55157342` with `54791000` is the most decorrelated pair at 0.9305 and the only pair whose half
average beats both members, by 0.1159.

Any further change now needs to beat cross fit CV **8.2251** and return a score. Never select
`54853374` or `54851870` (the 2026-07-20 pair at 26.939). Selection locks 2026-08-04.

---

## Submitted today, both PENDING a score

| sub | stack | deployed CV | cross-fit CV |
|---|---|---|---|
| `55047409` | MED2+realmlp | 8.7098 | 8.7266 |
| **`55048494`** | **MED2+realmlp+cnn_1d_v1_avg3** | **8.5822** | **8.6104** |

Against Pick-2's 8.9490 deployed. 2 of 5 daily submissions used.

`55048494` is the first submission ever to carry a whole-well CNN. Its kernel log confirms
`[cnn1d] root ...rogii-cnn1d-models; 15 fold models`, `[cnn1d] residual mean +3.5216 std 5.4379`,
and 14151 rows. That non-zero std is the production proof of the bug fix below; a collapsed CNN
emits std 0.0000.

---

## The finding that should reorder the remaining days

`reports/competitor_intel_2026_07_28.md`, from 146 mined discussions and the literature.

**A competitor reports a single pure PHYSICS model at CV 6.85 / LB 6.577**, and confirmed on
request that his CV is *"per-point RMSE on the post PS part, not the per-well avg"* — our exact
pooled metric. Our best five-base stack is **8.6082**. Our own PF is **10.3611**.

One good forward model beats our entire ensemble by 1.76 ft, and our forward model is 3.5 ft
behind his. The deficit is not the blend, the base count, or the CNN. `QUEUE.md` is reordered
around this.

Supporting calibration: k256.dev puts tabular's ceiling near "LB 6.0 (CV 7.0)" and later reports
CV 5.x with LB consistent; shu01 is at LB 4.859; public notebooks saturate LB 7.15-7.3, which is
**ahead of our 7.666**. The public kernel advertising "LB TOP 3" is by its own header a sub-9
solution.

**Metric ambiguity is real but does not rescue us.** Two competitors quote one model as
"mean-per-well 5.391, pooled 7.941" and "mean-per-well 5.22, pooled ~7", so unattributed "CV 5"
claims are probably per-well. Our best stack is pooled 8.6082 / mean-per-well 6.4784 / median
4.9358 — 0.67 behind pooled, 1.09 mean-per-well, **1.60 on the median well**. We are relatively
worse on the TYPICAL well, and pooled RMSE hid that.

**Tucker Arrants (LB 5.444):** *"you can get your single model CV score below 5ft without using
any neighbor well data, so you can take GR matching quite far here."* Second independent strike
against the "information-closed" verdict, after our own 2026-07-27 emission repair.

**CLAUDE.md's oracles are correctly valued but loosely worded.** Reproduced a competitor's oracle
table to two decimals (constant 9.0354 vs his 9.04, line 6.6972 vs 6.70, quadratic 5.3423 vs
5.34). Its "perfect per-well constant datum = 5.31" is the BLEND plus a perfect per-well constant
(5.4646 here), not a constant alone (9.04). "3.74" is blend plus perfect line (3.7873).

---

## The bug that would have shipped a dead base

`dataset.py` built `eval_mask` only when the TVT label was present, but `Seq1DNet` consumes
`eval_mask` as an **INPUT** — it gates the baseline cumsum, the datum pooling and the head. On a
hidden test well there is no TVT, so the mask was all zeros and the model emitted pure
carry-forward in both drift modes. A kernel port written from the training entry point would have
run, scored, and been silently worthless.

Caught by rebuilding fold 0 from the checkpoint and diffing against the banked OOF rather than
trusting the port: 8.34 ft error, with `resid` and `direct` bit-identical, the collapse signature.
Fixed; now reproduces all 155 fold-0 wells from the six legal test columns at 2.1e-3 ft mean.
`scripts/verify_cnn1d_inference_2026_07_28.py` is the regression test.

Second trap, documented in the kernel: `drift_mode` must be `resid`, which is not the launcher
default, and a `strict=True` load succeeds either way.

---

## Gated this session — LEDGER.tsv

| base | standalone | err-corr | blend-add | null | verdict |
|---|---|---|---|---|---|
| **cnn_1d_v1_avg3** | 12.9617 | 0.600 | **-0.1218** | -0.0610 | **KEEP** |
| cnn_1d_v1_avg5 | 12.9076 | 0.601 | -0.1202 | -0.0616 | KEEP |
| cnn_1d_v1_s7 | 13.1950 | 0.586 | -0.1140 | -0.0593 | KEEP |
| cnn_1d_wlocal1 | 13.3659 | 0.568 | -0.1112 | -0.0736 | MARGINAL |
| cnn_1d_v1_s2026 | 13.2675 | 0.586 | -0.1001 | -0.0525 | MARGINAL |
| cnn_1d_dim128 | 13.5137 | 0.578 | -0.0863 | -0.0546 | MARGINAL |
| cnn_1d_hfut384 | 13.6161 | 0.589 | -0.0858 | -0.0420 | MARGINAL |
| cnn_1d_v1_s123 | 13.4012 | 0.594 | -0.0788 | -0.0368 | MARGINAL |
| cnn_1d_direct | 14.4079 | 0.557 | -0.0612 | -0.0603 | **WASH** |

**Seed averaging is settled and closed.** avg3 -0.1218 beats avg5 -0.1202; returns go negative
past three seeds.

**The seed axis and the capacity axis are both spent.** dim128 MARGINAL matches the 2026-07-17
WarpMatch-XC kill: capacity is not the problem.

**My own fold-0 screen over-promised.** It ranked `cnn_1d_direct` at -0.1394 and I called it the
session's real lead; the 773-well gate says WASH at -0.0010. Treat the screen as a hint about
where to spend GPU, never as evidence. It also caught a genuine oracle (`untried19_e1_lam0.3`
fits its shape on TRUE TVT, now in `gate.py` INADMISSIBLE), so it earned its keep.

---

## In flight

- **MDN datum-head probes**, fold 0, 3 and 5 modes (`rogii-cnn-1d-f0-s42-2026-07-28-07-33-*`).
  The one research bet. Read `modes_used` and `datum_spread` in the log FIRST: at 1.00 the
  mixture collapsed and it is just the old head with more parameters.
- **geom-OFF SDF folds 1-4** (`rogii-cnn-sdf-f1-2-3-4-s42-...06-25-56-290`), ~2.5 h. err-corr
  0.386 on fold 0, the best decorrelation in the repo, but fold-0 REAL-NULL only -0.0316.
- Seed 77, optional; the seed axis is closed.

Collect everything with `bash scripts/collect_fleet_2026_07_28.sh`.
AWS spend ~$15 of the $30-50. g5.2xlarge quota raised 10 -> 20 (APPROVED); the actual constraint
was AWS regional capacity, not quota.

## Next

1. Gate the MDN probes and the SDF 5-fold.
2. **Q-3D tortuosity** — `QUEUE.md` item 1. The one concrete feature we do not have, reported as
   the toolkit's largest single ablation gain at -0.107 RMSE. One gated attempt, not a sweep.
3. Watch for scores on `55047409` and `55048494`.
4. **2026-08-04: select.** Current recommendation, pending those scores: `55048494` (best CV,
   no geom) and Pick-2 `54325084` (best LB 7.666, no geom, already survived a rerun).

## Resume update, 2026-07-28

Preflight passed at session resume. The current objective is CV improvement. Submission checks are deferred to the final day by user instruction.

The active work item is tracker drift accumulation. The first experiment will re anchor the particle filter within the lateral, while preserving the deployed reconstruction and comparing against the reconstructed deployed filter.

### Correction after source and measurement audit

Do **not** run the proposed global re-anchor. The maintained probe has already measured it. A
2% whole-typewell proposal every 800 eval rows produced 80.6084 pooled PF RMSE, against
11.2741 for the rebuilt BASE at the same aggregation. The posterior mean makes it structurally
unsafe because a low-weight alias still moves the reported mean.

The only tracker repair that has cleared the complete PF branch gate is `ROBUST200`, the
200-row Theil-Sen anchor-rate estimate. It improved the rebuilt control by 0.0527 on LEAK,
0.0663 on NOGEOM, and 0.0681 on PICK2. The remaining active technical lead is a joint
emission and rate-prior calibration of `run_pf_z`, not another global-proposal re-anchor.

`reports/sweep_gr_sigma_pilot_ABORT_2026_07_28.txt` records why the existing likelihood sweep
is inadmissible. It ports the wrong PF state, fails its known winner, and has a seed noise floor
far above its intended effect. Repair the real `run_pf_z` instrument first, keep defaults exactly
bit-identical, ensemble enough seeds, and pair real typewells with the stable rolled-typewell
control. The measured GR residual autocorrelation implies that changing likelihood temperature
alone is not meaningful. The likely mechanism is a joint likelihood deflation and a tighter
rate prior.

### In flight at handoff

The existing field safe `run_pf_z` ensemble was built as `predictions/oof_pfz_v1.npy` with eight
seeds and 14 workers, then gated through the standard base path. It is a clean WASH: standalone
15.7645, NOGEOM blend add `+0.0088` against its rolled control `+0.0034`, and only two of five
LEAK folds improved. `LEDGER.tsv` has the authoritative row. Do not revisit this exact ensemble
as a base.

The next tracker experiment remains the properly instrumented joint PFZ emission and rate prior
test described above. It must modify the real `run_pf_z` path, reproduce defaults exactly, use a
many seed ensemble, and carry a rolled typewell control. A one knob temperature sweep and the
current `sweep_gr_sigma_nu_2026_07_28.py` port are explicitly inadmissible.

The 773 well E2 dynamic-program matcher is now in the standard ledger as
`untried19_lam0.01`: MARGINAL, standalone 26.6275, NOGEOM blend add -0.0439, rolled null
-0.0281, four of five LEAK folds improved. Its emission remains informative, but the deployed
aggregation is not yet a reliable base. The canonical matcher status in `AGENTS.md` and
`QUEUE.md` was corrected.

Signed azimuth is not a live job. It has already been audited: the target correlation is 0.004 to
0.005, and the deployed feature store already contains continuous `azi_sin` and `azi_cos`.
`QUEUE.md` now records this closure so a third party claim does not cause a duplicate retrain.

## Resume update, 2026-07-29

Preflight passed on 2026-07-29. The target remains an honest pooled CV improvement before the
2026-08-03 submission cutoff. Work now starts with fresh Kaggle discussion and public-notebook
mining, plus primary scientific literature on log alignment and horizontal-well trajectory
correction. Any candidate will use the deployed gate and its required control.

## 2026-07-29 active measurement

Fresh Kaggle CLI mining completed in `discussion/kaggle_cli_2026_07_29/` and
`notes/kaggle_cli_2026_07_29/`. Current submissions have returned public scores:
`55048494` is 7.989 and `55047409` is 7.956. Do not infer a CV ranking from either.

The public HMM notebook is not economical. Its exact forward backward state has run for more
than three minutes on fewer than two wells, so it cannot produce an honest full 773 well base in
time. Its useful part is an affine horizontal to typewell GR calibration fitted only on the visible
prefix. The calibrated PFZ base `oof_pfz_affine_v1_2026_07_29.npy` is building across all wells
with 14 workers. Its default branch was verified bit identical to `oof_pfz_v1.npy` on 14,151
smoke rows before the calibration branch ran. Gate it as a base after the build finishes.

### Completed 2026-07-29 measurement

`pfz_affine_v1_2026_07_29` is a WASH. The full 773 well base has standalone
15.8284, NOGEOM blend add `+0.0051`, and rolled control `-0.0003`, with one of
five LEAK folds improved. `LEDGER.tsv` is authoritative. The affine GR
calibration does not repair PFZ drift and must not be re-run as a base.

### Active 2026-07-29 measurement

A single fixed joint PFZ branch is building: GR sigma is multiplied by 8.22,
from the measured 67.60 row residual autocorrelation time, and the velocity
prior multiplier changes from 2.0 to 1.0. It uses the real `run_pf_z` path,
eight seeds, and the standard base gate will create the rolled control. Defaults
were reproduced bit for bit again after adding the two knobs. Do not turn this
into a grid or choose values from a fold.

### Completed 2026-07-29 joint PFZ measurement

`pfz_joint_deflated_tight_v1_2026_07_29` is a WASH. Its standalone RMSE is
40.6405. NOGEOM blend add is `+0.0058`, versus `+0.0033` for the rolled
control. The combined likelihood deflation and tighter velocity prior is
closed. Do not revisit a GR temperature or rate prior knob without a new
emission model that changes the information aggregation.

### In flight 2026-07-29

The E2 matcher is rebuilding over all 773 wells with a fixed five sample
Savitzky Golay GR smoother on both horizontal and typewell logs. It changes the
emission only, preserves the same local shape grid and DP, and outputs every
predeclared lambda. Gate only `untried19_lam0.01_sg5`, the existing canonical
lambda, when the build finishes. Do not select a lambda on the new OOF.

### Completed 2026-07-30 E2 emission measurement

`untried19_lam0.01_sg5` is a KEEP. The five sample GR smoother raises
NOGEOM blend add from `-0.0439` to `-0.1032`; its rolled control is `-0.0443`.
It remains a poor standalone path at 26.6806, but it now clears the base control
with four of five LEAK folds improved. The next maintained action is the nested
ensemble to price it honestly beside the current stack.

### Completed 2026-07-30 E2 ensemble and portable stack measurement

Fresh Kaggle discussion and public notebook mining is stored under
`notes/kaggle_cli_2026_07_30/`. The advertised 6.391 public notebook does not demonstrate a
reproducible 6.391 CV result. It depends on external assets, disables its own CV report, and its
visible cross fit output is 8.5873.

The nested ensemble result has an honest held out mean gain of -0.2145. The E2 base is in the
intersection of both searches. `reports/ensemble_sg5_2026_07_30.log` contains the complete
selection guard.

The direct portable stack measurement is the immediate submission candidate. Adding E2 to the
existing submitted kernel bases gives cross fit **8.5105**, down 0.0999 from 8.6104. Its full fit
inference record is 8.4752. Exact full fit coefficients are in
`reports/e2_portable_stack_2026_07_30.json`. Do not submit until a test time E2 port has been
rebuilt on train wells and matched to `oof_untried19_lam0.01_sg5.npy`.

### 2026-07-31 submission and decision record

Kernel `wguesdon/rogii-pf-multidiv-proj-med-realmlp-cnn-e2` completed with E2 nonzero on every
one of the 14,151 public test rows. Its source rebuilt the 773 well banked absolute TVT track with
mean difference 8.4e-09 ft and maximum 7.8e-05 ft. Kaggle submission `55117845` returned public
LB 7.975. It has cross-fit CV 8.5105 and full-fit kernel CV 8.4752.

**Two pick recommendation for the owner to select in the Kaggle web UI on 2026-08-04:**
`54791000` and `55117845`. The new candidate replaces Pick-2 because it beats CV 8.9641 and has
returned a score. Public LB is not a model selection signal. Do not select any submission that has
not completed.

The owner confirmed both selections in the Kaggle web UI on 2026-07-31. Keep `54791000` and
`55117845` selected unless a newly scored candidate replaces the E2 model under the CV rule.

The physical 6.5 ft smoothing E2 variant is WASH. The E2 forward backward posterior decoder is
MARGINAL: standalone 21.8660, NOGEOM blend add -0.0254, rolled control -0.0161. The remaining
high upside experiment is a coupled local dip state decoder. It keeps the accepted E2 emission and
canonical lambda, but changes the zero mean transition prior to follow the local slope and survey
gain states. The predeclared instrument and decision bars are in
`scratchpad/claude_forward_model_plan_2026_07_30.txt` until the experiment script is committed.

### Completed 2026-07-31 persistent dip pilot

`untried19_dip_lam0.01_sg5` failed its predeclared B1 mechanism bar. Its full 773 well
standalone RMSE is 33.8827, above carry forward at 15.9107 and above the accepted E2 at 26.6806.
It is not a base and was not sent to `make gate`. This implementation held one shape globally for
each well, so it rejects that global shape decoder. It does not establish that every coupled local
dip state decoder is impossible. Do not spend more time on this exact decoder.

### Completed 2026-07-31 adaptive dip decoder

`untried19_adaptive_dip_lam0.01_sg5` also failed its predeclared B1 standalone bar. It used the
same accepted E2 local emissions, but made each candidate TVT cell follow the local slope and
survey gain selected at that cell. Full 773 well RMSE is 27.2320, against carry forward at
15.9107 and accepted E2 at 26.6806. It covered 3,783,582 of 3,783,989 rows and declined one well.
It is not a base and was not sent to `make gate`. The result closes nodewise independent local
shape transition means as well as the already rejected global shape mean. Final picks remain
`54791000` and `55117845`.

### Completed 2026-07-31 self reference E2 emission

The self reference experiment kept accepted sg5 E2 bit exact in pass one. It then used that
truth free path to bin calibrated lateral GR onto the typewell TVT axis and formed a fixed one
half reference blend. The full result is 27.4834 standalone RMSE, worse than accepted E2 at
26.6806, so it failed B1 and was not sent to `make gate`. Its deterministic rolled path control
scored 31.3891. The effect is alignment dependent, but it moves in the wrong direction. Do not
try mix weights or a second self reference pass. The run and the exact pass one reproduction are
in `reports/emit_untried19_selfref_2026_07_31.log`.

### Completed 2026-07-31 E2 feature GBDT

`lgbmede2feature_2026_07_31` adds the accepted E2 residual to the ten field safe deployed median
feature groups. Under the declared row step four, 1200 tree handicap it scores standalone 9.0484.
The canonical gate is MARGINAL: NOGEOM add -0.0421, rolled null -0.0119, margin -0.0302, three of
five folds improved, and error correlation 0.951. It does not earn a test port or submission.
The only remaining use is a full recipe retrain if compute remains after high upside matching work.

### Completed 2026-07-31 public rate state E2 decoder

`untried19_ratehmm_public_sg5` applies the public HMM's 41 state rate grid to E2's accepted sg5
emission. It runs in 450 seconds across all 773 wells but scores standalone 27.0097, above the
accepted E2 B1 bar of 26.6806. It is not a base and was not sent to `make gate`. Do not tune this
public rate model. Its advertised result also uses external learned artifacts absent from the repo.

### Completed 2026-07-31 rate posterior E2 decoder

Changing only the public rate model readout from Viterbi to its posterior mean produced
`untried19_ratehmm_posterior_sg5`, a KEEP. Standalone is 19.4995. NOGEOM add is -0.2146 against a
rolled null of -0.1187 and all five LEAK folds improve. The fixed six base portable stack scores
8.3811 cross fit and 8.3518 full fit, an improvement of 0.1294 against the current submitted E2
stack. `verify_ratehmm_kernel_inference_2026_07_31.py` reproduced banked absolute TVT with mean
8.48e-09 ft and maximum 7.81e-05 ft difference. The kernel is ready to push and submit. A
completed scored submission replaces `55117845`, not `54791000`.
