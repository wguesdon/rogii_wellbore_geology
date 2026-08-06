# QUEUE — what to do next

**2026-08-04. Deadline 2026-08-05 23:59 UTC. Read `SESSION_SUMMARY.md` first for live state.**

## 0. REAL BUT NOT ACTIONABLE BEFORE THE DEADLINE — for the next session, not this one

Recorded 2026-08-04 evening from a deep re-read of the whole discussion history and the non-fork
public notebooks. Each is a mechanism with a source; none can be made credible in the time left.
They are here so they are not rediscovered from scratch, and so nobody half-starts one.

* **Azimuth deprojection of transferred dip.** `connortynan/dz-dtvt-eda` and
  `connortynan/rogii-k16-spline-kernel-knn-adaptive-kappa` (LOO pooled 8.061), forum 711308. Models
  `dTVT = -dZ + D(x,y)*cos(az - theta0)` with a measured regional up-dip azimuth
  **theta0 = 118.4 deg**, and stores each segment's coefficient as `c_j / cos(az_j - theta0)`
  BEFORE transferring it, so two laterals crossing the same dipping bed on opposed headings do not
  cancel. mycarta's toolkit corroborates the geometry: the azimuth rose is bimodal at about 325 and
  140 degrees, 180 apart, which is exactly the configuration in which un-deprojected dip cancels.
  He reports the donor field explaining about 42 percent of drift variance against our per-well dip
  residual reading 0.196 from every observable we own.
  **Why not tonight:** the deprojection only changes anything for CROSS-WELL transfer, since
  azimuth is near-constant along a lateral so deprojecting a well's own prefix dip is a per-well
  constant and cannot move the prefix-to-tail correlation. That makes it irreducibly neighbour
  pooling, whose deployed instance `geom_k16` has a CV edge that is 98 percent within-field leakage
  and REVERSES by 0.233 ft on the board. Making it credible needs a rebuild, a field-blocked arm and
  a board test, and there is one board test's worth of time left. **Test theta0 = 118.4 and the
  bimodal rose first next session.**

* **`seg_b_well`, a five-datum decomposition, is a board-wide idiom rather than one notebook's
  trick.** `b_full / b_early / b_mid / b_late / b_wls` over `bv = TVT + Z - top` appears in EIGHT
  non-chain public notebooks (`davidburmeister`, `ezequiassousa/lithowarp-ultra`, `hongweiluan`
  arch001/idea002/idea004, `bioconvolutionyt/rogii-sca-u2net`, `wangzikai666/hc-dtw-knn-4`,
  `qujiahui/9-608`).
  **Why it is probably already ours, and the bound that says so:** it is "let the per-well constant
  drift across drilling phases", which is the RAMP in a phase-wise form, and we deploy the
  two-parameter version at -0.13 ft. The whole family is bounded: the entire ramp axis spans
  **0.0453 ft** from the constant-shift arm at 7.23628 to the in-sample optimum at 7.19099, the
  frozen pair already holds 0.0325 of it, and adding a QUADRATIC third parameter measured
  **+0.0053 ft cross-fitted** because `x` and `x^2` correlate 0.942 over the span wells cover. A
  five-parameter member of a family whose total headroom is under 0.05 ft and whose three-parameter
  member already loses is not a lever. **It escapes that bound only if the segment datums are
  estimated on the KNOWN PREFIX rather than being a reshaping of one estimate — if a future session
  reads it that way, that is a different mechanism and worth one run.**

* **GRU capacity, `--dim 288`.** Discussion 732455 msg 3508230, 2026-08-04: the rank-2 competitor
  says his recent gains come from scaling his model up, on the same whole-well-sequence family as
  ours. **REJECTED, not deferred:** width was swept AT THE COMPOUND RECIPE and lost.
  `psh_psr4_e160_dim160` reads -0.3571 against `psr4@160`'s -0.3755, and the plain-GRU dim-160 arm
  read -0.0229 against s42's -0.0338. The argument offered for reopening — that both width arms
  predate prediction-start resampling — is factually wrong, since `psh_psr4_e160_dim160` IS a psr4
  e160 arm. Reviving this is changing a constant on a swept axis.

## 1. STATE AS OF 2026-08-05 10:40 UTC — selection final, nothing left to submit

**Selection is LOCKED and correct: `55244616` (public 6.818, CV 7.1593) and `55244617` (6.618,
7.2113).** Both returned public scores, both Version 1, and the shared dataset slug
`rogii-seqalt-gru-psr4avg9` has never been re-versioned. **Do not push either kernel again and do
not re-version that slug** — re-versioning a slug a selected submission reads is what made
`55216428` unselectable.

~~**THE SUBMISSION CUTOFF IS ~15:30 UTC, NOT 23:59.** Observed queue latency ran 5h20m to 7h30m over
three submissions on 2026-08-04~~ **RETRACTED 2026-08-05 22:18. THE LATENCY FIGURE WAS WRONG BY
~35x.** Submission `55281768` was entered at 22:08 UTC on the final evening, the busiest hour of the
competition, and **returned a public score at 22:18. Ten minutes, end to end.**

The reconciliation is that latency is dominated by **the notebook's own runtime**, not by a shared
queue. Ours run 255-405 s. Topic 733099 msg 3509012 says "full 200 wells take only 4-5 mins, and
full submission kaggle too" while msg 3509011 reports 8.5 hours; both are true, of different
notebooks. The 5h20m-7h30m band was almost certainly three slow kernels, not a queue.

**What the wrong constant cost.** `bag4_add` (CV 7.135123, E[min] gain -0.0279 against the selected
pair) was ruled out on 2026-08-05 morning because its ~5 hour retrain "costs the whole remaining
window". Under the true latency it fitted with about seven hours to spare. It was still under the
0.05 ft floor and might well have been declined on merit, but the decision was never made because a
wrong number said the window was shut. **Measure submission latency on day one next time.**

Chris Deotte's rule in discussion 732947 stands unchanged and is the part that was always right:
only a submission which completes and has a public LB score before the deadline is selectable.

**Five submissions are unused and that is the deliberate, measured decision.** The final adversarial
sweep priced the entire remaining prize at **0.023 to 0.028 ft of expected private RMSE** against a
board whose difference noise is sd 0.1613 ft. Nothing clears the 0.05 ft shipping floor. A rushed
kernel with no retry margin is worse than an unspent slot.

**RE-CHECKED 2026-08-05 11:05 UTC against a fresh sweep of every 08-04 and 08-05 report, and the
answer holds on a CEILING rather than on a survey.** The selection-free arm `adv_bag4_and_gbdtball`
takes ALL 25 `gbdtdiv` arms, so no family is chosen, and reads **-0.0375 ramped, CI95 [-0.0909,
+0.0144], p_worse 0.081, 3 of 5 folds** on the deployed path. The -0.0536 headline was the
`lgb`-family pick made on full data and **the family choice is worth 0.016 of it**. That is the
maximum the whole GBDT-bag family can be worth with nothing withheld for portability, so **no
portable subset of it can exceed it**, and the shippable half stays at -0.0116. Live state
re-verified via the Kaggle API: both picks `COMPLETE` with returned scores, zero submissions used
today, slug unversioned. Two deltas past the floor turned up that were not in `SESSION_SUMMARY`:
`typewell_blocked_selection` is the delta between the two ALREADY-LOCKED picks and validates them
rather than adding a candidate, and it prices the stacking-level typewell leak at **0.0281 ft**;
`gbdtbag_lgb_swap` is the same unshippable family. Full record in `SESSION_SUMMARY.md`.

**External literature read the same session, `notes/external_literature_2026_08_05_read.md`.** The
Deep Hierarchical Graph Correlator is CLOSED by its own numbers, tying cross-correlation at p = 0.15
and losing to DTW. TST3D publishes no accuracy figure. **`negatives.md` N3 is misattributed:
SPE-202046-MS is the HOST COMPANY's paper (Denisenko, Kuvaev, Uvarov, Kushmantzev, Toporov, OOO ROGII
EUROPE), and the same four inventors' patent `US11480045B2` discloses its objective for free.**

## 2. What the last sweep measured, so nobody re-runs it

Six lenses, each adversarially verified by a separate agent instructed to refute. The verifiers cut
two headlines apart, which is the reason to trust the rest:

| candidate | finder | verifier, honest |
|---|---|---|
| residual const+slope | -0.0395 | **-0.0006** once `k` is nested; the headline was an un-nested 2-D argmax |
| PF blend cell (0.18, 0.18) | -0.0427 | **+0.0084** for the legal cross-fitted cell; an amplitude control reproduces 48-61 percent with no extra PF |
| bagged GBDT `bag4_and_gbdtbag` | -0.0536 | **-0.0375**, 3/5 folds, once the family pick is nested — and unshippable |
| U-projection, 4 banks | -0.0200 | selected on full-data rho across 5 forms, optimistically biased |
| confidence shrinkage | — | best -0.0133; a 400-draw permutation null shows the legal family has NO measurable ceiling against a -0.5237 oracle |

**The bagged-GBDT lesson is the transferable one.** Its gain was real and lived entirely in the half
that cannot ship: `zz_gbdtbag_lgb` is 28 GBDT families each needing its own inference-time feature
construction, several reading artefacts with no test-side counterpart. The cheaply shippable half,
a 4-seed bag of the ratecoupled GBDT using features the kernel already builds, is worth **-0.0116**
CI95 [-0.0454, +0.0223] alone. **Statistical significance and portability are two separate gates.**

## 3. Everything else measured is closed or below floor

* **The resampling axis, both knobs.** `psr2@160` -0.3603, `psr4@160` -0.3755, `psr8@80` -0.3744,
  `psr4@240` -0.3753: a 0.015 ft span against a 0.051 ft seed spread. `psr16`, width and `gr25` are
  flat or worse. The epochs-versus-cuts confound resolves as BOTH saturated.
* **Cross-recipe averaging.** Best pool -0.0204 against the seed-3 average. Mean pairwise error
  correlation FLAT at 0.787-0.801 across every pool tried. Different recipes do not decorrelate
  further than different seeds; the diversity is the training draw.
* **Test-time augmentation, all five axes.** Bin-grid origin -0.031 honest against a 0.078 oracle
  ceiling; typewell jitter a null; the typewell's sub-foot grid phase a non-axis (the model is
  provably invariant, spread 0.0028 ft); prefix shortening significantly WORSE at +0.18 to +0.65 ft,
  because prefix length is information the model spends rather than a nuisance it is invariant to.
* **Base pruning.** `cnn_1d_v1_avg3` is 0.0 in the seven-base fit and all five folds; dropping both
  zero-weight bases is -0.008 with the CI spanning zero. Shippable for robustness, not a gain.
  **But see the trap in SESSION_SUMMARY: it is NOT inert in the six-base GRU-free fit.**
* **The datum GR channel**, capped near rho 0.28, closed four ways. The aggregator was the fourth:
  four robust consensus estimators all lose to the incumbent mean, which earns its keep by
  1/sqrt(N) averaging already saturated at ~25 windows.
* **Re-selecting the post-processing cell nested by well is DESTRUCTIVE**, +0.109. Freeze it.
* **Synthetic pre-training is BLOCKED, not closed.** Every bank roughens its own template well's
  anchored degree-1 `U` residual by 1.67x to 1.73x. Unblock condition: a bank at or below 1.25.

## 4. The one open axis — CLOSED 2026-08-04, successor opened

`augmentation_induced_ensemble_diversity` is **closed**. Every way of deepening ensemble diversity
at fixed member count from artefacts already on disk was measured, and all three are dead for three
different reasons. `reports/axis_ensemble_diversity_fixed_members_2026_08_04.json`.

* **Weight averaging / SWA is not an operation that exists for this bank.**
  `aws/src/train_seq_alt.py` calls `set_seed(seed)` with the arm's own seed and then constructs a
  fresh model inside every fold, so the nine psr4 replicates are nine independent random inits with
  no shared trajectory. Measured on fold 0 over 307,394 parameters: cross-seed weight cosine
  **0.049**, indistinguishable from the same seed at an adjacent fold (0.048) and BELOW two FRESH
  random inits (0.095). The linear interpolation midpoint between two seeds scores **86.35 ft**
  against 10.56 at the endpoints and **14.63 for carry-forward**; the 9-way weight average is
  102.50 against the 9-way prediction average's 10.37. The seeds are not mode connected. Replicated
  on fold 3 at barrier +82.20 ft. No BatchNorm confound: the architecture is GroupNorm/LayerNorm
  only and the checkpoints carry zero running buffers, so there is no recalibration pass to have
  skipped.
* **Snapshot ensembling is structurally impossible from disk.** `train_fold` keeps one
  best-validation state in memory and writes it once after the epoch loop, so all 46 GRU
  checkpoint directories hold exactly `cnn_1d_fold0..4.pt` plus `metrics.json`. A repo-wide search
  finds **zero** epoch-indexed weights. Reopening needs a retrain with periodic saves, which is a
  training cost and outside this axis.
* **Diversity-encouraging selection is a clean negative, judged corrected with a rebuilt bank.**
  A 45-arm pool, subset chosen inside each outer training fold to minimise mean pairwise error
  correlation under a pre-registered 1.05x standalone-quality floor, at the comparator's own member
  count of nine. Ramped **+0.0305 ft against the fixed nine seeds**, CI95 [-0.0164, +0.0799], 2 of
  5 folds improved. **All sixteen cells of the (k, floor) surface lose**, +0.027 to +0.246 ft, as
  do the three same-recipe seed-pool cells, and the pre-registered cell is the best of the
  nineteen, so no untried setting is hiding a win. No rule-13 inversion: the correction is worth
  -0.1272 on the comparator and -0.1223 on the candidate.

  **Both easy explanations for the null are refuted, which is what makes it worth having.** The
  search did not fail: exhaustive enumeration of all `C(13,9) = 715` admissible subsets per fold
  reproduces the greedy PATH byte for byte, 0.0 ft, so the corrected verdict covers both. The
  statistic is not noise: the pairwise error-correlation matrix transfers at Spearman **0.915 to
  0.952** train to validation.

  **The frontier is measured rather than asserted, and it is sold at 3.4x its value.** Loosening the
  quality floor from 1.05 to 1.25 at nine members buys real decorrelation, mean pairwise error
  correlation **0.7831 -> 0.7284**, and the equicorrelated-average identity calibrated on the
  observed 3-to-9 step prices that 0.0547 drop at **-0.0546 ft**. The arm is **+0.1302 ft worse**.
  The same identity, **0.998 ft of CV per unit of member correlation**, says clearing the 0.05 ft
  shipping floor by decorrelation alone needs the members driven from 0.7825 to about **0.73**,
  which this bank reaches only at several times that cost in accuracy. Diversity and quality point
  the SAME way on a bank whose best recipe is also its most decorrelated.

  Even a small HINDSIGHT oracle loses: the best of the random subsets, scored on the very wells it
  was chosen from, reaches 7.3781 against the fixed nine's 7.3723.

  Two details that make the mechanism concrete. The rule swaps three of the nine seeds for three
  near-cousins, mostly the 80-epoch psr4 arm and psr2 at 160, and that trade is the whole +0.027; on
  the one fold where the floor admits only seven members, all seven ARE psr4 seeds and the fold
  delta collapses to -0.0004. And inside the nine same-recipe seeds min-correlation selection LOSES
  to random at every member count, which is what selecting on a near-degenerate statistic looks
  like.

**Also settled, and it retires the axis's own worry.** The registered concern was that the ensemble
and the datum correction are substitutes, since rho fell 0.2777 to 0.2490 from one seed to three.
It does not continue: 3 to 9 seeds takes rho **0.2490 -> 0.2551** and the correction gets STRONGER,
-0.1198 -> -0.1272, so the ramped gain -0.0550 is essentially the whole uncorrected -0.0543 rather
than a fraction of it. Still below the 0.05 ft floor.

**Successor axis, open: `permutation_aligned_weight_fusion`.** The barrier above closes NAIVE
co-ordinate-wise averaging, not weight-space fusion in general; hidden-unit permutation symmetry is
exactly why unrelated inits look unrelated. First experiment and its bar are in
`harness/forward_axes.json`. `make breadth-status BEST_CV=7.204`.

## 5. If nothing clears the bar, the honest use of the remaining slots

Insurance. `55234752` introduced genuinely new inference code (a second Ridge, a second projection,
a second matcher pass). If it or a successor returns a public number far off what the CV-to-board
fit predicts (slope 1.5818, residual sd 0.1755), that is a port bug and is diagnosable. Hold a slot
to fix it. Do not spend a submission chasing a public number: the board cannot rank models closer
than about 0.3 ft.

---

# ARCHIVE OF THE EARLIER QUEUE, kept for provenance

# QUEUE — what to train next

One section per job. Delete a section when its base is in `LEDGER.tsv`.

**Read `reports/competitor_intel_2026_07_28.md` before planning anything.** It contains the
measurements that reorder this file, and they are not obvious from the code.

## Active forward model campaign

This list must retain a bounded next action while the user target of CV below 6 remains
unreached. `make breadth-status` reads `harness/forward_axes.json`.

**Read `notes/emission_certificate_2026_08_02.md` before anything else on this axis.** It is the
session's single document and it refutes the framing this file carried until 2026-08-02.

### What is banked and shipped

A per-well DATUM correction from trust-gated typewell window matches. Fully cross-fitted CV
**8.2251 -> 8.0435**, weighted correlation with the true datum 0.2718, five folds of five,
against rolled +0.0415, across-well shuffle +0.0208 and the cross-fitted global scalar +0.0158.
`src/trust_datum.py` is the single implementation used by both CV and inference and is verified
against the banked measurement at 1.33e-15. Submissions `55184703` (coarse reference, public
**7.581**) and `55190292` (fine reference, pending). It TRANSFERS to the geom pick with frozen
constants, 8.2826 -> 8.1757, and correcting both picks leaves their error correlation at 0.9263
against 0.9305, so the pair stays decorrelated.

### 2026-08-03 afternoon: what moved, and what is running

**BANKED, -0.11266 ft.** Aim the loss at the term that survives the pipeline. `compute_loss`
weighted absolute TVT at 1.0 and 58.6 percent of absolute TVT is the datum `src/trust_datum.py`
already corrects downstream, so most of the gradient went to a quantity another stage fixes.
`--w-shape 5.0 --w-global 0.25` on the GRU gives real-minus-null -0.0963 against -0.0338 for the
same model without it, outside the 0.033 ft band every seed, width, depth and lookahead arm falls
in. Corrected 7.85499 against the shipped 7.97724; frozen ramp at LAMBDA 0.25 MU 1.4 prices at
**7.82033** against the incumbent's 7.93299 under the identical leave-one-fold-out rule. Kernel
generated and verified, NOT pushed. `notes/shape_supervision_2026_08_03.md`.

**CLOSED, each with a control, do not re-propose.**

* *The datum AGGREGATOR.* Four robust consensus estimators against the incumbent
  isolation-weighted mean: median 0.2323, +/-2 ft vote 0.1756, KDE mode 0.1697, mixture MAP
  0.1509, summed raw profiles 0.0626, against 0.2772. Cross-fitted constants pushed every arm
  back toward a mean in 5 of 5 folds. The motivating argument is recorded as WRONG: rho is
  scale-invariant so dilution cannot attenuate it, and the mean earns its keep by 1/sqrt(N),
  0.1011 at one window to 0.2759 at all ~25, already saturated.
* *The ~13 ft alias lattice does not exist.* Real/rolled at 12-14 ft is 0.73, i.e. DEPLETED, and
  competing peaks read off the profiles are flat from 4 to 16 ft. `AGENTS.md` is corrected.
* *"U is linear so the hidden region is one parameter."* R^2 replicates at median 0.9923 and means
  nothing: a PERFECT one-parameter slope oracle scores 7.5872 against the 7.82 we hold. The
  anchored ladder in U, never priced before: 1 dof 7.5872, 2 dof 4.2792, 3 dof 2.9794, 4 dof
  2.2584. The anchor beats a degree of freedom once curvature is allowed.
* *The sparse StarSteer dip / TV-prior reading.* Piecewise-linear LOSES to polynomials at equal
  dof, 3.7612 against 2.9794 at 3 dof. `dU/dMD` has excess kurtosis **-0.78**, flatter than
  Gaussian rather than spiked. Do not rebuild the estimators around a TV prior.
* *The decomposed level+shape Ridge.* -0.0075 raw, below the 0.02 bar, and it moves SHAPE the
  wrong way at +0.0131. Tied control reproduces the deployed Ridge at 0.00e+00.
* *The host's conditional prefix tip* (Kuvaev, 698825: use the lateral's pre-PS GR when the well
  travels NEGATIVE in TVT). The class is real, 348 of 773 wells and 43.9 percent of squared error,
  and legally identifiable at 0.8435 agreement from the sign of our own predicted drift. The
  prefix still loses INSIDE it: hit-rate delta -0.002 / +0.005 / -0.008 / -0.014 at
  L=300/400/600/800, paired rho 0.1322 against 0.2327. The cross-fitted gate selects
  typewell-only in 5 of 5 folds; the ORACLE gate is WORSE than nothing.
* *Synthetic pre-training, BLOCKED not closed.* Paired against each synthetic well's OWN template,
  `data/synth` roughens the anchored degree-1 U residual by **1.730x** on 83.4 percent of wells and
  `synth_v2_cnnf0` by 1.667x on 80.2 percent, against a real median of 4.9973 ft. That corrupts
  exactly the term shape supervision just won on. The leak worry does NOT hold and was measured:
  Gate B slope R^2 -0.006 for `data/synth` against the real 0.060; the 0.39 the v2 docstring cites
  belongs to `data/synth_highdrift`, the gate's default directory. **Unblock condition: a bank
  whose paired U-residual ratio against its own template is at or below 1.25.**

**A self-inflicted floor nobody had measured.** Passing the TRUE labels through our own
post-processing costs **1.5497 ft**: savgol(17,3) is nearly free at 0.0264 and the anchored robust
degree-4 U-projection at blend 0.70 does all of it. That is 19.8 percent of the current CV, and it
explains why `probe_setk_postproc_retune` reads `PROJ_BLEND` flat at -0.0021: the variance removed
and the signal destroyed are balanced AT THE CURRENT ERROR LEVEL. The term becomes binding as the
model improves, so retune it jointly with any material gain rather than treating the axis as shut.

**Running.** Two AWS fleets queued behind a roughly 5-concurrent `ml.g5.2xlarge` limit: four
resolution arms (`gr_filter` 50 -> 9 and 25 at fixed sequence length, then 6 ft and 3 ft bins with
`h_future` scaled to hold the 9,216 ft lookahead — never varied in 28 jobs of this family, and the
query is smoothed at 50 ft against a 1 ft reference), and four prediction-start resampling arms
with a `--ps-resample 0` control.

### The one open axis

The registry currently holds no open axis after the confidence-weighted shrinkage closed as
MARGINAL. **Open one before running anything else**; `make breadth-status` enforces it. The
honest starting point for whoever picks this up is the summary below, not a fresh idea.

### What the session established, in one place

1. **The emission is SOUND.** At the true path the typewell puts the datum argmax within 1 ft on
   76 percent of wells, contrast 0.3688 against 0.0012 rolled. The recorded claim that it is
   information limited is wrong.
2. **Localisation decays with the SHAPE fed in**, 76.4 percent within 1 ft at 0 ft of shape error
   down to 18.4 at our 4.91. A path at 1 to 2 ft makes the datum nearly free. This is the
   governing curve.
3. **The emission is a sound discriminator over a LOW-DIMENSIONAL family and an unsound objective
   for free path search.** One parameter, truth wins on 76 percent; two parameters, no signal;
   65^25 paths, truth loses on 153 of 153. This explains the 90-of-90 result and retires it.
4. **Freedom that can move the implied TVT is harmful; freedom orthogonal to TVT is necessary.**
   Per-node TVT gradients break decoders; the window's per-window affine GAIN refit is
   load-bearing and removing it costs everything.
5. **CONTRAST, not oracle R^2 at the truth, is the figure of merit for a reference log.** The
   prefix wins on R^2 and loses on hit rate. Contrast is set by resolution, and the estimator
   wants MAXIMUM resolution on both logs: reference unsmoothed, query raw.
6. **The per-well datum is spatially white**, structured fraction 0.183 and not distinguishable
   from position-shuffled, so no neighbour pooling can help.

### Closed on 2026-08-02, do not re-propose without new evidence

Joint (datum, slope) correction; window-local MD-interpolated correction; iterating the
correction; log-likelihood aggregation across windows; per-well path selection over the base
bank; the from-scratch self-consistency beam decoder; sequential decoding in ANY form, including
the globally calibrated one; field-scale or kriged pooling of offset constraints; the prefix as a
second reference; cross-window agreement as a trust statistic; per-well fixed calibration;
query-side GR smoothing; reweighting the base bank for shape. Each has a control and a stated
scope in the note.

## The four numbers that govern what is worth doing

1. **Recombination is capped.** In-sample least squares over all 60 banked bases gives 7.9537.
   Nothing below 8 comes from another base, seed or blend.
2. **A competitor's single pure-physics model scores CV 6.85** on our exact pooled metric, while
   our five-base stack is 8.6082 and our own PF is 10.3611. The deficit is the FORWARD MODEL.
3. **Our tracker error accumulates 3.7x from heel to toe** (3.07 ft over the first 20% of the
   eval region, 11.40 ft over the last 20%), flat across well-length quartiles. Our PF's SHAPE is
   good — fix only its datum and it scores 6.6720 alone. It loses lock as it integrates.
4. **This CV cannot see a per-well correction smaller than ~0.10 ft.** Under the curvature
   weighting that the pooled loss actually applies, the 773 wells have a Kish effective N of 205
   (NOGEOM) and 158 (PICK2), and every arm measured on 2026-07-28 had a well-cluster bootstrap
   sd near 0.05 ft. Do not queue post-hoc corrections whose honest point estimate is under 0.10.

So: work on the tracker's drift. Not on more bases, and not on reweighting what we have. The
per-well rescaling axis was opened, measured to its ceiling and priced on 2026-07-28, and it is
shut; the identity below is why, and it is worth reading before proposing anything per-well.

---

## The polarisation identity, and why it reorders everything

Measured and independently re-verified 2026-07-28,
`reports/verify_polarisation_2026_07_28.txt`. With `r = pred - last_known`, `t = y - last_known`,
`e = pred - y`, and the per-well squared norms `den = <r,r>`, `S = <t,t>`, `N = <e,e>`:

```
c_w = <r,t>/<r,r> = (den + S - N) / (2*den)        exact to 3.7e-14
```

`c_w` is the per-well scale that minimises the deployed loss. `den` is observed with no truth.
So the entire per-well correction is governed by two unknown MAGNITUDES, and rank correlation
against `c_w` says only one of them binds:

| statistic | Spearman vs `c_w`, NOGEOM / PICK2 |
|---|---|
| `S/den`, the well's TRUE drift energy | **+0.692 / +0.706** |
| `N/den`, the error energy | +0.023 / +0.005 |
| `den`, observed | +0.009 / +0.010 |

Ceilings with a perfect oracle for one magnitude and the **link cross-fitted**, against the
cross-fitted global scalar (8.7107 / 8.9474):

| arm | NOGEOM | PICK2 |
|---|---|---|
| `N`-oracle, affine | 8.5523 (-0.158) | 8.8847 (-0.063) |
| `S/den`-oracle, isotonic | **7.3658 (-1.345)** | **7.5262 (-1.421)** |
| both magnitudes true | 6.2517 (-2.459) | 6.2159 (-2.732) |

Every uncertainty feature this project owns (`pfgap`, `div_spread`, `dense_std`, `dtw_stoch_std`)
predicts **N**. So the headroom lives in **S**, and `S` was then built and priced.

**The price of S**, from corrupting the true `S` and running the same cross-fitted isotonic
conversion. Read it by den-weighted correlation on `log(S/den)`:

| corr | NOGEOM | PICK2 | | corr | NOGEOM | PICK2 |
|---|---|---|---|---|---|---|
| 1.000 | -1.345 | -1.421 | | 0.825 | -0.846 | -0.882 |
| 0.968 | -1.257 | -1.328 | | 0.714 | -0.724 | -0.716 |
| 0.915 | -1.150 | -1.172 | | 0.619 | -0.426 | -0.453 |
| | | | | 0.478 | -0.250 | -0.157 |

A ledger KEEP (-0.05) needs correlation about **0.45**. Half a foot needs **0.70**.

**What we reach is 0.316**, and only from the stack's own residual shape. Split by feature
origin, everything else is at or below zero correlation with `S`: alignment (dtw, gr_offsets,
self_corr, beam, 378 columns) -0.021, well geometry -0.063, cross-well spatial agreement +0.002,
raw log statistics -0.058. The converted best arm is -0.1014 on NOGEOM with 95% CI
[-0.3922, +0.1326] and **+0.0694 on PICK2**, the same arm with the opposite sign.

**Cross-fit the link or the number is fiction.** In-sample links produced -0.356 (isotonic) and
-0.617 (degree 5) on the N-oracle; cross-fitted they are +0.005 and +118. Reproduced here.
Assert monotonicity of any prize curve in the corruption level: an affine link on the
heavy-tailed `S/den` produces a curve on which a corrupted `S` beats the true `S` by 0.26 ft.

**Width is poison at 773 wells, even with an in-fold screen.** Three confirmations on
2026-07-28: the full 1446-column matrix reaches R2 -0.073 and -0.233 where 16 columns reach
+0.078; adding 378 alignment columns to those 16 cuts correlation from 0.316 to 0.112; the
703-column neighbour group reaches -0.266. Start from the 16 and add deliberately.

---

## 1. Attack the drift accumulation in the tracker

The measured failure is drift, not mis-lock: near the heel we are already at ~3 ft. Ideas, in
order of cost:

- **Bias-correct the dip.** The 3.71x growth is faster than diffusion (sqrt(5) = 2.24) and slower
  than a pure linear bias (5x), so a systematic dip-error component exists and is estimable.
- **Emission model / GR calibration** between horizontal and typewell. The tracker's likelihood
  is where lock is won or lost.

Do NOT re-try a full-typewell GR re-anchor. It is catastrophic because posterior-mean readout
lets even a small fraction of global proposals drag the estimate to unrelated aliases:
`REANCHOR_GENTLE`, 2% every 800 rows, scores 80.61 on `pf_scale_5` against 11.27 for BASE.
Any future re-acquisition must retain the local posterior and demonstrate a different mechanism
before consuming a full rebuild.

The repaired E2 dynamic-program matcher has been run at all 773 wells. The raw `lam=0.01` base
was MARGINAL. A fixed five sample quadratic GR smoother produces
`untried19_lam0.01_sg5`, a KEEP: NOGEOM blend add is -0.1032 and the rolled control is -0.0443.
The fixed portable stack of the current kernel plus this base scores 8.5105 cross fit, down
0.0999 from 8.6104. The next task is a verified inference port, not another lambda or smoother
sweep. The physical width mismatch in the current smoother remains one predeclared future
experiment after the port is settled.

The physical 6.5 ft smoother is now WASH, and the posterior readout is MARGINAL. A globally
persistent slope and gain decoder then failed its mechanism bar at standalone 33.8827. It held one
local shape for the whole well and is closed. A future coupled state decoder must allow local shape
changes and prove its emission reduction control before it consumes another full 773 well run.

The local shape adaptive decoder also failed its standalone bar. It applied the best E2 local shape
at each candidate TVT cell to centre the next transition. That reduced the global shape failure to
27.2320 RMSE, but it remains far above carry forward at 15.9107 and the accepted E2 at 26.6806.
It was not sent to the base gate. This closes both the globally fixed and nodewise independent
shape transition means. Do not rerun either decoder form. A future coupled state must demonstrate
a smaller state construction with a predeclared emission reduction control.

A one pass self reference emission update also failed B1. It reproduced accepted E2 exactly on
pass one, then placed calibrated lateral GR into typewell TVT bins from that pass and blended the
two reference logs one half each. The refined path scored 27.4834 against accepted E2 at 26.6806.
The within well rolled track control was still worse at 31.3891, which confirms the alignment
operation rather than generic smoothing caused the change. It nevertheless misses the required
admission bar, so it is not a base and was not sent to the gate. Do not iterate reference mix
weights or run another self reference pass.

The first direct E2 feature GBDT is MARGINAL, not a KEEP. It adds the accepted E2 residual to the
ten deployed median feature groups and retains a 9.0484 standalone RMSE under the row step four,
1200 tree handicap. Its NOGEOM blend add is -0.0421 against a rolled null of -0.0119. The margin
is -0.0302 where a KEEP requires -0.05, and its error correlation is 0.951. Do not port or
submit this cheap model. A full recipe is justified only if spare compute remains after a new
forward model attempt.

The public HMM rate state was tested on the accepted sg5 E2 emissions. It carries 41 states for
d(TVT + Z)/dMD, a prefix rate prior, a rate random walk and a five point position transition. The
local vectorized Viterbi implementation runs all 773 wells in 450 seconds, so runtime is not the
reason to abandon it. Its 27.0097 standalone RMSE misses accepted E2's 26.6806 B1 bar. Do not
send it to the gate or tune its rate, position or emission constants. The public notebook's claim
cannot justify a parameter search because its strong result depends on external learned assets.

The posterior readout of that same fixed rate state is a KEEP. It scores 19.4995 standalone and
has NOGEOM blend add -0.2146 against the rolled null -0.1187, with all five LEAK folds improved.
The six base portable stack has cross fit CV 8.3811 and full fit 8.3518, beating the submitted
E2 stack by 0.1294 cross fit. Its legal inference source reproduced absolute OOF TVT with mean
difference 8.5e-09 ft and maximum 7.9e-05 ft. Kernel version 1 scored normally. The user
selected it on the web UI as Pick 2.

One narrow model test is live: `lgbmede2ratefeature_2026_07_31` adds only the accepted E2 residual
and rate posterior residual to the deployed ten group median LightGBM recipe. A full five fold
OOF is required. It then needs the standard rolled control and gate. Do not expand the feature
set, tune parameters, or port it until it clears the gate.

**Result.** It is a KEEP: standalone 8.7453, NOGEOM blend add -0.2582 versus rolled control
-0.0374, five of five LEAK folds improve. Added to the six base rate posterior stack it gives
cross fit CV 8.3422 and full fit 8.3019. This is incremental, not the sub 5 single model
mechanism. A test port is only justified after the active direct matcher work.

**2026-08-01 replacement result.** The rate coupled likelihood is too inaccurate to enter the
Ridge itself, but its residual is a stronger LightGBM input than the posterior alone.
`lgbmede2ratecoupledfeature_2026_08_01` adds accepted E2, the rate posterior, and the rate
coupled residual to the deployed ten feature groups. It is a KEEP: standalone 8.5161, NOGEOM
blend add -0.4088 versus rolled control -0.0481, and all five LEAK folds improve. Replacing the
earlier GBDT gives portable seven base cross fit CV 8.2251 and full fit CV 8.1833. The next step
is limited to an OOF exact legal inference port and its five fold LightGBM artifact. Do not add
both nearly identical GBDTs to the Ridge.

The coupled matcher itself is now closed as a standalone predictor. Its 17.4909 RMSE improves
on rate posterior 19.4995, while missing carry forward 15.9107. The reduced control gives
24.8337 and the rolled control 49.4023. The direct state template test is also closed. It
generated each local GR template with the nine rate states and unit survey gain, removing the
unequal shape bin maximum. Its full 773 well RMSE is 33.9114, versus 17.4909 for the coupled
profiled emission, 36.8873 for its reduced control, and 46.7203 for its rolled control. This
misses the predeclared 17.4909 mechanism bar by 16.4205 ft. Do not tune its states, transition,
window schedule, smoothing, or grid to rescue it.

Also do not re-try prefix-dip extrapolation (corr 0.030, dead across five windows), more matcher
capacity (dim128 MARGINAL, WarpMatch-XC killed), or a single-effective-dip model (anchored oracle
is only 12.44).

## 2. k256.dev's hint: an auxiliary future-related target

From discussion 717573: *"predicting an auxiliary future-related target and then using those
predictions as features for the main model"*, borrowed from financial forecasting. He reached
tabular LB 6.798 with ordinary LightGBM/CatBoost/XGBoost and *"no feature engineering"*, insisting
*"what really matters is the input features themselves"*. Two-stage, untried here for the MAIN
target. Note that `S`, the per-well drift energy, is exactly such an auxiliary target and was
tried on 2026-07-28: it reaches correlation 0.32 against the 0.45 a KEEP needs. That is evidence
about `S` specifically, not about the two-stage idea, which remains untested on a row-level
auxiliary target.

**2026-08-01 result.** The one predeclared future trajectory test is functionally a wash.
Three strict GroupKFold predictions of future `U = TVT + Z` at 250, 500, and 1000 ft were fed
to the median LightGBM. It has standalone RMSE 8.4986 and is formally a base KEEP, with NOGEOM
blend add minus 0.4084 against rolled minus 0.0488. But its portable stack cross fit CV is
8.2247, only 0.0004 ft below the rate coupled GBDT at 8.2251. The two models are substitutes,
not a new path. Do not port the future target model or add both models to the Ridge.

## 3. Neighbour-well profile transfer — real upside, real risk

Competitor De DQ says top teams copy a close neighbour's TVT profile when one is within ~150 ft.
Tucker Arrants (LB 5.444) says neighbours are not needed at all. They disagree, so treat it as
unresolved. Our own geom-prior work found its CV edge is within-field leakage that REVERSES by
0.233 ft on the public LB, so any version of this must be validated field-blocked before it is
believed.

---

## NOT QUEUED — measured shut, do not re-propose without new evidence

- **Shrinkage keyed on predicted UNCERTAINTY, and the whole per-well rescaling axis.** Opened,
  measured to its ceiling and priced on 2026-07-28. A perfect `N` oracle with a cross-fitted
  link buys only -0.158 / -0.063. The idea was not wrong, the magnitude was, and predicting the
  right magnitude `S` needs correlation 0.45 for a KEEP where we reach 0.32. See the identity
  section above for the full price table before re-proposing any of this.
- **Per-well stack/pf blend weight.** -0.0712 on PICK2 and +0.0097 on NOGEOM once shape selection
  is in-fold, p = 0.098 under a size-preserving sign-permutation null. Below the 0.10 ft floor.
- **Where the per-well scale is applied.** Constrain the scale equal across wells and all four
  insertion stages sit within 0.0021 ft cross-fitted. Points B and C are provably the same
  estimator (`last_known` is exactly constant per well, savgol reproduces constants).
- **Sign of the per-well correction as a hard decision.** Detection is a den-weighting artefact:
  unweighted AUC 0.5058 / 0.5128 is chance. Best honest conversion is +0.014 / +0.011 vs the
  global scalar, and a PERMUTED score converts BETTER than the real one. Anyone re-proposing this
  must show den-weighted correlation between the score and `c_w - 1`, not AUC: a real score at
  AUC 0.591 carries 0.13 ft less than a graded latent at the same AUC.
- **A per-well OFFSET.** Its 5.52 / 5.62 oracle is the known 5.31 datum oracle reparameterised,
  not a new lead.
- **More CNN-1D seeds.** avg3 -0.1218, avg5 -0.1202. Returns NEGATIVE past three seeds.
- **Combining CNN-1D heads.** err-corr(avg3, mdn5) = 0.943; adding both makes the honest
  cross-fit WORSE (8.6165 vs 8.6048). Pick one head.
- **Shipping the SDF base.** ADMISSIBLE (its one differing column is the first eval column on
  every well, never the prefix, so no eval-path leak) but NOT deployable: the test-time image
  cannot be reproduced exactly, giving mean 0.138 ft / max 114 ft perturbation against a -0.027 ft
  stack contribution. See `src/cnn_sdf/kernel_infer.py`.
  **Resolved 2026-08-02, this entry is correct and the code now agrees.** Until today
  `kernel_infer.py`'s docstring claimed the repair reproduced the image at `0.000e+00` "on every
  well tested", flatly contradicting this line and costing a session time to re-derive. Commit
  `288156c8` holds the measurement: synthesising the marker from `last_known_tvt` is exact on ~70%
  of wells, and on the other ~30% the true first-eval TVT crosses a 1 ft raster row boundary so the
  marker lands one pixel off, which can flip the SDF zero-crossing decode for a whole well. The
  docstring is corrected. **Consequence: no CNN-SDF artifact can ship, so collecting one is worth
  CV knowledge only and its ceiling is the measured -0.027 ft.** That applies to the completed AWS
  trial `rogii-cnn-sdf-fall-s42-2026-08-01-16-26-31-932`. A shippable SDF would need the history
  channel rasterised from `last_known_tvt` at TRAINING time, i.e. a full retrain, which is not worth
  it at that ceiling.
- **Stacking or averaging typewells to raise reference-side SNR.** Tempting, because the NCC noise
  floor has a reference-side term and the emission is information limited. Already falsified
  2026-07-22, `scripts/probe_field_stacked_reference_2026_07_22.py`,
  `reports/field_stacked_reference_2026_07_22.json`. The field stacked reference scores held-out
  R^2 **-0.315 against the shipped typewell's -0.178**, so it loses to the channel it was meant to
  replace and fails its pre-registered bar (it beats the other-template control by +0.297, which
  only shows the stack is doing something, not something useful). The kill stands but its stated
  reason was WRONG and is corrected here, 2026-08-04: `med_sib_support = 1.34` is not the sibling
  count. The median well shares its typewell with **3.0** siblings, so there was plenty to stack
  and "nothing to stack" was never the reason. The real reason is measured: **sharing a typewell is
  a chronology and convenience relation, not a proximity one.** The host says so in discussion
  698449 — the geologist picks the typewell "that is close and available at the time", the wells
  span ten years, and some typewells are pseudo-typewells built from earlier laterals — and the
  geometry agrees, with the median eval row sitting **2,063 ft** from its nearest sibling's path
  and p90 at 7,015 ft. Do not re-propose a composite, stacked, or multi-typewell reference, and do
  not re-propose it on the grounds that the sibling count is actually adequate: it is, and the
  transfer still fails.

  **The sibling PREDICTION channel is separately dead, measured 2026-08-04** on the 681 scorable
  wells with siblings over 3,319,299 eval rows, against the deployed path on the identical rows:
  deployed 6.878, carry-forward 15.570, sibling surface at the nearest donor **75.287**, IDW over
  all siblings 68.802, random non-sibling control 120.851. The frame alignment is genuine twice
  over — the surface identity `|(TVT+Z-C) - ANCC|` reproduces at median 0.030 ft and the sibling arm
  beats the random control by 45 ft — so this is a real relation that carries no usable signal.
  Stratified by donor distance it loses to carry-forward EVERYWHERE, including the closest 2 percent
  of rows at 200 ft, 15.84 against 11.54.

  **One usable output survives:** the 260-way physical typewell grouping is an exact, label-free CV
  block. It is not a scoring lever; it is a stricter grouping than `GroupKFold(5)` by well for
  anyone who wants to test whether our CV leaks across typewell-sharing wells.
- **More GBDT feature subsets / target reparameterisation / row weighting.** Four WASH bases.
- **Signed drilling azimuth.** The direction is real but already present as continuous
  `azi_sin` and `azi_cos` in `kinematics_train.parquet`. The independent audit found target
  correlation 0.004 to 0.005 and no useful direction split. `reports/eda_v2_indepth_2026_07_16.md`.
  Do not retrain this from a competitor anecdote.
  Independently corroborated: the toolkit measured +0.476 RMSE for 46 AEON features.
- **Q-3D tortuosity as currently built.** Max |corr| 0.073 vs true drift. BUT the implementation
  is suspect — it derives angles from 1-ft interpolated XYZ, so it measures interpolation noise
  (`tort_well_mean` sd 0.43 on mean 18.25). Needs survey stations or ~90 ft decimation to test
  honestly. Not a closed axis.
- **The free PF weight, meta-stacker changes, post-processing re-tuning.** All dead, see
  `reports/stacks_v2_2026_07_26.txt`.
- **`[UNTRIED]` #19.** Wash at 773 wells; its `e1` arm is an ORACLE and is in gate.py INADMISSIBLE.
- **Public notebooks.** They saturate at LB 7.15-7.3; the one advertising "LB TOP 3" is by its own
  header a sub-9 solution.

## Traps that have each cost half a session

**Fold-0 screens over-promise.** `cnn_1d_direct` screened at REAL-NULL -0.1394 and gated WASH
(-0.0010). MDN K=3 had the best fold-0 standalone in the family (12.5253) and gated MARGINAL,
worse than K=5. 155 wells cannot rank bases that sit within 0.1 ft of each other. Use fold-0 only
to decide where to spend GPU, never as evidence a base works.

**`gate.py`'s rolled null does not transfer to per-well estimators.** It is correct for a BASE,
where a rolled copy adds roughly nothing. For a per-well factor a permuted copy is actively
destructive, so REAL-NULL turns every arm into a spurious KEEP. Judge a per-well estimator
against the cross-fitted global scalar instead.

**Fit an in-sample link and you will report a number that does not exist.** Three separate arms
on 2026-07-28 did this. Isotonic looks safe and is not: it resolved to 8 level sets whose top
well carried 67% of the den-weighted leverage.

**Verify every kernel port against the banked OOF before shipping.** Both CNN codebases build a
model INPUT from the label and silently degrade when it is absent. The CNN-1D one emitted pure
carry-forward and is fixed; the SDF one is why the SDF cannot ship. `scripts/
verify_cnn1d_inference_2026_07_28.py` is the pattern to copy.

## The 2026-08-05 "Sub-6" notebook is a board probe, and it closes the fork-wall question

`raunakdey07/rogii-ultra-sub-6-rmse`, 119 votes in under three hours, last run 06:55 UTC on
2026-08-05, title claiming sub-6 RMSE. Torn down directly, 327,974 characters over 51 cells.

**It names exactly three well ids and they are the three RELEASED test wells** — `000d7d20`,
`00bbac68`, `00e12e8b` — and no others. Its own overview states the mechanism: it "identifies the
single target well (`00e12e8b`) where the PF seed-branch hedge is applied", adds 10 percent of a
centred shape residual "hard-limited to approximately +/-0.40 ft", and preserves "the successful
+2.0 ft branch level". Source scan: 45 hits on `__horizontal_well.csv` train-file lookup, 40 on the
gold-contact lookup, 11 on hardcoded released test wells, 4 on the 14,151-row guard.

That is the Q0522 / blacklions leaderboard-descent family this repo tore down on 2026-08-01, now at
its logical conclusion: a per-well additive constant fitted to returned public scores on ONE of the
three visible wells. **The scored rerun ships about 200 different wells and the data description
says the visible `test/` wells are examples that get replaced, so every one of those constants is
inert or absent on the private set.** Its "sub-6" is not a model.

**Consequence for how the private board is read.** Two independent lines now agree: the wall's
source notebook has an honest CV of 10.4451 on our exact metric (protocol equivalence to 0.0001 ft),
and the fastest-rising notebook of the final day gets its headline from tuning a constant on a
single visible well. Public rank should read considerably better on the private rerun. This does NOT
license planning around a collapse — it is one more reason not to chase the public number.
