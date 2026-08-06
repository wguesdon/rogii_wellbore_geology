# 131st place: an LLM harness that refuses to lie to you

ROGII - Wellbore Geology Prediction

Solution Writeup · 131st place · Aug 6, 2026


Finished 131 / 6,191 with a private score of 7.860, up from 1,887th on the public board.
A move of 1,756 places.

I did not hand write the code. Three months, 595 commits, 792 Python scripts, about 171,000 lines,
all generated inside an agent system I designed, and none of it accepted until it passed rules I
wrote. So this writeup starts with the harness, because the harness is the reason the result held.

Of all 51 submissions I made, the one I selected was the best one on private. Nothing I left
behind would have scored better.

## The Harness

Agents are excellent at generating plausible experiments and terrible at judging their own results.
Left alone they find an improvement, believe it, and build three more things on top of it. Every
file below exists because that happened first.

Worth separating two things that get confused. The four files above are what an agent is allowed to
read; they are context control, and they are deliberately small against 6,400 lines of archived
strategy that would otherwise be spent on being wrong. They are not what enforces anything. The
enforcement lives in about 42 KB of Python that an agent has to pass: `gate.py`, `ensemble.py`
and `breadth_gate.py`. Documents can be argued with. Those cannot.

## The whole system in one picture

```text
   ┌─────────────────────────────────────────────────────────────────┐
   │  WHAT AN AGENT MAY READ  (four files. not the other 6,400 lines)│
   │                                                                 │
   │  AGENTS.md ......... the rules. what is true, what is banned    │
   │  QUEUE.md .......... what to run next, with the exact command   │
   │  LEDGER.tsv ........ every candidate ever gated, one row each   │
   │  SESSION_SUMMARY.md  live handoff, and anything blocking on me  │
   └───────────────────────────────┬─────────────────────────────────┘
                                   │
                                   ▼
                    ┌──────────────────────────────┐
                    │  agent proposes ONE bounded  │
                    │  experiment from QUEUE.md    │
                    └───────────────┬──────────────┘
                                    │
                                    ▼
              make preflight   ─── FAIL ───▶  STOP EVERYTHING.
              does the known-winner arm       a failing control means every
              still return 8.2826 /           number this checkout can produce
              8.7300 / 8.9641 ?               is INADMISSIBLE, not merely suspect
                                    │ PASS
                                    ▼
                    ┌──────────────────────────────┐
                    │  make gate BASE=<candidate>  │
                    │       harness/gate.py        │
                    │  the only route from a base  │
                    │       to a number            │
                    └───────────────┬──────────────┘
                                    │ runs BOTH arms, always
                     ┌──────────────┴──────────────┐
                     ▼                             ▼
             the real candidate          the SAME candidate with
                                         rows rolled within each well
                     └──────────────┬──────────────┘
                                    ▼
                    real minus rolled-null, per fold
                                    │
                 ┌──────────────────┴──────────────────┐
                 ▼                                     ▼
          margin survives                      null reproduces it
                 │                                     │
                 ▼                                     ▼
          KEEP → one row to LEDGER.tsv          KILL. the base was
                 │                              carrying per-well amplitude,
                 │                              not row-level information
                 ▼
        ┌─────────────────────────┐
        │ make lock-check         │ ◀── refuses to let the campaign END
        │ harness/breadth_gate.py │     while the objective is unreached
        └─────────────────────────┘
```

## The files, and what each one refuses to do

| file | size | its job |
| --- | ---: | --- |
| `harness/gate.py` | 15 KB | The gate for every stack base, and the only route from one to a number. Reproduces the deployed pipeline, refuses to report if the known winner control drifts, builds the within well rolled copy itself, appends exactly one row to `LEDGER.tsv`. Particle filter branches have a separate gate so they are judged against a rebuilt control. |
| `harness/ensemble.py` | 7.6 KB | Guarded greedy over gated bases. Selects on one half of the wells and reports the other, in both directions, because a best of N maximum is a maximum of N noisy statistics. |
| `harness/breadth_gate.py` | 10 KB | The completion gate. Refuses campaign completion while the objective is unreached, and rejects a written list of excuses. |
| `harness/forward_axes.json` | 57 KB | 39 forward model axes, each open, tried or closed, each carrying evidence, `next_action` and `close_reason`. An axis cannot be closed without a reason. |
| `harness/forward_target.json` | 4 KB | The objective, the working CV proxy, the deadline, and the admissible and inadmissible reasons to stop. |
| `harness/campaign_override_log.tsv` | 2.5 KB | Every early stop, with its justification, timestamped. A campaign can be closed early but never quietly. |
| `LEDGER.tsv` | 5.9 KB | 52 gated candidates, twelve columns: standalone score, error correlation against two blends, blend add, rolled null, folds improved, verdict. Outcomes were 21 KEEP, 16 WASH, 14 MARGINAL, 1 REJECTED, so fewer than half of everything measured survived. |
| `archive/kill_audit.md` | 40 KB | Re examinations of previously rejected ideas. Several kills turned out to have been measured on the wrong axis. |
| `archive/negatives.md` | 7.7 KB | Numbered negative results, each with an expiry condition stating what would overturn it. |

## The one I would steal for any other project

`make lock-check` will not let a campaign end while its objective is unreached, and it carries a
machine readable list of excuses it rejects:

```text
   ┌─────────────────────────────────────────────────────────┐
   │  INADMISSIBLE reasons to stop                           │
   ├─────────────────────────────────────────────────────────┤
   │  x  not enough time remaining                           │
   │  x  too close to the deadline                           │
   │  x  this is post-deadline work                          │
   │  x  this is a multi-day build                           │
   │  x  the remaining ideas are too speculative             │
   │  x  we have plateaued                                   │
   │  x  every axis is closed                                │
   │  x  the expected gain is too small to be worth it       │
   │  x  further work is unlikely to succeed                 │
   │  x  the score is already locked in by the picks         │
   │  x  the remaining time is better spent documenting      │
   ├─────────────────────────────────────────────────────────┤
   │  ADMISSIBLE, and only these four                        │
   │  +  the objective is reached                            │
   │  +  the deadline has passed                             │
   │  +  the owner instructs the work to stop                │
   │  +  a NAMED external blocker, with the attempt that     │
   │     hit it                                              │
   └─────────────────────────────────────────────────────────┘
```

Every rejected line is something an agent said to me, in those words, to justify stopping early.
There is also a deferral detector, so "next session" and "after the deadline" trip the gate.

## Four rules that each caught a real error

Ship a control in the same run. The rolled null in the diagram above. A base whose blend add is
reproduced by a within well shuffled copy of itself was never carrying row level content.

Nest selection inside the fold. Choosing the best of N arms on full data and reporting that
arm's CV inflates it by about 40% of the headline here. Measured, not assumed.

Judge the corrected object. One base set beat its replacement uncorrected and lost to it
corrected, because the datum correction's value depends on the shape of the path it wraps.

Rank by CV. Use the board to price a target and prove a kernel runs, never to rank. The section
below is what that bought.

## Solution Overview

### The whole pipeline in one picture

```text
   WHAT EACH TEST WELL HANDS YOU
     trajectory   X, Y, Z at every measured depth
     gamma ray    the whole lateral, labelled rows and not
     TVT          the heel only, a median 26% of rows
     typewell     one vertical reference log, geology known
                            │
                            ▼
       ┌───────────────────────────────────────────────┐
       │  1. Seven base models, each guessing drift    │
       │     a different way                           │
       │                                               │
       │     rate-coupled LightGBM    rate-HMM  x3     │
       │     RealMLP                  1D CNN           │
       │     GRU sequence, 9 seeds                     │
       └───────────────────┬───────────────────────────┘
                           │ seven OOF columns
                           ▼
       ┌───────────────────────────────────────────────┐
       │  2. Positive Ridge, cross-fitted              │
       │     GroupKFold(5) BY WELL, never by row       │
       └───────────────────┬───────────────────────────┘
                           │ one stacked drift curve
                           ▼
       ┌───────────────────────────────────────────────┐
       │  3. Blend with a particle-filter selector     │
       │     last_known + 0.88 stack + 0.12 selector   │
       └───────────────────┬───────────────────────────┘
                           │
                           ▼
       ┌───────────────────────────────────────────────┐
       │  4. Smooth, then project                      │
       │     Savitzky-Golay(17,3) per well, then an    │
       │     anchored degree-6 fit in normalised MD    │
       └───────────────────┬───────────────────────────┘
                           │ a physically plausible path
                           ▼
       ┌───────────────────────────────────────────────┐
       │  5. Trust-gated typewell datum correction     │
       │     match the path against the reference log, │
       │     gate on peak isolation, one offset/well   │
       └───────────────────┬───────────────────────────┘
                           │ per-well shift estimate
                           ▼
       ┌───────────────────────────────────────────────┐
       │  6. Apply it as a RAMP, not a constant        │
       │     shift x (0.3 + 1.0 x normalised MD)       │
       └───────────────────┬───────────────────────────┘
                           │
                           ▼
      TVT FILLED IN FOR EVERY UNLABELLED ROW
              CV 7.159      private 7.860
```

### Why step 6 is a ramp

This was the single most valuable observation in the competition, and it is visible the moment you
plot the residual against measured depth instead of summarising it.

```text
   mean signed residual of the SHIPPED path, by position along the well
   773 wells, 3,783,989 rows, deciles of normalised MD

  +0.2 │                                             ▄▄▄▄
   0.0 │          ▄▄▄▄ ▄▄▄▄ ▄▄▄▄      ▄▄▄▄ ▄▄▄▄ ▄▄▄▄
  -0.2 │     ▄▄▄▄                ▄▄▄▄
  -0.4 │▄▄▄▄
       └──────────────────────────────────────────────────
        heel                                        toe

        -0.39 -0.19 -0.04 +0.03 -0.00 -0.11 -0.06 -0.01 +0.06 +0.12
```

Two separate measurements, and it matters which is which.

Before the ramp existed, measured 2026-08-03 on the then-deployed path: the per-well constant
and the per-well slope of the residual correlate at +0.7640, 58.4% shared variance. The
prediction anchors at the last known TVT and drifts away roughly linearly, so the datum error and
the slope error are not two prizes, they are one defect seen twice. That is the entire argument for
a ramp: a per-well estimate applied as a flat shift fixes the heel and leaves the toe.

After, the profile above, which is the shipped path measured over all 773 wells. Every decile
now sits inside 0.4 ft and the residual no longer climbs monotonically toward the toe. Re-shaping
the correction, changing nothing else, was worth 0.044 ft on CV.

Honest caveat on that picture: it is the residual that survives, not the one the ramp was built to
kill. The heel deciles are the worst, which is the opposite of the pre-correction shape and suggests
the ramp slightly overcorrects early in the well. Re-fitting lambda was measured and declined at
a 0.0087 ft wash, so this is a known and priced imperfection rather than an oversight.

## The other lever: virtual prediction-start resampling

A training well is labelled everywhere. The boundary between "known prefix" and "predict this" is
derived purely from which rows carry a label, so on a training well that boundary can be re-cut
anywhere.

```text
   one training well, labelled end to end
   ├────────────────────────────────────────────────────────┤

   the real task cuts it once:
   ├──────────  known  ──────────┤────── predict ───────────┤

   but it can be cut anywhere, and every cut is REAL geology
   ├────  known  ────┤─────────── predict ──────────────────┤
   ├──────────────  known  ──────────────┤──── predict ─────┤
   ├───  known  ──┤────────────── predict ──────────────────┤

   cuts drawn from the MEASURED known-fraction distribution
   median 0.26, p10 0.20, p90 0.35
```

Only the question changes; the rock and the log are untouched. Dose response was monotone against a
rolled-null control: 0 recuts −0.086, one −0.187, two −0.244, four −0.328, four with longer training
−0.376.

It also decorrelates seeds for free, because each seed draws different cuts. Three seeds of the
resampled recipe correlate 0.77 to 0.80, against 0.89 to 0.92 for ordinary same-recipe seeds, so
averaging them buys far more than usual.

CV went 8.225 to 7.159 in the final week.

## The Public Leaderboard Could Not Rank

Every submission carrying both a CV score and a final one.

| submission | CV | public | private |
| --- | ---: | ---: | ---: |
| selected A | 7.159 | 6.818 | 7.860 |
| selected B | 7.211 | 6.618 | 7.883 |
| C | 7.204 | 6.891 | 7.889 |
| D | 7.270 | 6.731 | 7.946 |
| E | 7.372 | 6.992 | 7.989 |
| F | 7.513 | 6.888 | 8.112 |
| G | 7.686 | 7.064 | 8.310 |
| H | 7.820 | 7.616 | 8.765 |
| I | 7.933 | 7.848 | 8.803 |
| J | 7.977 | 7.796 | 8.791 |
| K | 8.133 | 7.581 | 8.784 |

Rank correlation of CV to private is +0.955. CV to public is +0.818. The public set ran
0.403 ft easier than CV; the private set ran 0.714 ft harder.

The two selected rows are the whole argument:

```text
                    CV said          public said       private said
   A vs B      A better by 0.052   B better by 0.200   A better by 0.023
                     ✓                    ✗                  ✓
                                   opposite direction,
                                   4x the magnitude
```

CV was right and the board was backwards on the one comparison that decided my selection.

## What Did Not Work

The well-as-image family. Tucker Arrants (5th) mentioned on the last day that his inference is
~200 forward passes per model, one per well, each well treated as an image. We had built exactly
that months earlier: a 2D U-Net over a typewell-depth by horizontal-column image emitting a signed
distance field. I was convinced this was what we had missed.

So I spent an expiring submission on it standalone rather than argue about it. 16.204, with the
gap to the main pipeline widening from 7.70 ft on the released wells to 9.59 ft on the board. It
does not transfer standalone. Its real value is as a decorrelated stack column worth 0.017 ft, which
is where we already had it. Killing my own favourite theory was the best use of a submission all
competition.

The train/test overlap trick. Several scored public notebooks read
`train/{well_id}__horizontal_well.csv` for test well ids and lift the label. All three released test
wells are train wells and reconstruct at 0.0053 ft. It is inert on the graded set, and a submission
spent confirming that went the wrong way.

Three constants I quoted instead of re-deriving. Submission latency was recorded at 5 to 7
hours; it is 10 minutes, and the wrong number shaped my entire final day. Board difference noise
was modelled at 0.1613 ft when the empirical value over 11 submissions is 0.2782. Two of the three
happened to support whatever I was arguing at the time.

## The Endgame

Five submissions on the last day. I used two, both early, then stopped. The final sweep priced
everything remaining at 0.023 to 0.028 ft of expected private RMSE against a board whose difference
noise is 0.278 ft. Nothing cleared the 0.05 ft floor, so nothing shipped.

Selection ran on `E[min(RMSE_A, RMSE_B)]` over 4,000 well-cluster bootstraps of 200 wells, since
Kaggle scores each pick independently and keeps the better. Late in the evening a re-check said a
different pair was 0.008 ft better. Tested properly it was 0.011 ft with a confidence interval
through zero and a 21% chance of being worse, so I left it alone.

Private says that was a genuine coin flip and I will not claim otherwise. The alternative pair would
have scored exactly the same 7.860, because pick A won inside both pairs and the second slot
never got used. The decision did not matter, which is precisely what a confidence interval through
zero was telling me.

Holding still is what produced the result.

## Thanks

To Chris Deotte, twice over. The harness described above exists because of his
1st place writeup from NeuroGolf 2026,
which is where I took the idea from; what is above is that idea carried onto a different problem.
And his read on this competition, that the public board was noisy and a large shakeup was coming, is
what made me willing to sit on two picks and spend nothing on the final day.

To Tucker Arrants for saying plainly on the final day that CV should be trusted over the public
board, and for publishing the fold spread that proved it, and to Georgy Mamarin for measuring
the resubmission noise floor. Between them they set the bar I priced every remaining candidate
against.

To ROGII for a dataset where the hard part is a real physical problem rather than an artefact of
the scoring.

## Author

Will
wguesdon

## Citation

Will. 131st place: an LLM harness that refuses to lie to you. https://www.kaggle.com/competitions/rogii-wellbore-geology-prediction/writeups/131st-place-an-llm-harness-that-refuses-to-lie-to. 2026. Kaggle
