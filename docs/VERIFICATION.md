# Verifying the published scores

Until now every number on the site was a stored constant. A visitor could read
`docs/RESULTS.json`, but nothing on the page let them check that a score was the score
those predictions actually earn. This document records what was built to close that gap,
and — more importantly — the boundary of what it establishes.

## What the site now does

The **Verify** page ships the saved validation predictions for 30 recorded runs, plus the
reference labels they were scored against. The visitor's browser hashes those files,
compares the hash to a digest committed in this repository, decodes the predictions and
recomputes COCO AP50:95 from scratch. The number it displays is the number it just
calculated. No account, no GPU, no network call beyond fetching the static files.

Four cohorts are published, each one a matched comparison — same recipe, same update
budget, one row per seed:

| Cohort | Dataset | Runs | Recomputed effect (aware − ordinary) |
| --- | --- | ---: | --- |
| `pawns-base` | Chess pawns, 2 classes | 9 | **+12.56 AP**, mean of 3 seeds |
| `pawns-augmented` | Chess pawns, stronger recipe | 9 | +2.57 AP, mean of 3 seeds |
| `all-pieces-base` | All 13 chess pieces | 9 | +11.42 AP, mean of 3 seeds |
| `construction` | Construction safety, 5 classes | 3 | +1.80 AP, one seed |

## What it establishes, and what it does not

**Established.** The published score is the score these predictions earn against these
labels; the arithmetic is correct and was not tuned. The bytes the browser scored are the
bytes this repository committed, because the browser checks the digest before scoring.
The evaluator is a JavaScript reimplementation, and `scripts/check_verification_parity.mjs`
confirms it reproduces the Python result **bit for bit on all 30 runs** — so agreement is
not two copies of one bug.

**Not established.** That these predictions came from the checkpoint named in the
manifest. Proving that requires the weights and a GPU to re-run inference; what stands
behind it is the recorded checkpoint SHA-256 and the provider's own run records, which is
an audit trail, not something the page can recompute. The page says this in plain words
rather than leaving it implied. Nothing here speaks to performance beyond these validation
images, and these are single training runs per seed on small datasets that share a
camera domain.

## Why the payload is small enough to ship

Saved predictions are ~4.5 MB of JSON per run. Two reductions bring that to ~250 KB of
binary, and both are *asserted lossless at build time* against the published metrics
rather than assumed:

1. **Four fields only.** `pycocotools`' `COCO.loadRes` overwrites `area`, `id` and
   `iscrowd` and derives `segmentation` from the box, so only `image_id`, `category_id`,
   `bbox` and `score` can affect a bbox score.
2. **Top 100 per image and class.** This is the `maxDets=100` cap COCOeval already
   applies, so detections past it cannot enter the result.
3. **float32.** The precision the detector emitted. About 2% of the stored float64 values
   are not exactly representable in float32, so this one is *not* obviously safe — it is
   kept only because re-scoring after quantisation reproduced all 13 published metrics
   exactly for every run. `build_verification_bundle.py` refuses to export a run whose
   reduced predictions do not reproduce its published metrics.

## The two bugs this found in the browser evaluator

Both were caught by parity testing against Python, and both would have shipped a
confident, wrong number:

- **The precision curve past maximum recall is zero, not absent.** `pycocotools`
  initialises each curve to zeros and overwrites only the reachable part, and those zeros
  are averaged in. Treating them as missing inflated AP badly at strict IoU thresholds —
  construction `no-helmet` AP read 0.76 instead of 0.34.
- **`np.mean` sums pairwise, not left to right.** Naive accumulation differed from the
  published value in the last two or three bits. The evaluator now mirrors numpy's
  `pairwise_sum_DOUBLE`, which is what makes bit-exact agreement possible.

## The Benchmarks page now renders from this bundle

A dress rehearsal of the site (`scripts/practice_run.py`) found that the Benchmarks page
disagreed with the verifiable truth, badly. It selected each arm's *best* recorded result
across different recipes, budgets and inference settings, so its headline read +17.80 /
+17.37 / **+10.32** where the matched cohorts give +12.56 / +11.42 / **+1.80**. Eight of the
eighteen values it displayed could not be recomputed at all.

The cards are now derived from `manifest.json`, the same file the Verify page scores, so the
two pages cannot drift: 24 of 24 displayed values are recomputable and the rehearsal fails
if any headline stops matching. The extrema are kept in a disclosure that states plainly
that they are not a matched comparison.

## How to check it yourself

```bash
python scripts/build_verification_bundle.py --check   # rebuild and compare to committed bytes
node scripts/check_verification_parity.mjs            # JS vs Python, expect 30/30, deviation 0
python -m pytest tests/test_verification_bundle.py    # digests, formats, publication parity
python scripts/qa_verify_page.py                      # drive the real page at 1440px and 390px
python scripts/practice_run.py                        # every displayed score, and page-to-page agreement
```

The page also carries its own negative control: **“Break one detection on purpose”**
halves a single detection's confidence after the hash check. One altered detection out of
87,020 moves the pawns score by ~2e-5 and turns the verdict red, which is how a viewer can
tell the page recalculates the score rather than printing it. `scripts/qa_verify_page.py`
asserts that this control goes red and returns to green, so a green page is a tested
result rather than a hopeful one.
