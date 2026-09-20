# Demo video — ~40 seconds

You're talking over this, so below are **beats, not lines**. Say them however they come
out. The only things that need to be exact are the numbers.

Landing, Predictions and Verify all use the same chess-pawn dataset now, so you never
change subject.

## Before

```bash
cd /Users/idide/projects/coolcvproject
.venv/bin/python scripts/practice_run.py        # 45 values, 0 unverifiable
curl -sf -o /dev/null http://127.0.0.1:8767/ && echo up || \
  (cd public-demo && ../.venv/bin/python -m http.server 8767 --bind 127.0.0.1 &)
```

1440×900, zoom 100%, bookmarks hidden, hard-reload.

Opening frame should read **"A missing label is not a negative example."**, with
**A · White-Pawn Dataset** ticked, **Ordinary merge** selected, and the board showing solid
blue boxes on white pawns and dashed red ones on black pawns.

## Beats

**0:00 — hold on the board**
> "So, two datasets. This one labelled the white pawns — the blue boxes. The other one only
> did the black ones."

**0:06 — point at the dashed boxes**
> "Which means these black pawns? Nobody labelled them. And normal training reads that as
> 'not a pawn' — so it actually learns to ignore them."

**0:14 — click CoverageCV merge**
> "So we just leave that class alone. Nobody checked it, don't train against it."

**0:19 — Predictions → Chess pawns tab**
> "Same model, same images, same training budget. Normal training finds three of these seven
> pawns. Ours finds six."

**0:27 — point at the reference card**
> "With every label, seven."

**0:30 — Verify → Recompute all 9 scores**
> "And the scores aren't just typed into the page — it pulls the saved predictions, checks
> the hash, and recomputes the whole thing in your browser."

**0:38 — stop**
> "All client-side. Go try it yourself."

~115 words. If you run long, drop the 0:27 beat.

## Numbers that must be right

- **3 / 6 / 7** predictions on image 1, out of 7 labelled objects
- **63.79 / 76.35 / 78.04** AP50:95, mean of 3 seeds, +12.56
- Recall across the set: **58.6% → 99.3%**

## Don't

- Don't call the board a result — it's badged "Illustrative example".
- Don't say 95 AP or state of the art. Best construction score is 52.76.
- Don't say it beats full labels. 76.35 vs 78.04 — it closes most of the gap.
- Don't open "Best recorded result for each arm" — mixes recipes, not the claim.
- Don't say Roboflow trained CoverageCV. Its hosted trainer takes epochs/lr only, so a
  custom loss can't go in. Roboflow shows a model exists, not that this page computes.

## If someone asks after

- **"How do I know it's not hardcoded?"** Tick "Break one detection on purpose" on Verify —
  change one detection out of 87,020 and the score stops matching. Untick, identical again.
- **"Why is it so fast?"** Scoring isn't training. Training was ~7 min/run on an L40S;
  scoring 11k boxes against 241 is milliseconds. That's why it runs in a browser.
- **"Is the comparison fair?"** All three models in a seed start from one identical weight
  digest — it's under "The chain behind these files". Only the loss differs.
- **"Does it hold with better training?"** AP gain drops to +2.57, but the normal model
  still misses a quarter of the pawns. Augmentation fixed the boxes, not the misses.
- **"Did the predictions come from that checkpoint?"** Recomputing can't prove that. It
  rests on the checkpoint hash, Modal call ids, and the GPU-recorded AP matching all 30
  runs. Audit trail, not arithmetic — say so.
- Other datasets: 13 chess classes +11.42 (3 seeds). Construction +1.80 (one seed).
