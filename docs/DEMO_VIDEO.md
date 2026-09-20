# Demo video — 40 seconds

Every screen and quote below was read off the running page, not remembered.

Note the landing page is a **construction** helmet/person illustration, badged
"Illustrative example" — it is not the chess result. The measured result is chess pawns,
so the script moves from one to the other in a single clause. Don't imply the landing
image produced a number.

## Before

```bash
cd /Users/idide/projects/coolcvproject
.venv/bin/python scripts/practice_run.py        # 45 values, 0 unverifiable
curl -sf -o /dev/null http://127.0.0.1:8767/ && echo up || \
  (cd public-demo && ../.venv/bin/python -m http.server 8767 --bind 127.0.0.1 &)
```

1440×900, zoom 100%, bookmarks hidden, hard-reload.

Confirm the opening frame before rolling: heading **"A missing label is not a negative
example."**, source **A · Helmet Dataset · Labels helmet only** highlighted, mode
**Ordinary merge** selected, panel reading **"The object is there. The label is not."**

## The cut

| Time | Do | Say | On screen |
|---|---|---|---|
| 0:00 | Hold on landing | "One dataset labels helmets. Another labels people." | *Helmet Dataset / Labels helmet only · Person Dataset / Labels person only* |
| 0:05 | Hold | "Merge them, and a person who was never labelled looks like background. Training pushes the model away from a correct answer." | *"The object is there. The label is not." · Unmatched person prediction → **Apply a negative signal*** |
| 0:12 | Click **CoverageCV merge** | "We leave the unreviewed class alone instead of training against it." | *"Unknown stays unknown." · → **Ignore the negative signal*** |
| 0:17 | **Predictions** → tab **Chess pawns** | "Measured on a set built the same way — one source labelled white pawns, the other black." | *Image 1 of 6 · 7 labeled objects* |
| 0:23 | Point at the three cards | "Ordinary training finds three of these seven. Ours finds six. With every label, seven." | *3 · 6 · 7 predictions · 63.79 / 76.35 / 78.04* |
| 0:29 | **Verify** → **Recompute all 9 scores →** | "Those scores aren't stored." | button reads *Recompute all 9 scores →* |
| 0:33 | Rows land green | "The browser fetches the saved predictions, checks their hash, and recomputes average precision from eighty-seven thousand detections." | *9 of 9 reproduced exactly · difference 0* |
| 0:39 | Stop | "It's client-side — open it and run it yourself." | |

~105 words. If you run long, cut the 0:05 second sentence, not the numbers.

## Don't

- Don't call the landing image a result. It is badged "Illustrative example".
- Don't say 95 AP or state of the art. Best construction score is 52.76.
- Don't say it beats full labels. 76.35 vs 78.04 — it closes most of the gap.
- Don't open "Best recorded result for each arm" — mixes recipes, not the claim.
- Don't claim Roboflow trained CoverageCV. Its hosted trainer takes epochs/lr only, so a
  custom criterion can't be injected. Roboflow proves a model exists, not that this page
  computes anything.

## If asked afterwards

- **"How do I know it's not hardcoded?"** Tick "Break one detection on purpose" on Verify:
  change one detection of 87,020 and the score stops matching; untick and it is identical
  again. Left on the page for this question.
- **"Why so fast?"** Scoring isn't training. Training was ~7 min/run on an L40S; scoring
  11k boxes against 241 is milliseconds. That's why it runs in a browser.
- **"Is the comparison fair?"** All three arms of a seed start from one identical weight
  digest, shown under "The chain behind these files". Only the loss differs.
- **"Does it hold on a better recipe?"** AP gain drops to +2.57, but the ordinary model
  still misses a quarter of the pawns. Augmentation fixed the boxes, not the misses.
- **"Did the predictions come from that checkpoint?"** Recomputation can't prove it. That
  rests on the checkpoint hash, Modal call ids, and the GPU-recorded AP matching all 30
  runs. Audit trail, not arithmetic.
- **Numbers:** pawns 63.79 / 76.35 / 78.04, +12.56 over 3 seeds. Recall 58.6% → 99.3%.
  13 classes +11.42. Construction +1.80, one seed.
