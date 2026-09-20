# Demo video — 40 seconds

Show the bug, show the fix, show the score being computed rather than printed. Stop.

## Before

```bash
cd /Users/idide/projects/coolcvproject
.venv/bin/python scripts/practice_run.py        # 45 values, 0 unverifiable
curl -sf -o /dev/null http://127.0.0.1:8767/ && echo up || \
  (cd public-demo && ../.venv/bin/python -m http.server 8767 --bind 127.0.0.1 &)
```

1440×900, zoom 100%, bookmarks hidden, hard-reload. Verify button must read
"Recompute all 9 scores". Predictions opens on image 1 — don't click through images.

## The cut

| Time | Screen | Say |
|---|---|---|
| 0:00 | Landing. Board with unlabelled black pawns | "Two datasets merged. One labelled only white pawns, the other only black." |
| 0:05 | Stay | "So these black pawns are unlabelled, and ordinary training reads them as background." |
| 0:10 | **Predictions** (pawns, image 1) | "Same model, same images, same update budget." |
| 0:14 | Point at the three counts | "Ordinary training finds three of seven pawns. Ours finds six." |
| 0:19 | Stay | "Across the validation set, 59 percent of objects versus 99." |
| 0:24 | **Verify** → **Recompute all 9 scores** | "These scores aren't stored." |
| 0:28 | Rows land green | "The page downloads the saved predictions, checks their hash, and recomputes average precision in the browser." |
| 0:34 | Open **How that number is built** | "Ten values, one per overlap threshold, averaging to the published score." |
| 0:39 | Stop | "It's client-side. Open the page and run it yourself." |

~95 words. Let it breathe; don't add more.

## Don't

- Don't say 95 AP or state of the art. Best construction score is 52.76.
- Don't say it beats full labels. 76.35 vs 78.04 — it closes most of the gap.
- Don't open "Best recorded result for each arm" — mixes recipes, not the claim.
- Don't claim Roboflow trained CoverageCV. Its hosted trainer takes epochs/lr only;
  a custom criterion can't be injected, so those runs are on Modal. Roboflow proves a
  model exists, not that this page computes anything — don't use it as that.

## If asked afterwards

- **"How do I know it's not hardcoded?"** Tick "Break one detection on purpose" on the
  Verify page: change one detection of eighty-seven thousand and the score stops matching;
  untick and it's identical again. Left on the page for exactly this question.
- **"Why so fast?"** Scoring isn't training. Training was ~7 min/run on an L40S; scoring
  11k boxes against 241 is milliseconds. That's why it runs in a browser.
- **"Is the comparison fair?"** All three arms of a seed start from one identical weight
  digest — shown under "The chain behind these files". Only the loss differs.
- **"Does it hold on a better recipe?"** AP gain drops to +2.57, but the ordinary model
  still misses a quarter of the pawns. Augmentation fixed the boxes, not the misses.
- **"Did the predictions come from that checkpoint?"** Recomputation can't prove that. It
  rests on the checkpoint hash, Modal call ids, and the GPU-recorded AP matching all 30
  runs. Audit trail, not arithmetic. Say so.
- **Numbers:** pawns 63.79 / 76.35 / 78.04 (+12.56, 3 seeds). Recall 58.6% → 99.3%.
  13 classes +11.42. Construction +1.80, one seed.
