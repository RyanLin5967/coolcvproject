# Demo video — 60 seconds

Show the bug, show the fix, prove the number is computed. Nothing else fits.

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
| 0:19 | Stay | "Across the validation set: 59 percent versus 99." |
| 0:23 | **Verify** → **Recompute all 9 scores** | "That score isn't stored." |
| 0:27 | Rows land green | "Your browser downloaded the saved predictions, checked their hash, and recomputed average precision from scratch." |
| 0:34 | Tick **Break one detection on purpose** | "Here's how you know it's live." |
| 0:38 | Banner goes red | "Change one detection out of eighty-seven thousand — it stops matching." |
| 0:43 | Untick | "Put it back. Identical to the last bit." |
| 0:48 | Open **The chain behind these files** | "All three models start from the same weights." |
| 0:52 | Point at the init digests | "The only difference is whether a missing label counts as a negative example." |
| 0:57 | Stop | "CoverageCV. Code and data to check it are public." |

~135 words. Speak at a normal pace; don't rush to add more.

## Don't

- Don't say 95 AP or state of the art. Best construction score is 52.76.
- Don't say it beats full labels. 76.35 vs 78.04 — it closes most of the gap.
- Don't open "Best recorded result for each arm" — mixes recipes, not the claim.
- Don't claim Roboflow trained CoverageCV. Its hosted trainer takes epochs/lr only;
  a custom criterion can't be injected, so those runs are on Modal.

## If asked afterwards

- **"Why so fast?"** Scoring isn't training. Training was ~7 min/run on an L40S; scoring
  11k boxes against 241 is milliseconds. That's why it runs in a browser.
- **"Does it work on a better recipe?"** AP gain drops to +2.57 — but the ordinary model
  still misses a quarter of the pawns. Augmentation fixed the boxes, not the misses.
- **"Did the predictions come from that checkpoint?"** Recomputation can't prove that.
  It rests on the checkpoint hash, the Modal call ids, and the GPU-recorded AP matching
  all 30 runs. Audit trail, not arithmetic. Say so.
- **Numbers:** pawns 63.79 / 76.35 / 78.04 (+12.56, 3 seeds). Recall 58.6% → 99.3%.
  13 classes +11.42. Construction +1.80, one seed.
