# Demo video — ~40 seconds

You're talking over this. These are beats, not a script — say them however they come out.
Only the numbers need to be exact.

Landing, Predictions and Verify all open on the same chess-pawn dataset now, so you never
switch subject and never need to pick a tab.

## Before

```bash
cd /Users/idide/projects/coolcvproject
.venv/bin/python scripts/practice_run.py        # 45 values, 0 unverifiable
curl -sf -o /dev/null http://127.0.0.1:8767/ && echo up || \
  (cd public-demo && ../.venv/bin/python -m http.server 8767 --bind 127.0.0.1 &)
```

1440×900, zoom 100%, bookmarks hidden, hard-reload.

First frame: **"A missing label is not a negative example."**, **A · Black-Pawn Dataset**
ticked, **Ordinary merge** selected, board showing solid blue boxes on the black pawns and
dashed red ones on the white.

## Beats

**0:00 — on the board**
> "Okay so, two chess datasets. This one only labelled the black pawns. The other one only
> did the white ones."

**0:06 — point at the dashed boxes**
> "So these white pawns here — nobody ever labelled them. And the model doesn't know that.
> It just sees an object with no label and goes, cool, that's background. So it learns to
> not find them."

**0:15 — click CoverageCV merge**
> "So all we do is, if a dataset never checked for white pawns, we don't let it train
> against white pawns. That's basically it."

**0:21 — click Predictions** (opens on Chess pawns)
> "And here's what that does. Same model, same images, same amount of training. Normal
> training finds three of the seven pawns — and look, it's missed every single white one."

**0:29 — point at the middle and right cards**
> "Ours finds six. With every label, seven."

**0:32 — Verify → Recompute all 9 scores**
> "Oh and these numbers aren't hardcoded. It's pulling the actual saved predictions and
> recomputing the score right now, in the browser."

**0:39 — stop**
> "It's all client-side, so go check it yourself."

~130 words. Running long? Drop the second half of 0:06.

## Numbers that have to be right

- Image 1 reads **found 3 of 7 · 0 false**, then **6 of 7**, then **7 of 7**
- Those 7 pawns are 4 white and 3 black. Ordinary training finds the 3 black and
  **none** of the white; ours gets all 3 black plus 3 of the 4 white
- **63.79 / 76.35 / 78.04** AP50:95, mean of 3 seeds, +12.56
- Recall across the set: **58.6% → 99.3%**

## Don't

- Don't call the board a result — it's badged "Illustrative example".
- Don't say 95 AP or state of the art. Best construction score is 52.76.
- Don't say it beats full labels. 76.35 vs 78.04 — it closes most of the gap.
- Don't open "Best recorded result for each arm" — mixes recipes, not the claim.
- Don't say Roboflow trained CoverageCV. Its hosted trainer only takes epochs/lr, so a
  custom loss can't go in. Roboflow shows a model exists, not that this page computes.

## If someone asks "what's a prediction?"

The detector looks at the image and outputs boxes — each one a location, a class, and a
confidence. A prediction is one guessed box: *"I think there's a white pawn here, 93%
sure."* The card counts the ones above the confidence slider, and says how many of them
actually landed on a real pawn (50% overlap, right class) versus how many were wrong.

So "found 3 of 7 · 0 false" means: seven pawns in the picture, the model correctly found
three, and didn't invent any.

## If someone asks after

- **"How do I know it's not hardcoded?"** Tick "Break one detection on purpose" on Verify —
  change one detection out of 87,020 and it stops matching. Untick, identical again.
- **"Why's it so fast?"** Scoring isn't training. Training was ~7 min a run on an L40S.
  Scoring 11k boxes against 241 is milliseconds — that's why it runs in a browser.
- **"Is that a fair comparison?"** All three models in a seed start from one identical
  weight digest, under "The chain behind these files". Only the loss differs.
- **"Does it hold with better training?"** AP gain drops to +2.57, but the normal model
  still misses a quarter of the pawns. Augmentation fixed the boxes, not the misses.
- **"Did those predictions come from that checkpoint?"** Recomputing can't prove it. That
  one rests on the checkpoint hash, Modal call ids, and the GPU-recorded AP matching all
  30 runs. It's an audit trail, not arithmetic — just say that.
- Other datasets: 13 chess classes +11.42 (3 seeds). Construction +1.80 (one seed).

## Note

The old `public-demo/media/walkthrough.mp4` still shows the construction merge story and
its captions still say helmets, so the "Watch the walkthrough" button is hidden. Drop your
new recording in at that path with a matching `.vtt`, then set
`WALKTHROUGH_MATCHES_PAGE = true` in `merge-story.js` and rebuild.
