# Demo video — click list

**Dataset: Chess pawns.** Two sources merged; one labelled only white pawns, the other only
black. So black pawns sit in training images unlabelled — the bug is visible on screen.
Strongest evidence too: +12.56 AP, 3/3 seeds positive, recall 0.585 → ~1.000.

## Before recording

There is no bare `python` on this machine, and `python3` is Xcode's 3.9 without the deps.
Use the repo venv for everything:

```bash
cd /Users/idide/projects/coolcvproject
.venv/bin/python scripts/build_verification_bundle.py --check   # 30 runs reproduce exactly
node scripts/check_verification_parity.mjs                      # 30/30, deviation 0
.venv/bin/python scripts/practice_run.py                        # 24 recomputable, 0 unverifiable
.venv/bin/python scripts/build_public_demo.py
```

**The server is probably already up on 8767.** Check before starting another one —
"Address already in use" just means it is running:

```bash
curl -sf -o /dev/null http://127.0.0.1:8767/ && echo "already serving" || \
  (cd public-demo && ../.venv/bin/python -m http.server 8767 --bind 127.0.0.1 &)
```

To force a fresh one: `lsof -ti tcp:8767 | xargs kill` then run the line above.
Re-run `build_public_demo.py` and hard-reload after any code change, or you record a
stale page.

Browser 1440×900, zoom 100%, bookmarks hidden. Open `http://127.0.0.1:8767`.
**Hard-reload** — the button must say "Recompute all 9 scores", not "Recompute again".

## Record

| # | Press | You should see |
|---|---|---|
| 1 | Land on **Merge demo** | A board with black pawns visible and unlabelled |
| 2 | Click through the **source toggle** | Coverage flips; same rule protects the other class |
| 3 | Sidebar → **Predictions** | Gallery, chess pawns |
| 4 | Toggle **Ordinary** ↔ **CoverageCV** on one image | Ordinary misses pawns; ours finds them |
| 5 | Sidebar → **Benchmarks** | `63.79 / 76.35 / 78.04`, **+12.56 AP points** |
| 6 | Click **Recompute them in your browser →** | Verify page, cohort *Chess pawns* |
| 7 | Click **Recompute all 9 scores →** | 9 rows "✓ identical", **+12.56** recomputed |
| 8 | Click cohort **Chess pawns · stronger recipe** → **Recompute** | `76.25 / 78.82 / 78.94`, +2.57 |
| 9 | Back to **Chess pawns**, tick **Break one detection on purpose** | Banner turns **red**, "differs by 2.12e-5" |
| 10 | Untick it | Green again, 9/9 identical |
| 11 | Cohort **All chess pieces** → Recompute | +11.42, 13 classes |
| 12 | Cohort **Construction safety** → Recompute | +1.80 — say "one seed" out loud |
| 13 | Cut to terminal: `node scripts/check_verification_parity.mjs` | `30/30 ... deviation 0` |
| 14 | `.venv/bin/python -m pytest -q` | `268 passed, 1 skipped` |
| 15 | Roboflow tab → **`coveragecv-chess-hosted`** → version 1 | A real `fine-tune` run, finished, live endpoint |

Steps 9–10 are the proof. Everything else is setup.

## The Roboflow claim — read this before step 15

There are two projects, and only one of them is a real Roboflow training run.

| Project | What it is | Safe to call it |
|---|---|---|
| `coveragecv-chess-hosted` v1 | `jobType: fine-tune`, finished, **live endpoint** `serverless.roboflow.com/coveragecv-chess-hosted/1` | **A real Roboflow training run.** But it is the *baseline* — ordinary training. |
| `coveragecv-chess-mvp` v1–3 | every training is `jobType: external-upload` | A model **uploaded** to Roboflow. Roboflow did not train it. No served endpoint. |

CoverageCV itself **cannot** be trained by hosted Roboflow — it needs a custom loss, so those
runs are on Modal. So: *"Roboflow trained the baseline on the same 201/58 split; our method
needs a custom criterion, so it trained on Modal — here are those call ids."*

Do **not** say "here is the Roboflow run that produced these numbers." It didn't.

**The stronger proof is on the Verify page anyway.** Each row now shows the Modal call id
(`fc-01M2RR9JRJ…`), the GPU (L40S), the training time, and the AP recorded *on the GPU when
training finished*. That GPU-side number matches the browser-recomputed number for all 30
runs — two independent records, one from the training machine, one from the viewer's laptop.

## Don't

- Don't open **"Best recorded result for each arm"** — mixes recipes, bigger numbers, not the claim.
- Don't say 95 AP or state of the art. Best construction is 52.76.
- Don't say it beats the fully-labelled model. 76.35 vs 78.04 — it *closes most of the gap*.
- Don't claim the Roboflow run produced the CoverageCV numbers. It trained the baseline only.
- Don't claim `coverage.ryanlin.dev` is live until Cloudflare is connected.
- Don't cut from the gallery (step 4) straight to +12.56. Gallery = augmented recipe
  (74.53/78.90/78.95); +12.56 = base recipe. Step 8 exists to make that explicit.

## If it breaks

- "Verification bundle unavailable" → you're serving the repo root, not `public-demo/`.
- A row mismatches with the self-test **off** → stop. Run the parity check before anything else.
- Recompute looks instant (~10 ms) → expected. Don't pad it; step 9 is the proof.
