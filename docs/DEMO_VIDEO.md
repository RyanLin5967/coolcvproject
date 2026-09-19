# Demo video plan

The video has one job: make a sceptical viewer believe the numbers. Everything else —
the UI, the story, the polish — is in service of that. A viewer who finishes thinking
"nice interface, but those figures are whatever he typed into the page" is a failure,
even if they enjoyed it.

So the spine is: **show the mechanism → show the consequence → recompute the number in
front of them → break it on purpose to prove the recomputation is live.** That last beat
is what separates this from every demo that shows a bar chart.

Target length **3:30–4:00**. Screen recording with voice-over.

---

## Which dataset, and why

**Lead with Chess pawns (`pawns-base`).** Not because the numbers are prettiest — because
it is the only dataset where a viewer can *see the bug with their own eyes* in about five
seconds.

The dataset is built by merging two sources: one annotates only white pawns, the other
only black pawns. So in the merged training data there are board images where the black
pawns are plainly visible and simply **not labelled**. Ordinary training reads those empty
regions as background and actively learns that a pawn is not a pawn. You do not have to
take anyone's word for the mechanism; it is on screen.

It also happens to carry the strongest evidence in the project:

| | Ordinary | CoverageCV | Fully labelled |
| --- | ---: | ---: | ---: |
| AP50:95, mean of 3 seeds | 63.79 | **76.35** | 78.04 |
| AP50 | 82.65–84.71 | 98.93–100 | 100 |
| Recall @ 0.25 | **0.585** | 0.983–1.000 | 1.000 |

+12.56 AP points, and **all three seeds are positive** (min +10.51, max +15.59). The
recall row is the one to say out loud: ordinary training finds about 59% of the pawns.
Coverage-aware finds essentially all of them. That is a sentence a non-specialist
understands immediately, and it is the real story — this is a recall failure, not a
general "our model is better" claim.

58 validation images and 241 boxes also make it small enough to recompute in the browser
in about 10 ms, which matters for the pacing of the verification beat.

**Use the others as breadth, late and briefly.** *All chess pieces* (13 classes, +11.42,
3 seeds) answers "does it survive more classes". *Construction safety* (5 classes, 120
images, +1.80, **one seed**) answers "is this only a toy" — real site photography, helmets
and vests. Say "one seed" out loud when you show it; it is the weakest cohort and a
technical viewer will ask.

---

## One trap left, and one that was fixed in code

**Fixed: the Benchmarks page used to overstate the result.** It selected the *minimum*
recorded control against the *maximum* recorded CoverageCV run, across different recipes,
budgets and inference settings, and reported the difference as the headline: +17.80 for
pawns, +17.37 for all pieces, **+10.32 for construction against a matched truth of +1.80**.
The rehearsal caught this. The page now renders from the verification manifest, so every
card is a matched cohort — same recipe, same update budget, every recorded seed averaged —
and every number on it is one the Verify page recomputes. The best-recorded-result table
still exists, in a disclosure labelled *"not a matched comparison"*. **Do not open that
disclosure on camera** unless you are willing to explain why those numbers are bigger;
the honest headline is the card.

**Still live: pictures and numbers come from different recipes unless you are careful.**
The Predictions gallery uses the **augmented 512px** checkpoints — seed 20260917, scoring
74.53 / 78.90 / 78.95. The headline +12.56 is the **base 384px** recipe. If you show the
gallery and then cut to +12.56, you have silently changed models.

The fix is also the best moment in the video: show the gallery, then verify the *augmented*
cohort so numbers and pictures match exactly, and only then switch to the base recipe and
say why the gap is larger there. Augmentation and the coverage fix partly solve the same
problem, so a stronger recipe shrinks the benefit from +12.56 to +2.57. Saying that
yourself is far more convincing than being caught not saying it.

## Pre-flight

```bash
python scripts/build_verification_bundle.py --check   # 30 runs reproduce exactly
node  scripts/check_verification_parity.mjs           # expect 30/30, deviation 0
python -m pytest -q                                   # expect 268 passed, 1 skipped
python scripts/build_public_demo.py
cd public-demo && python3 -m http.server 8767 --bind 127.0.0.1
```

Then the dress rehearsal, which is the one that matters — it reads every score the site
actually displays and fails if any of them cannot be recomputed, or if the Benchmarks page
and the Verify page disagree:

```bash
python scripts/practice_run.py     # expect: 24 recomputable, 0 unverifiable, cross-page OK
```

Browser: 1440×900, bookmarks bar hidden, zoom 100%, a clean profile with no extensions
visible. Open `http://127.0.0.1:8767` and **hard-reload before recording** — the Verify
button must read "Recompute all 9 scores →", not "Recompute again".

Have a second tab already logged into Roboflow on `coveragecv-chess-mvp` → version 3, and
a terminal window sized to match, so the cuts do not show you typing passwords or resizing.

---

## Shot list

**0:00–0:20 · The mechanism.** Merge demo page. Point at a board image where black pawns
are visible and unlabelled.
> "This dataset is two datasets. One of them labelled only the white pawns. The other
> labelled only the black ones. Merge them and you get images like this — the black pawns
> are right there, and nothing says they exist."

**0:20–0:45 · Why that is worse than missing data.** Stay on the coverage panel.
> "Ordinary object detection training treats every unlabelled region as background. So it
> doesn't just fail to learn these pawns. It gets told, thousands of times, that this
> thing is not a pawn. That's the problem — and it's a data problem, not a model problem."

**0:45–1:15 · The consequence.** Predictions page, chess pawns. Toggle Ordinary →
CoverageCV on the same image.
> "Same architecture, same images, same number of training steps. The only difference is
> whether the training respects what each source actually annotated. Ordinary training is
> finding about three-quarters of these pawns. Ours finds all of them."

Optional insert, 5s: Benchmarks page. The cards now read 63.79 / 76.35 / 78.04 with
"+12.56 AP points", matched and seed-averaged, and they are the same numbers the next beat
recomputes — which is why this cut is safe to make.

**1:15–1:55 · Recompute those exact numbers.** Verify page → **Chess pawns · stronger
recipe** → Recompute. Rows land on 74.53 / 78.90 / 78.95, all "identical".
> "These are the three checkpoints whose predictions you just looked at. The site isn't
> showing you a number I saved. Your browser just downloaded the saved predictions, checked
> their hash against the repository, and recalculated COCO average precision from scratch.
> No GPU, no account, no server."

**1:55–2:25 · The headline, and the honest caveat.** Switch to **Chess pawns** (base
recipe). Let the +12.56 card land.
> "On the base recipe the effect is much bigger: 63.8 to 76.3, twelve and a half points,
> and every one of three seeds moves the same direction. Coverage-aware training gets
> within 1.7 points of a model that was given every label. And to be straight about it —
> on the stronger recipe the gap was only two and a half points. Good augmentation fixes
> some of the same damage. The effect is real, and it depends on your recipe."

**2:25–2:55 · Prove the page is actually computing.** Tick **"Break one detection on
purpose."** The banner turns red; a row reads "differs by 2.12e-5". Untick — green again.
> "Here's how you know that's a live calculation and not a stored constant. This halves
> the confidence of one single detection out of eighty-seven thousand, after the hash
> check. The score moves in the fifth decimal place and the page stops matching. Turn it
> off, and it matches again, exactly."

**2:55–3:20 · Independence.** Cut to terminal.
> "The evaluator running in the browser is a separate implementation in JavaScript. This
> checks it against pycocotools, the standard scorer, on all thirty published runs —
> identical to the last bit, largest deviation zero. Two bugs turned up while getting
> there, which is rather the point of checking."

Run `pytest -q` → 267 passed.

**3:20–3:40 · Breadth.** Verify → All chess pieces, then Construction safety.
> "Thirteen classes instead of two: plus eleven and a half. And real construction site
> photography, helmets and safety vests: plus one point eight — that one is a single seed,
> so treat it as a signal, not a result."

**3:40–4:00 · Third-party record. Close.** Roboflow tab, version 3.
> "And the data isn't only on my laptop. This is the project on Roboflow — 201 training
> images, 58 validation images, the same split the scores you just watched were computed
> against, with a finished model artifact. What that proves is the data and the model exist
> independently of me. It isn't a live endpoint, and it doesn't prove the predictions came
> from that exact checkpoint — for that you'd need the weights and a GPU. Everything short
> of that, you just checked yourself."

---

## Do not say

| Do not say | The actual position |
| --- | --- |
| "95 AP" / "state of the art" | Best construction score is 52.76. Nothing here is SOTA. |
| "beats the fully-labelled model" | It does not. 76.35 vs 78.04; it *closes most of the gap*. |
| "the fully-labelled reference is the ceiling" | It is one comparison run, not a frontier model. |
| "hosted on Roboflow / live inference" | Version 3 reports no served model. Training record only. |
| "coverage.ryanlin.dev is live" | Cloudflare Pages still needs the repo connected in the dashboard. |
| "these predictions came from this checkpoint" | Not provable in-browser. Say "recorded hash and provider records". |
| "works on any dataset" | Four cohorts, small datasets, chess shares one camera domain. |
| "150 review units = 150 human hours" | Review units are not measured human time or money. |
| the numbers in "Best recorded result for each arm" | Those mix recipes. The matched card is the claim. |

## If something breaks on camera

- Verify page says the bundle is unavailable → you are serving the repo root, not
  `public-demo/`. The bundle is at `/static/verify/manifest.json`.
- A row shows a mismatch with the self-test **off** → stop recording. That is a real
  failure; run `node scripts/check_verification_parity.mjs` before doing anything else.
- Recompute finishes too fast to see → that is expected (~10 ms per run). Do not pad it;
  let the self-test beat carry the proof instead.
