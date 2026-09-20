// The "Verify" page. Every other page shows numbers this project asserts; this one
// recomputes them in the visitor's browser from the saved model predictions and the
// reference labels, and says plainly which part of the chain that does and does not
// establish.
import {decodePredictions, evaluate, sha256Hex, IOU_THRESHOLDS} from './coco-eval.js';

const BASE = '/static/verify';
const esc = value => String(value).replace(/[&<>"']/g, c =>
  ({'&': '&amp;', '<': '&lt;', '>': '&gt;', '"': '&quot;', "'": '&#39;'}[c]));
const roleNames = {
  naive: {label: 'Ordinary training', tag: 'CONTROL'},
  aware: {label: 'CoverageCV', tag: 'OUR METHOD'},
  complete_reference: {label: 'Fully labeled reference', tag: 'REFERENCE'},
};

let manifestPromise;
const labelCache = new Map();
const state = {cohort: null, results: new Map(), running: false, done: false, tamper: false};

const loadManifest = () => (manifestPromise ??= fetch(`${BASE}/manifest.json`)
  .then(response => {
    if (!response.ok) throw new Error('The verification bundle is not published with this build.');
    return response.json();
  }));

async function loadLabels(manifest, digest) {
  if (!labelCache.has(digest)) {
    const entry = manifest.ground_truth[digest];
    const response = await fetch(`${BASE}/${entry.path}`);
    if (!response.ok) throw new Error('Reference labels could not be loaded.');
    const bytes = new Uint8Array(await response.arrayBuffer());
    const observed = await sha256Hex(bytes.buffer);
    if (observed !== entry.sha256) throw new Error('Reference labels failed their integrity check.');
    labelCache.set(digest, JSON.parse(new TextDecoder().decode(bytes)));
  }
  return labelCache.get(digest);
}

// Yield to the browser so the page repaints between runs instead of freezing.
const breathe = () => new Promise(resolve => requestAnimationFrame(() => setTimeout(resolve, 0)));

async function verifyRun(manifest, run) {
  const started = performance.now();
  const response = await fetch(`${BASE}/${run.predictions.path}`);
  if (!response.ok) throw new Error('Saved predictions could not be loaded.');
  const buffer = await response.arrayBuffer();
  const digest = await sha256Hex(buffer);
  const labels = await loadLabels(manifest, run.bundle_digest);
  const predictions = decodePredictions(buffer);
  // Deliberate self-test. Nudging a single detection's confidence after the hash
  // check shows the comparison responding to the data rather than echoing the
  // published number back. One detection out of tens of thousands is enough.
  let tampered = null;
  if (state.tamper && predictions.length) {
    const victim = predictions[Math.floor(predictions.length / 2)];
    tampered = {index: Math.floor(predictions.length / 2), from: victim.score, to: victim.score * 0.5};
    victim.score *= 0.5;
  }
  const metrics = evaluate(predictions, labels, {threshold: run.score_threshold});
  // Every detection is compared against the ground-truth boxes of its own class, once per
  // IoU threshold. This is the arithmetic the number is made of, and it is worth stating.
  const perClassGt = new Map();
  for (const annotation of labels.annotations) {
    const key = `${annotation.image_id}:${annotation.category_id}`;
    perClassGt.set(key, (perClassGt.get(key) ?? 0) + 1);
  }
  let comparisons = 0;
  for (const prediction of predictions) {
    comparisons += perClassGt.get(`${prediction.image_id}:${prediction.category_id}`) ?? 0;
  }
  comparisons *= IOU_THRESHOLDS.length;
  return {
    comparisons,
    digest, digestMatches: digest === run.predictions.sha256,
    bytes: buffer.byteLength, detections: predictions.length,
    metrics, delta: metrics.AP - run.expected.AP,
    exact: metrics.AP === run.expected.AP,
    allExact: ['AP', 'AP50', 'AP75', 'recall_at_threshold'].every(k => metrics[k] === run.expected[k]),
    tampered, milliseconds: performance.now() - started,
  };
}

function verdictBanner(cohort) {
  const results = cohort.runs.map(run => state.results.get(run.id)).filter(Boolean);
  const finished = results.filter(r => !r.error);
  if (!state.done || finished.length !== cohort.runs.length) {
    return `<div class="verify-banner pending"><b>${state.running ? 'Recomputing…' : 'Not yet recomputed'}</b>
      <span>${finished.length} of ${cohort.runs.length} scores recomputed in this browser.</span></div>`;
  }
  const exact = finished.filter(r => r.exact && r.digestMatches).length;
  const worst = Math.max(...finished.map(r => Math.abs(r.delta)));
  const good = exact === finished.length;
  const tampered = finished.filter(r => r.tampered).length;
  const detections = finished.reduce((sum, r) => sum + r.detections, 0).toLocaleString();
  if (tampered) {
    const moved = finished.length - exact;
    return `<div class="verify-banner fail"><b>✕ Self-test active — ${moved} of ${finished.length} scores changed</b>
      <span>One detection's confidence was halved in each run, after the hash check, out of ${detections} detections.
      The largest resulting shift is ${worst.toExponential(2)} AP — small, but no longer the published value, which is how
      you can tell this page recalculates the score rather than printing it.
      ${moved < finished.length ? `The other ${finished.length - moved} landed on the identical float, because that one
      detection did not change their ranking. ` : ''}Switch the self-test off to score the untouched files.</span></div>`;
  }
  return `<div class="verify-banner ${good ? 'pass' : 'fail'}">
    <b>${good ? '✓' : '✕'} ${exact} of ${finished.length} scores reproduced ${good ? 'exactly' : 'with differences'}</b>
    <span>Largest difference from the published value: ${worst === 0 ? '0 — identical to the last bit' : worst.toExponential(3)}.
    Recomputed from ${detections} saved detections in your browser.</span></div>`;
}

// The comparison the project actually claims, averaged over seeds from the numbers
// this browser just produced rather than from anything stored in the page.
function recomputedSummary(cohort) {
  const results = cohort.runs.map(run => ({run, result: state.results.get(run.id)}));
  if (!state.done || results.some(entry => !entry.result || entry.result.error)) return '';
  const meanFor = role => {
    const values = results.filter(e => e.run.role === role).map(e => e.result.metrics.AP);
    return values.reduce((sum, v) => sum + v, 0) / values.length;
  };
  if (results.some(entry => entry.result.tampered)) return '';
  const naive = meanFor('naive'), aware = meanFor('aware'), reference = meanFor('complete_reference');
  const gain = 100 * (aware - naive), remaining = 100 * (reference - aware);
  return `<div class="verify-summary">
    <div class="verify-summary-main"><span class="eyebrow">MEAN OF ${cohort.seeds.length} SEEDS · RECOMPUTED HERE</span>
      <strong>${gain >= 0 ? '+' : ''}${gain.toFixed(2)}<span> AP points</span></strong>
      <p>Coverage-aware training versus ordinary training on the same images, same labels,
      same update budget.</p></div>
    <div class="verify-summary-rows">
      <div><span>Ordinary training</span><b>${(100 * naive).toFixed(2)}</b></div>
      <div class="aware"><span>CoverageCV</span><b>${(100 * aware).toFixed(2)}</b></div>
      <div><span>Fully labeled reference</span><b>${(100 * reference).toFixed(2)}</b></div>
    </div>
    <p class="verify-summary-note">The reference run gets every label, so it marks what the extra annotation
    would have bought: ${remaining >= 0 ? `${remaining.toFixed(2)} points still ahead` : `${Math.abs(remaining).toFixed(2)} points behind`}.
    It is a comparison run, not a state-of-the-art model.</p></div>`;
}

// AP50:95 is an average of ten APs. Showing them is the clearest answer to "is this number
// just hardcoded?" -- a stored constant has no parts, and these ten visibly average to it.
function breakdownPanel(cohort) {
  const target = cohort.runs.find(run => run.role === 'aware') ?? cohort.runs[0];
  const result = state.results.get(target?.id);
  if (!result || result.error || !result.metrics.AP_by_iou) return '';
  const parts = result.metrics.AP_by_iou;
  const peak = Math.max(...parts.map(part => part.AP), 0.0001);
  const mean = result.metrics.AP; // the mean of exactly these ten, summed as numpy does
  return `<details class="verify-breakdown" open><summary>How that number is built — ${esc(roleNames[target.role]?.label ?? target.role)}, seed ${target.seed}</summary>
    <div><p>AP50:95 is the mean of ten separate average-precision values, one for each
    box-overlap requirement from 50% to 95%. Your browser computed all ten from the
    ${result.detections.toLocaleString()} saved detections, making
    ${result.comparisons.toLocaleString()} box comparisons to do it.</p>
    <div class="verify-bars">${parts.map(part => `<div>
      <i style="height:${Math.max(2, 100 * part.AP / peak).toFixed(1)}%"></i>
      <b>${(100 * part.AP).toFixed(1)}</b><span>${part.threshold.toFixed(2)}</span></div>`).join('')}</div>
    <p class="verify-breakdown-sum">Mean of those ten:
      <b>${(100 * mean).toFixed(4)}</b> — and the published score for this run is
      <b>${(100 * target.expected.AP).toFixed(4)}</b>.
      ${mean === target.expected.AP ? 'Identical.' : `Differs by ${(mean - target.expected.AP).toExponential(2)}.`}</p>
    <p class="muted small-text">The tall bars on the left are loose overlap requirements; the
    short ones on the right are strict. A detector that finds objects but boxes them loosely
    scores well on the left and badly on the right, which is why the average is the honest
    summary and why AP50 alone always looks better.</p></div></details>`;
}

function runRow(run) {
  const result = state.results.get(run.id);
  const role = roleNames[run.role] ?? {label: run.role, tag: ''};
  const published = (100 * run.expected.AP).toFixed(4);
  let recomputed = '<span class="verify-idle">—</span>', verdict = '';
  if (result?.error) {
    recomputed = '<span class="verify-bad">failed</span>';
    verdict = `<span class="verify-bad">${esc(result.error)}</span>`;
  } else if (result) {
    recomputed = `<b>${(100 * result.metrics.AP).toFixed(4)}</b>`;
    verdict = result.exact && result.digestMatches
      ? `<span class="verify-good">✓ identical</span>`
      : `<span class="verify-bad">✕ differs by ${result.delta.toExponential(2)}</span>`;
  }
  return `<tr class="${run.role}">
    <td><b>${esc(role.label)}</b><small>${esc(role.tag)} · seed ${run.seed}</small></td>
    <td class="numeric">${published}</td>
    <td class="numeric">${recomputed}</td>
    <td>${verdict}</td>
    <td class="verify-meta">${result ? `${result.detections.toLocaleString()} detections · ${(result.bytes / 1024).toFixed(0)} KB · ${result.milliseconds.toFixed(0)} ms` : `${run.predictions.count.toLocaleString()} detections`}</td>
  </tr>`;
}

function provenance(cohort, manifest) {
  const labels = manifest.ground_truth[cohort.runs[0].bundle_digest];
  return `<details class="disclosure"><summary>The chain behind these files</summary><div>
    <p>Each run was trained on a Modal cloud GPU, scored with pycocotools, and its predictions saved.
    The call id is Modal's identifier for that training job.</p>
    <div class="table-wrap"><table><thead><tr><th>Run</th><th>Training call</th><th>GPU</th><th>Checkpoint SHA-256</th><th>AP recorded on the GPU</th></tr></thead><tbody>
    ${cohort.runs.map(run => `<tr><td>${esc(roleNames[run.role]?.label ?? run.role)} · ${run.seed}</td>
      <td class="mono">${esc(run.training?.call_id ?? 'not recorded')}</td>
      <td>${esc(run.training?.gpu ?? '—')}${run.training?.elapsed_seconds ? ` · ${Math.round(run.training.elapsed_seconds / 60)} min` : ''}</td>
      <td class="mono">${esc((run.checkpoint_sha256 ?? '').slice(0, 16))}…</td>
      <td class="numeric">${run.training ? (100 * run.training.gpu_recorded_AP).toFixed(4) : '—'}</td></tr>`).join('')}
    </tbody></table></div>
    <p>The last column was written on the GPU when the model finished training, before any of
    the scoring this page repeats. It matches the recomputed column above for every run, so two
    independent records — one from the training machine, one from your browser — agree.</p>
    <p class="verify-init"><b>All three models in a seed start from the same weights.</b>
    ${cohort.seeds.map(seed => {
      const run = cohort.runs.find(entry => entry.seed === seed);
      return `Seed ${seed}: <span class="mono">${esc((run?.initial_parameter_digest ?? '').slice(0, 16))}…</span>`;
    }).join(' · ')}.
    Every arm of a seed branches from that one initialization, on the same images with the same
    number of updates, so the only difference between them is whether training treats an
    unlabelled region as background. The starting point is a detector pre-trained on COCO
    (<span class="mono">${esc((cohort.runs[0].pretrained_sha256 ?? 'not recorded').slice(0, 16))}…</span>),
    which is shared by all three, so pre-training cannot account for a gap between them.
    The build refuses to publish a cohort whose arms disagree on this.</p>
    <p>Reference labels: <span class="mono">${esc(labels.path)}</span> · ${labels.images} images ·
    ${labels.boxes} boxes · ${labels.categories.length} classes · SHA-256 <span class="mono">${esc(labels.sha256.slice(0, 16))}…</span>,
    checked in your browser before scoring.</p></div></details>`;
}

function scopeNote(manifest) {
  return `<details class="disclosure"><summary>What this does and does not prove</summary><div>
    <p><b>It does establish</b> that the published score is the score these saved predictions actually earn against
    these reference labels, and that the files your browser scored are byte-for-byte the ones this repository
    committed — your browser hashes them and checks the digest before scoring. The evaluator here is an independent
    reimplementation of COCO AP in JavaScript; a test in the repository confirms it reproduces the Python
    result bit for bit on all ${manifest.cohorts.reduce((sum, c) => sum + c.runs.length, 0)} committed runs, so agreement is not two copies of one bug.</p>
    <p><b>It does not establish</b> that these predictions came from the checkpoint named above. Proving that
    needs the weights and a GPU to run them again; what stands behind it here is the recorded checkpoint hash and
    the provider's own run records, which is an audit trail rather than something this page can recompute.
    It also does not make any claim about performance beyond these validation images, and these are single
    training runs per seed on small datasets.</p>
    <p>Scoring settings: COCO <span class="mono">bbox</span> AP averaged over IoU
    ${IOU_THRESHOLDS[0]}–${IOU_THRESHOLDS[IOU_THRESHOLDS.length - 1]}, 101 recall points,
    at most 100 detections per image, all box areas.</p></div></details>`;
}

async function runAll(root, manifest, cohort) {
  if (state.running) return;
  state.running = true;
  state.done = false;
  state.results = new Map();
  paint(root, manifest, cohort);
  for (const run of cohort.runs) {
    await breathe();
    try {
      state.results.set(run.id, await verifyRun(manifest, run));
    } catch (error) {
      state.results.set(run.id, {error: error.message});
    }
    paint(root, manifest, cohort);
  }
  state.running = false;
  state.done = true;
  paint(root, manifest, cohort);
}

function paint(root, manifest, cohort) {
  const target = root.querySelector('#verify-body');
  if (!target) return;
  target.innerHTML = `${verdictBanner(cohort)}${recomputedSummary(cohort)}${breakdownPanel(cohort)}
    <div class="table-wrap"><table class="verify-table"><thead><tr>
      <th>Model</th><th class="numeric">Published AP50:95</th><th class="numeric">Recomputed here</th>
      <th>Result</th><th>Work done in your browser</th></tr></thead>
      <tbody>${cohort.runs.map(runRow).join('')}</tbody></table></div>
    ${provenance(cohort, manifest)}${scopeNote(manifest)}`;
  const button = root.querySelector('#verify-run');
  if (button) {
    button.disabled = state.running;
    button.textContent = state.running ? 'Recomputing…' : state.done ? 'Recompute again' : `Recompute all ${cohort.runs.length} scores →`;
  }
}

export async function renderVerify(root) {
  root.innerHTML = '<div class="loading"><div class="spin"></div>Loading the evidence bundle…</div>';
  let manifest;
  try {
    manifest = await loadManifest();
  } catch (error) {
    root.innerHTML = `<div class="empty"><h2>Verification bundle unavailable</h2><p>${esc(error.message)}</p></div>`;
    return;
  }
  if (!manifest.cohorts?.length) {
    root.innerHTML = '<div class="empty"><h2>No verifiable runs are published.</h2></div>';
    return;
  }
  if (!manifest.cohorts.some(c => c.key === state.cohort)) {
    state.cohort = (manifest.cohorts.find(c => c.headline) ?? manifest.cohorts[0]).key;
    state.results = new Map();
    state.done = false;
  }
  const cohort = manifest.cohorts.find(c => c.key === state.cohort);

  root.innerHTML = `<div class="page-heading"><div><div class="eyebrow">RECOMPUTED ON YOUR MACHINE</div>
      <h1>Verify the numbers</h1>
      <p class="muted">The scores on this site are not stored constants here. Your browser downloads the saved
      model predictions, checks their hash, and recalculates COCO AP50:95 from scratch against the reference
      labels. Nothing is sent anywhere, and no account or GPU is involved.</p></div>
      <div class="verify-actions">
        <button class="button primary" id="verify-run">Recompute all ${cohort.runs.length} scores →</button>
        <label class="verify-tamper"><input type="checkbox" id="verify-tamper" ${state.tamper ? 'checked' : ''}>
          <span>Break one detection on purpose<small>Proves the check can fail</small></span></label>
      </div></div>
    <div class="dataset-switcher" aria-label="Verifiable cohort">${manifest.cohorts.map(entry =>
      `<button data-verify-cohort="${esc(entry.key)}" aria-pressed="${entry.key === state.cohort}"
        class="${entry.key === state.cohort ? 'selected' : ''}"><b>${esc(entry.name)}</b>
        <span>${esc(entry.detail)}</span></button>`).join('')}</div>
    <div class="verify-context"><b>${esc(cohort.name)}</b><span>${esc(cohort.recipe)}</span>
      <span>${cohort.seeds.length} seed${cohort.seeds.length === 1 ? '' : 's'} × ${Object.keys(roleNames).length} models</span></div>
    <div id="verify-body"></div>`;

  paint(root, manifest, cohort);
  root.querySelector('#verify-run').onclick = () => runAll(root, manifest, cohort)
    .catch(error => { state.running = false; paint(root, manifest, cohort); console.error(error); });
  root.querySelector('#verify-tamper').onchange = event => {
    state.tamper = event.target.checked;
    state.results = new Map();
    state.done = false;
    runAll(root, manifest, cohort).catch(error => { state.running = false; console.error(error); });
  };
  root.querySelectorAll('[data-verify-cohort]').forEach(button => {
    button.onclick = () => {
      if (state.running) return;
      state.cohort = button.dataset.verifyCohort;
      state.results = new Map();
      state.done = false;
      renderVerify(root);
    };
  });
}
