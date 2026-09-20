// The policies below are an explicit illustration, not a claim about this image's training provenance.
import {cohortBenchmark, loadVerificationManifest} from './benchmark-view.js';

const esc = value => String(value ?? '').replace(/[&<>"']/g, char => ({'&':'&amp;','<':'&lt;','>':'&gt;','"':'&quot;',"'":'&#39;'}[char]));
const requests = new WeakMap();
let lastSource = 0;
let lastMode = 'ordinary';

// Illustrate the merge with the dataset the measured result comes from, so the story does
// not change subject halfway through. Falls back to the construction example, then to any
// recorded image carrying two labelled classes.
// The recorded walkthrough shows the old construction merge story. Set true once
// public-demo/media/walkthrough.mp4 and .vtt are re-recorded against the chess story.
const WALKTHROUGH_MATCHES_PAGE = false;

const STORY_PREFERENCES = [
  {project: /chess.*partial|partial.*chess/i, classes: ['white-pawn', 'black-pawn']},
  {project: /construction/i, classes: ['helmet', 'person']},
];

async function loadStory(api, state) {
  const projects = state?.projects ?? [];
  const preference = STORY_PREFERENCES.find(entry => projects.some(item => entry.project.test(item.name)));
  const project = preference
    ? projects.find(item => preference.project.test(item.name))
    : projects[0];
  const wanted = preference?.classes ?? [];
  if (!project) throw new Error('The recorded merge example is not available yet.');
  let cache = requests.get(api);
  if (!cache) { cache = new Map(); requests.set(api, cache); }
  if (!cache.has(project.id)) cache.set(project.id, (async () => {
    const detail = await api(`/projects/${project.id}`);
    const jobs = (detail.jobs ?? []).filter(job => job.status === 'completed' && ['naive', 'aware', 'complete_reference'].every(arm => job.result?.arms?.[arm]));
    jobs.sort((a, b) => Number(b.result.arms.aware.run?.steps ?? 0) - Number(a.result.arms.aware.run?.steps ?? 0) || a.id.localeCompare(b.id));
    if (!jobs.length) throw new Error('The recorded merge example is not available yet.');
    const examples = await api(`/jobs/${jobs[0].id}/examples`);
    const classes = examples.classes ?? [];
    const [first, second] = wanted.map(name => classes.indexOf(name) + 1);
    const images = examples.images ?? [];
    let sample = images.find(image => first > 0 && second > 0 && [first, second].every(id => image.annotations.some(box => box.category_id === id)));
    let classIds = [first, second];
    if (!sample) {
      sample = images.find(image => new Set(image.annotations.map(box => box.category_id)).size >= 2);
      if (sample) classIds = [...new Set(sample.annotations.map(box => box.category_id))].slice(0, 2);
    }
    if (!sample) throw new Error('This example needs a recorded image containing two labeled classes.');
    return {sample, classIds, names: classIds.map(id => classes[id - 1] ?? `Class ${id}`)};
  })().catch(error => { cache.delete(project.id); throw error; }));
  return cache.get(project.id);
}

function boxMarkup(box, sample, known, mode) {
  const [x, y, width, height] = box.bbox;
  const status = known ? 'known' : mode === 'coverage' ? 'protected' : 'unsafe';
  return `<rect class="merge-box ${status}" x="${x}" y="${y}" width="${width}" height="${height}" rx="2" vector-effect="non-scaling-stroke"><title>${esc(known ? 'Label provided by this illustrative source' : mode === 'coverage' ? 'Unreviewed class: no negative classification supervision' : 'Unreviewed class: an unmatched prediction can receive negative classification supervision')}</title></rect>`;
}

function updateStory(root, data) {
  const {sample, classIds, names} = data;
  const source = Number(root.dataset.source);
  const mode = root.dataset.mode;
  const knownId = classIds[source];
  const unknownId = classIds[1 - source];
  const knownName = names[source];
  const unknownName = names[1 - source];
  const hidden = sample.annotations.filter(box => box.category_id === unknownId);
  const aware = mode === 'coverage';
  root.querySelectorAll('[data-story-source]').forEach(button => button.setAttribute('aria-pressed', String(Number(button.dataset.storySource) === source)));
  root.querySelectorAll('[data-story-mode]').forEach(button => button.setAttribute('aria-pressed', String(button.dataset.storyMode === mode)));
  root.querySelector('[data-story-overlay]').innerHTML = sample.annotations.filter(box => classIds.includes(box.category_id)).map(box => boxMarkup(box, sample, box.category_id === knownId, mode)).join('');
  root.querySelector('[data-story-known]').textContent = `${knownName}: provided labels`;
  root.querySelector('[data-story-unknown]').textContent = `${unknownName}: ${aware ? 'unreviewed, protected' : 'unreviewed, at risk'}`;
  root.querySelector('[data-story-outcome]').innerHTML = `
    <span class="merge-result-kicker">${aware ? 'CoverageCV · our method' : 'Ordinary merge · baseline'}</span>
    <h2>${aware ? 'Unknown stays unknown.' : 'The object is there. The label is not.'}</h2>
    <p class="merge-result-description">${aware ? `A missing <strong>${esc(unknownName)}</strong> label is not evidence of absence. CoverageCV protects predictions for that unreviewed class from negative classification supervision.` : `This source never labeled <strong>${esc(unknownName)}</strong>. An ordinary training loss can penalize a correct prediction because it cannot match a provided label.`}</p>
    <div class="merge-decision"><span>Unmatched ${esc(unknownName)} prediction</span><strong>${aware ? 'Ignore the negative signal' : 'Apply a negative signal'}</strong></div>
    <div class="merge-retained"><span aria-hidden="true">✓</span><span>Keep all provided ${esc(knownName)} labels.</span></div>
    <div class="merge-retained"><span aria-hidden="true">${aware ? '✓' : '!'}</span><span>${hidden.length} reference ${hidden.length === 1 ? 'box illustrates' : 'boxes illustrate'} the unreviewed class.</span></div>
    ${aware ? '' : '<p class="merge-detail">The dataset merge can be valid JSON and still erase the distinction between “not present” and “not annotated.”</p>'}`;
  root.querySelector('[data-story-action]').textContent = aware ? 'View benchmark results →' : 'Fix the merge with CoverageCV →';
}

function resultsPreview(manifest) {
  // Matched cohorts from the verification manifest, so the landing table cannot disagree
  // with the Benchmarks page or with what the Verify page recomputes. This used to show
  // each arm's best recorded result across different recipes, which read as a controlled
  // comparison and was not one.
  if (!manifest) return '';
  const roles = ['naive', 'aware', 'complete_reference'];
  const rows = manifest.cohorts.map(cohort => {
    try { return cohortBenchmark(manifest, cohort.key); } catch { return null; }
  }).filter(Boolean);
  if (!rows.length) return '';
  return `<section class="merge-results" aria-labelledby="merge-results-title">
    <div class="merge-results-heading"><div><span class="merge-results-kicker">RECORDED EXPERIMENTS</span><h2 id="merge-results-title">The measured results</h2></div><p>AP50:95 · higher is better</p></div>
    <table class="merge-results-table"><thead><tr><th scope="col">Dataset</th><th scope="col">Ordinary training</th><th scope="col" class="ours">CoverageCV<span>OUR METHOD</span></th><th scope="col">Fully labeled reference</th></tr></thead><tbody>
      ${rows.map(row => `<tr data-story-result="${esc(row.key)}"><th scope="row"><button data-story-compare="${esc(row.key)}" aria-label="View ${esc(row.name)} benchmarks">${esc(row.name)} <span aria-hidden="true">↗</span></button></th>${roles.map(role => `<td class="${role === 'aware' ? 'ours' : ''}">${(row.rows.find(entry => entry.role === role).metrics.AP * 100).toFixed(2)}</td>`).join('')}</tr>`).join('')}
    </tbody></table>
    <p class="merge-results-note">Each row is one matched comparison: same recipe, same update budget, same images, averaged over every recorded seed. Every number here can be recomputed in your browser.</p>
    <div class="merge-results-next"><p>Explore each dataset and its model predictions.</p><button class="merge-primary" data-story-compare>Open full benchmarks <span aria-hidden="true">→</span></button></div>
  </section>`;
}

/** Render a cached, recorded-image explanation. api is the workbench or public-snapshot adapter. */
export async function renderMergeStory(container, {api, state, publicDemo = false, onCompare = () => {}}) {
  const existing = container.querySelector('.merge-story');
  if (existing) { existing.compare = onCompare; return; }
  container.innerHTML = `<section class="merge-story-loading" role="status">Loading the recorded merge example…</section>`;
  const loading = container.firstElementChild;
  let data, manifest = null;
  // The results table is matched-cohort data from the verification bundle; if it cannot
  // be loaded the table is omitted rather than filled with anything less checkable.
  try { manifest = await loadVerificationManifest(); } catch { manifest = null; }
  try { data = await loadStory(api, state); }
  catch (error) {
    if (container.firstElementChild !== loading) return;
    container.innerHTML = `<section class="merge-story-empty"><h1>A missing label is not a negative example.</h1><p>${esc(error.message)}</p><button class="button primary" data-story-fallback>View benchmark results →</button></section>`;
    container.querySelector('[data-story-fallback]').addEventListener('click', () => onCompare());
    return;
  }
  if (container.firstElementChild !== loading) return;
  const {sample, names} = data;
  container.innerHTML = `<div class="merge-story" data-source="${lastSource}" data-mode="${lastMode}">
    <header class="merge-hero">
      <div class="merge-eyebrow"><span aria-hidden="true">↳</span> COVERAGE-AWARE DATASET COMPILER</div>
      <h1>A missing label is not<br class="merge-desktop-break"> a negative example.</h1>
      <p>Two datasets. Different labeling rules. Merge them carelessly, and a model can learn to ignore the objects you want it to find.</p>
      <div class="merge-hero-actions"><button class="merge-primary" data-story-compare>View benchmark results <span aria-hidden="true">→</span></button>${publicDemo&&WALKTHROUGH_MATCHES_PAGE?'<button class="merge-watch" data-action="watch-demo">Watch the walkthrough <span aria-hidden="true">↗</span></button>':''}</div>
    </header>
    <section class="merge-stage" aria-label="Interactive dataset merge explanation">
      <div class="merge-stage-heading"><span class="merge-live-dot" aria-hidden="true"></span><strong>Try the merge</strong><span class="merge-recorded">Illustrative example</span></div>
      <div class="merge-inputs">
        <div class="merge-source-list" role="group" aria-label="Illustrative source policy">
          ${names.map((name, index) => `<button class="merge-source" data-story-source="${index}" aria-pressed="${index === lastSource}"><span class="merge-source-letter">${index === 0 ? 'A' : 'B'}</span><span><strong>${esc(name)} dataset</strong><small>Labels ${esc(name)} only</small></span><span class="merge-source-check" aria-hidden="true">✓</span></button>`).join('')}
        </div>
        <div class="merge-mode" role="group" aria-label="Merge behavior"><button data-story-mode="ordinary" aria-pressed="${lastMode === 'ordinary'}">Ordinary merge</button><button data-story-mode="coverage" aria-pressed="${lastMode === 'coverage'}">CoverageCV merge <span>OURS</span></button></div>
      </div>
      <div class="merge-comparison">
        <figure class="merge-visual">
          <div class="merge-photo" style="aspect-ratio:${sample.width}/${sample.height}"><img src="${esc(sample.image_url)}" width="${sample.width}" height="${sample.height}" alt="Recorded validation image with explanatory boxes for ${esc(names.join(' and '))}" decoding="async"><svg data-story-overlay viewBox="0 0 ${sample.width} ${sample.height}" aria-label="Reference label illustration" role="img"></svg></div>
          <div class="merge-legend"><span><i class="merge-legend-known"></i><span data-story-known></span></span><span><i class="merge-legend-unknown"></i><span data-story-unknown></span></span></div>
        </figure>
        <div class="merge-outcome-panel"><div data-story-outcome aria-live="polite" aria-atomic="true"></div><button class="merge-primary" data-story-action></button></div>
      </div>
    </section>
    ${resultsPreview(manifest)}
    <section class="merge-pipeline" aria-label="How CoverageCV works"><div><span>01</span><h3>Declare what is known</h3><p>Record which classes each source actually reviewed.</p></div><div><span>02</span><h3>Compile a safe dataset</h3><p>Keep coverage, labels and provenance together in an immutable artifact.</p></div><div><span>03</span><h3>Train with that context</h3><p>Preserve observed labels. Suppress unjustified negative supervision.</p></div></section>
  </div>`;
  const root = container.firstElementChild;
  root.compare = onCompare;
  root.dataset.publicDemo = String(publicDemo);
  root.addEventListener('click', event => {
    const source = event.target.closest('[data-story-source]');
    const mode = event.target.closest('[data-story-mode]');
    if (source) { lastSource = Number(source.dataset.storySource); root.dataset.source = String(lastSource); }
    if (mode) { lastMode = mode.dataset.storyMode; root.dataset.mode = lastMode; }
    if (event.target.closest('[data-story-action]')) {
      if (root.dataset.mode === 'coverage') { root.compare(); return; }
      lastMode = 'coverage'; root.dataset.mode = 'coverage';
    }
    const compare = event.target.closest('[data-story-compare]');
    if (compare) { root.compare(compare.dataset.storyCompare || undefined); return; }
    if (source || mode || event.target.closest('[data-story-action]')) updateStory(root, data);
  });
  updateStory(root, data);
}
