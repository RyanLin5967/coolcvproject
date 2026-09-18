// The policies below are an explicit illustration, not a claim about this image's training provenance.
const esc = value => String(value ?? '').replace(/[&<>"']/g, char => ({'&':'&amp;','<':'&lt;','>':'&gt;','"':'&quot;',"'":'&#39;'}[char]));
const requests = new WeakMap();
let lastSource = 0;
let lastMode = 'ordinary';

async function loadStory(api, state) {
  const project = state?.projects?.find(item => /construction/i.test(item.name)) ?? state?.projects?.[0];
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
    const helmet = classes.indexOf('helmet') + 1;
    const person = classes.indexOf('person') + 1;
    const images = examples.images ?? [];
    let sample = images.find(image => helmet > 0 && person > 0 && [helmet, person].every(id => image.annotations.some(box => box.category_id === id)));
    let classIds = [helmet, person];
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
  root.querySelector('[data-story-image-caption]').textContent = `Source ${source === 0 ? 'A' : 'B'} supplies ${knownName} labels. ${unknownName} labels are withheld in this illustration.`;
  root.querySelector('[data-story-known]').textContent = `${knownName}: provided labels`;
  root.querySelector('[data-story-unknown]').textContent = `${unknownName}: ${aware ? 'unreviewed, protected' : 'unreviewed, at risk'}`;
  root.querySelector('[data-story-outcome]').innerHTML = `
    <span class="merge-result-kicker">${aware ? 'CoverageCV · our method' : 'Ordinary merge · baseline'}</span>
    <h2>${aware ? 'Unknown stays unknown.' : 'The object is there. The label is not.'}</h2>
    <p class="merge-result-description">${aware ? `A missing <strong>${esc(unknownName)}</strong> label is not evidence of absence. CoverageCV protects predictions for that unreviewed class from negative classification supervision.` : `This source never labeled <strong>${esc(unknownName)}</strong>. An ordinary training loss can penalize a correct prediction because it cannot match a provided label.`}</p>
    <div class="merge-decision"><span>Unmatched ${esc(unknownName)} prediction</span><strong>${aware ? 'Ignore the negative signal' : 'Apply a negative signal'}</strong></div>
    <div class="merge-retained"><span aria-hidden="true">✓</span><span>Keep all provided ${esc(knownName)} labels.</span></div>
    <div class="merge-retained"><span aria-hidden="true">${aware ? '✓' : '!'}</span><span>${hidden.length} reference ${hidden.length === 1 ? 'box illustrates' : 'boxes illustrate'} the unreviewed class.</span></div>
    <p class="merge-detail">${aware ? 'No invented boxes. No relabeling. A coverage declaration changes which classification errors the trainer is allowed to penalize.' : 'The dataset merge can be valid JSON and still erase the distinction between “not present” and “not annotated.”'}</p>`;
  root.querySelector('[data-story-action]').textContent = aware ? 'See the measured results →' : 'Fix the merge with CoverageCV →';
}

/** Render a cached, recorded-image explanation. api is the workbench or public-snapshot adapter. */
export async function renderMergeStory(container, {api, state, publicDemo = false, onCompare = () => {}}) {
  const existing = container.querySelector('.merge-story');
  if (existing) { existing.compare = onCompare; return; }
  container.innerHTML = `<section class="merge-story-loading" role="status">Loading the recorded merge example…</section>`;
  const loading = container.firstElementChild;
  let data;
  try { data = await loadStory(api, state); }
  catch (error) {
    if (container.firstElementChild !== loading) return;
    container.innerHTML = `<section class="merge-story-empty"><h1>A missing label is not a negative example.</h1><p>${esc(error.message)}</p><button class="button" data-story-fallback>See the measured results →</button></section>`;
    container.querySelector('[data-story-fallback]').addEventListener('click', onCompare);
    return;
  }
  if (container.firstElementChild !== loading) return;
  const {sample, names} = data;
  container.innerHTML = `<div class="merge-story" data-source="${lastSource}" data-mode="${lastMode}">
    <header class="merge-hero">
      <div class="merge-eyebrow"><span aria-hidden="true">↳</span> COVERAGE-AWARE DATASET COMPILER</div>
      <h1>A missing label is not<br class="merge-desktop-break"> a negative example.</h1>
      <p>Two datasets. Different labeling rules. Merge them carelessly, and a model can learn to ignore the objects you want it to find.</p>
      ${publicDemo?'<button class="merge-watch" data-action="watch-demo">Watch the walkthrough <span aria-hidden="true">↗</span></button>':''}
    </header>
    <section class="merge-stage" aria-label="Interactive dataset merge explanation">
      <div class="merge-stage-heading"><span class="merge-live-dot" aria-hidden="true"></span><strong>Try the merge</strong><span class="merge-recorded">Interactive explanation · recorded image</span></div>
      <div class="merge-inputs">
        <div class="merge-source-list" role="group" aria-label="Illustrative source policy">
          ${names.map((name, index) => `<button class="merge-source" data-story-source="${index}" aria-pressed="${index === lastSource}"><span class="merge-source-letter">${index === 0 ? 'A' : 'B'}</span><span><strong>${esc(name)} dataset</strong><small>Labels ${esc(name)} only</small></span><span class="merge-source-check" aria-hidden="true">✓</span></button>`).join('')}
        </div>
        <div class="merge-mode" role="group" aria-label="Merge behavior"><button data-story-mode="ordinary" aria-pressed="${lastMode === 'ordinary'}">Ordinary merge</button><button data-story-mode="coverage" aria-pressed="${lastMode === 'coverage'}">CoverageCV merge <span>OURS</span></button></div>
      </div>
      <div class="merge-comparison">
        <figure class="merge-visual">
          <div class="merge-photo" style="aspect-ratio:${sample.width}/${sample.height}"><img src="${esc(sample.image_url)}" width="${sample.width}" height="${sample.height}" alt="Recorded construction validation image with explanatory boxes for ${esc(names.join(' and '))}" decoding="async"><svg data-story-overlay viewBox="0 0 ${sample.width} ${sample.height}" aria-label="Reference label illustration" role="img"></svg></div>
          <div class="merge-legend"><span><i class="merge-legend-known"></i><span data-story-known></span></span><span><i class="merge-legend-unknown"></i><span data-story-unknown></span></span></div>
          <figcaption data-story-image-caption></figcaption>
        </figure>
        <div class="merge-outcome-panel"><div data-story-outcome aria-live="polite" aria-atomic="true"></div><button class="merge-primary" data-story-action></button></div>
      </div>
      <p class="merge-disclaimer">Illustration, not model predictions: the two source policies above are synthetic. Boxes come from a recorded validation image and explain the training rule. This image was not trained under these illustrative policies. Measured experiments are reported separately.</p>
    </section>
    <section class="merge-pipeline" aria-label="How CoverageCV works"><div><span>01</span><h3>Declare what is known</h3><p>Record which classes each source actually reviewed.</p></div><div><span>02</span><h3>Compile a safe dataset</h3><p>Keep coverage, labels and provenance together in an immutable artifact.</p></div><div><span>03</span><h3>Train with that context</h3><p>Preserve observed labels. Suppress unjustified negative supervision.</p></div></section>
    <div class="merge-proof-link"><p>Does this improve the model? Compare our method with the ordinary merge and a fully labeled control.</p><button data-story-compare>Explore the benchmarks <span aria-hidden="true">↗</span></button></div>
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
    if (event.target.closest('[data-story-compare]')) { root.compare(); return; }
    if (source || mode || event.target.closest('[data-story-action]')) updateStory(root, data);
  });
  updateStory(root, data);
}
