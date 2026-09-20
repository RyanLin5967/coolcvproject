import {renderMergeStory} from './merge-story.js';
import {benchmarkTasks, modelRoles, recordedExtrema, cohortBenchmark, loadVerificationManifest} from './benchmark-view.js';
import {renderVerify} from './verify-view.js';
const $=id=>document.getElementById(id);
const esc=x=>String(x??'').replace(/[&<>"']/g,c=>({'&':'&amp;','<':'&lt;','>':'&gt;','"':'&quot;',"'":'&#39;'}[c]));
const states={unknown:['Unknown','#e4bd70','#fff4dd','#a5792d'],positive_only:['Positives only','#8a9bd8','#edf0fb','#6075bc'],exhaustive:['Exhaustive','#6da58b','#e8f3ee','#167864'],verified_absent:['Verified absent','#b798cf','#f2ebf8','#8860a8']};
const armNames=Object.fromEntries(Object.entries(modelRoles).map(([key,role])=>[key,role.label]));
const publicDemo=document.documentElement.dataset.demo==='true';
let snapshotPromise, predictionHighlightsPromise;
function loadPredictionHighlights(){
 return predictionHighlightsPromise??=fetch('/static/prediction-highlights.json').then(response=>{if(!response.ok)throw new Error('Recorded prediction selections could not be loaded.');return response.json()}).catch(error=>{predictionHighlightsPromise=null;throw error});
}
const predictionCache=new Map(), imageCache=new Map(), exampleRequests=new Map();
let predictionRequest=0, researchRequest=0, renderedSignature='';
const deviceNames={cpu:'CPU',mps:'Apple GPU',cuda:'CUDA GPU'};
const colors=['#477dce','#e49938','#ad60b5','#46a99a','#d26068','#859849','#766dcc','#41a1c6','#bd8461','#7196bd','#bb668e','#8ea777','#99809d'];
const S={page:'merge',tab:'overview',state:null,project:null,revision:null,projectId:localStorage.getItem('coveragecv-project'),revisionId:null,pending:{},split:'train',filter:'all',limit:24,selectedJob:null,examples:{},sample:null,sampleMode:'observed',example:0,confidence:.25,researchTask:'pawns-base',research:null,verifyManifest:null};
let refreshToken=0,toastTimer,refreshTimer;
async function api(path,body,method='POST'){
 if(path.startsWith('/jobs/highlights-')){
  if(body!==undefined)throw new Error('Recorded selections are read-only.');
  const route=(await loadPredictionHighlights()).routes[path];
  if(!route)throw new Error('Recorded selection is unavailable.');
  if(!publicDemo&&path.endsWith('/examples'))return {...route,images:route.images.map(image=>({...image,image_url:image.local_image_url}))};
  return route;
 }
 if(publicDemo){
  if(body!==undefined)throw new Error('This is the public demo. Training and dataset edits run in the local workbench.');
  snapshotPromise??=fetch('/data/snapshot.json').then(r=>{if(!r.ok)throw new Error('Demo data could not be loaded.');return r.json()});
  const snapshot=await snapshotPromise;
  if(!(path in snapshot.routes))throw new Error('This view is available in the local workbench.');
  return snapshot.routes[path];
 }

 const options=body===undefined?{}:{method,headers:{'X-CoverageCV':'workbench'}};
 if(body instanceof FormData)options.body=body;else if(body!==undefined){options.body=JSON.stringify(body);options.headers['Content-Type']='application/json'}
 const response=await fetch('/api'+path,options);const data=await response.json();
 if(!response.ok)throw new Error(typeof data.detail==='string'?data.detail:data.message??JSON.stringify(data.detail??data));return data;
}
function toast(message,error=false){clearTimeout(toastTimer);$('toast').textContent=message;$('toast').className='visible'+(error?' error':'');toastTimer=setTimeout(()=>$('toast').className='',6000)}
function badge(status){return `<span class="badge ${esc(status)}">${esc(status)}</span>`}
function timeLabel(seconds){return seconds<60?`${seconds.toFixed(0)}s`:`${(seconds/60).toFixed(1)}m`}
function when(timestamp){return timestamp?new Date(timestamp*1000).toLocaleString([], {month:'short',day:'numeric',hour:'2-digit',minute:'2-digit'}):'—'}
function pct(value){return value==null?'—':(value*100).toFixed(1)+'%'}
function stateLabel(state){const [name,,bg,color]=states[state]??states.unknown;return `<span class="state-label" style="--state-bg:${bg};--state-color:${color}">${name}</span>`}
function pending(){return S.pending[S.revisionId]??{}}
function scheduleRefresh(){clearTimeout(refreshTimer);refreshTimer=setTimeout(()=>refresh().catch(e=>toast(e.message,true)),150)}
async function refresh(){
 const token=++refreshToken;const data=await api('/state');if(token!==refreshToken)return;S.state=data;
 if(!data.projects.some(p=>p.id===S.projectId)){S.projectId=defaultProject(data.projects)?.id??null;S.revisionId=null}
 if(S.projectId){const project=await api('/projects/'+S.projectId);if(token!==refreshToken)return;S.project=project;
  if(!S.revisionId||!project.revisions.some(r=>r.id===S.revisionId))S.revisionId=project.active_revision;
  const revision=await api('/revisions/'+S.revisionId);if(token!==refreshToken)return;S.revision=revision;
 }
 renderShell();
 const signature=JSON.stringify([S.page,S.projectId,S.revisionId,S.tab,S.project,S.revision]);
 if(signature!==renderedSignature){renderedSignature=signature;render()}
}

function renderShell(){
 document.body.dataset.page=S.page;
 document.querySelectorAll('[data-page]').forEach(b=>b.classList.toggle('selected',b.dataset.page===S.page));
 $('project-list').innerHTML=(S.state?.projects??[]).map(p=>`<button class="project-item ${p.id===S.projectId?'selected':''}" data-project="${p.id}" title="${esc(p.name)}">${esc(p.name)}</button>`).join('');
 const active=S.state?.jobs.filter(j=>['running','cancelling'].includes(j.status)).length??0;
 const queued=S.state?.jobs.filter(j=>j.status==='queued').length??0;
 $('running-count').textContent=active||'';$('queue-label').textContent=publicDemo?'Recorded results':active?`${active} active · ${queued} queued`:queued?`${queued} queued`:'Queue idle';
 $('breadcrumb').innerHTML=`Demo <span>/</span> ${esc(S.page==='datasets'?(S.project?.name??'Datasets'):({merge:'Merge demo',research:'Benchmarks',experiments:'Prediction comparison',verify:'Verify the numbers'}[S.page]??S.page[0].toUpperCase()+S.page.slice(1)))}`;
}
function render(){
 $('content').dataset.page=S.page;
 if(!S.state){$('content').innerHTML='<div class="loading"><div class="spin"></div>Opening your workspace…</div>';return}
 if(S.page==='merge'){renderMergeStory($('content'),{api,state:S.state,publicDemo,onCompare:(task)=>{S.researchTask=cohortKeyForTask(task);S.page='research';renderShell();render();window.scrollTo({top:0,behavior:'instant'})}});return}
 if(S.page==='platform'){renderPlatform();return}
 if(S.page==='research'){renderResearch();return}
 if(S.page==='verify'){renderVerify($('content')).catch(error=>toast(error.message,true));return}
 if(S.page==='activity'){renderActivity();return}
 if(!S.project){$('content').innerHTML=`<div class="empty"><div class="empty-symbol">▦</div><div class="eyebrow">EVERY LABEL HAS A CONTEXT</div><h1>Make coverage part of your dataset.</h1><p>Combine datasets, declare what each source actually annotates, and measure what happens when your model respects that policy.</p><button class="button primary" data-action="demo">Explore the working example →</button> <button class="button" data-action="import">Import your dataset</button></div>`;return}
 if(S.page==='experiments'){renderExperiments().catch(error=>toast(error.message,true));return}
 renderDataset();
}
// Map a dataset key from elsewhere in the app onto its headline cohort.
function cohortKeyForTask(task){
 const cohorts=S.verifyManifest?.cohorts??[];
 const forTask=cohorts.filter(c=>c.task===task);
 return (forTask.find(c=>c.headline)??forTask[0])?.key??(task?`${task}-base`:'pawns-base');
}
// The benchmark selection is a cohort key; the predictions page needs its dataset.
function selectedDatasetTask(){
 return S.verifyManifest?.cohorts.find(c=>c.key===S.researchTask)?.task??S.researchTask;
}
function pageHeading(title,description,actions=''){return `<div class="page-heading"><div><div class="eyebrow">${S.page==='datasets'?'DATASET PROJECT':S.page==='experiments'?'RECORDED MODEL PREDICTIONS':'LOCAL WORKBENCH'}</div><h1>${esc(title)}</h1><p class="muted small-text">${description}</p></div><div class="heading-actions">${actions}</div></div>`}
function stat(value,label,detail){return `<div class="stat"><div class="stat-label">${label}</div><div class="stat-number">${value}</div><div class="stat-detail">${detail}</div></div>`}
function scoreCards(rows){
 return `<div class="score-cards">${rows.map(row=>`<article class="score-card ${row.role}"><span class="role-tag">${row.tag}</span><h2>${row.label}</h2><p>${row.description}</p><div class="score-value">${(row.metrics.AP*100).toFixed(2)}<span>AP50:95</span></div><div class="score-track"><i style="width:${row.metrics.AP*100}%"></i></div><div class="score-secondary">AP50 <b>${(row.metrics.AP50*100).toFixed(2)}</b></div></article>`).join('')}</div>`;
}
function methodExplainer(){
 return `<p class="reference-explanation"><b>Which one is ours?</b> CoverageCV is our method. The control uses ordinary training. The fully labeled reference is our comparison run with extra labels—not a frontier model.</p>`;
}
async function renderResearch(force=false){
 const request=++researchRequest;
 let data,manifest;
 try{
  [data,manifest]=await Promise.all([(!force&&S.research)?S.research:api('/research'),
                                     S.verifyManifest??loadVerificationManifest()]);
 }catch(error){
  if(S.page!=='research')return;
  $('content').innerHTML=`<div class="empty"><h2>Recorded results are unavailable</h2><p>${esc(error.message)}</p></div>`;return;
 }
 if(S.page!=='research'||request!==researchRequest)return;
 S.research=data;S.verifyManifest=manifest;
 const cohorts=manifest.cohorts;
 if(!cohorts.some(c=>c.key===S.researchTask))S.researchTask=(cohorts.find(c=>c.headline)??cohorts[0]).key;
 const benchmark=cohortBenchmark(manifest,S.researchTask);
 let html=`<div class="page-heading"><div><div class="eyebrow">MATCHED COMPARISONS ON COMPLETE VALIDATION LABELS</div><h1>Benchmarks</h1><p class="muted">Each comparison holds the recipe, the update budget and the images fixed, and averages every recorded seed. Every score here can be recomputed in your browser.</p></div><a class="button small" href="https://github.com/RyanLin5967/coolcvproject" target="_blank" rel="noopener">View source ↗</a></div>`;
 html+=`<div class="callout neutral verify-pointer">These are not stored constants. <button class="link-button" data-page="verify">Recompute them in your browser →</button></div>`;
 html+=`<div class="benchmark-heading"><div><p>Green is our method.</p></div><span class="metric-key">AP50:95 · higher is better</span></div><div class="dataset-switcher" aria-label="Benchmark dataset">${cohorts.map(c=>`<button data-research-task="${c.key}" aria-pressed="${c.key===S.researchTask}" class="${c.key===S.researchTask?'selected':''}"><b>${esc(c.name)}</b><span>${esc(c.detail)}</span></button>`).join('')}</div>`;
 if(!benchmark){$('content').innerHTML=html+'<div class="card card-body">Recorded results are not available yet.</div>';return}
 const {rows,gain,gap}=benchmark;
 const seeds=benchmark.seeds.length;
 html+=`<div id="benchmark-comparison" data-task="${benchmark.key}"><div class="benchmark-context"><b>${esc(benchmark.name)}</b><span>${esc(benchmark.recipe)}</span><span>${benchmark.images} validation images · ${benchmark.boxes} boxes</span><span>${seeds} seed${seeds===1?'':'s'}, averaged</span></div>${scoreCards(rows)}<div class="benchmark-takeaway"><strong>${gain>=0?'+':''}${gain.toFixed(2)}<span> AP points</span></strong><div><b>Coverage-aware training versus ordinary training</b><p>Same images, same labels, same recipe and the same number of updates. The only change is whether training treats an unlabelled region as background.</p></div></div>${methodExplainer()}</div>`;
 html+=`<p class="benchmark-note">The fully labelled reference is ${gap>=0?`${gap.toFixed(2)} points ahead`:`${Math.abs(gap).toFixed(2)} points behind`}; it is one comparison run with every label, not a frontier model.${seeds===1?' This cohort has a single seed, so treat it as a signal rather than a result.':''}</p>`;
 if(benchmark.task==='construction')html+=`<p class="benchmark-note">Construction annotations have known inconsistencies, recorded in the label audit.</p>`;
 else html+=`<p class="benchmark-note">These chess results share a small board and camera domain. Scores from different datasets are not directly comparable.</p>`;
 html+=`<details class="disclosure"><summary>What does this score measure?</summary><div><p><b>AP50:95</b> rewards finding the right objects and drawing accurate boxes around them. It averages performance across several box-overlap requirements. A score of 78 is not “78% of images correct.”</p><p><b>AP50</b> uses a looser overlap requirement, so its number is usually higher. All three models are scored on the same validation images and reference labels.</p><p>${rows.map(row=>`${row.label}: ${esc(row.runs[0].method.replaceAll('_',' '))} · ${benchmark.steps.toLocaleString()} updates · ${benchmark.resolution}px · seed${seeds===1?' ':'s '}${benchmark.seeds.join(', ')}`).join('<br>')}. Recall at the operating point: ${rows.map(row=>`${row.label} ${(100*row.metrics.recall_at_threshold).toFixed(1)}%`).join(' · ')}.</p></div></details>`;
 const extrema=recordedExtrema[benchmark.task];
 if(extrema)html+=`<details class="disclosure"><summary>Best recorded result for each arm · not a matched comparison</summary><div><p>Across every recorded recipe and seed, the strongest CoverageCV result and the weakest control look like this. We keep it visible because it is in the results ledger, but it selects a minimum against a maximum across <b>different</b> recipes, budgets and inference settings, so the difference is not an isolated coverage effect and is not the number we claim.</p><div class="table-wrap"><table><thead><tr><th>Arm</th><th>Recipe as recorded</th><th>AP50:95</th></tr></thead><tbody>${['naive','aware','complete_reference'].map(role=>`<tr class="${role}"><td>${esc(modelRoles[role].label)}</td><td>${esc(extrema[role].method.replaceAll('_',' '))} · ${extrema[role].steps.toLocaleString()} updates · ${extrema[role].resolution}px · ${extrema[role].passes} pass${extrema[role].passes===1?'':'es'}</td><td>${(100*extrema[role].metrics.AP).toFixed(2)}</td></tr>`).join('')}</tbody></table></div><p>The matched comparison above is the claim; this table is context.</p></div></details>`;
 const acquisition=data.acquisition;
 if(benchmark.task==='construction'&&acquisition?.status==='completed'){
  const guided=acquisition.cases.guided,random=acquisition.cases.random;
  html+=`<details class="disclosure" id="acquisition-evidence"><summary>Separate experiment: choosing what to label · +${acquisition.primary_AP_delta_points.toFixed(2)} AP</summary><div><p>We also tested which missing labels to review. Both strategies get 150 image/class reviews and the same extra training.</p><div class="table-wrap"><table><thead><tr><th>Review strategy</th><th>New boxes</th><th>AP50:95</th></tr></thead><tbody><tr><td>Random selection · control</td><td>${random.acquisition.added_boxes}</td><td>${(random.metrics.AP*100).toFixed(2)}</td></tr><tr class="aware"><td>CoverageCV guided selection · ours</td><td>${guided.acquisition.added_boxes}</td><td>${(guided.metrics.AP*100).toFixed(2)}</td></tr></tbody></table></div><p class="benchmark-note">One-run simulation using withheld published training labels. These are additional annotations, not new human work. Equal review counts do not imply equal annotation effort.</p></div></details>`;
 }
 html+=`<div class="benchmark-next"><p>Recompute these scores yourself, or inspect the predictions behind them.</p><button class="button primary" data-page="verify">Verify the numbers →</button> <button class="button" data-page="experiments">Compare predictions →</button></div>`;
 const task=data.tasks?.[benchmark.task];
 if(task)html+=`<details class="disclosure archive"><summary>Every recorded recipe & limitations</summary><div><p>The cards above show one matched cohort. The table below retains every recorded recipe average for this dataset, including stronger controls and regressions.</p><div class="table-wrap"><table><thead><tr><th>Recorded recipe</th><th>Runs</th><th>AP50:95</th></tr></thead><tbody>${Object.values(task.methods).map(method=>`<tr><td>${esc(method.label)}</td><td>${method.n}</td><td>${(100*method.metrics.AP.mean).toFixed(2)}</td></tr>`).join('')}</tbody></table></div><p>Rows here are not automatically matched comparisons. Validation guided experiment design. These results do not establish state-of-the-art performance.</p><a href="https://github.com/RyanLin5967/coolcvproject/blob/main/docs/DECISIONS.md" target="_blank" rel="noopener">Read the full experiment record ↗</a></div></details>`;
 const viewKey=JSON.stringify([S.researchTask,rows.map(row=>row.metrics),acquisition?.status]);
 if($('content').dataset.viewKey===viewKey&&$('benchmark-comparison'))return;
 $('content').dataset.viewKey=viewKey;
 $('content').innerHTML=html;
}
function renderDataset(){
 const r=S.revision,ready=r?.status==='ready',d=r?.diagnostics,stats=r?.class_stats??{},values=Object.values(stats);
 const known=values.reduce((n,c)=>n+c.known_images,0),unknown=values.reduce((n,c)=>n+c.unknown_images,0);
 const actions=`${badge(r?.status??'queued')} ${ready?`<a class="button" href="/api/revisions/${r.id}/download">↓ Export learner view</a><button class="button primary" data-action="train">Run comparison →</button>`:''}`;
 let html=pageHeading(S.project.name,`Revision ${r?.number??1} <span class="muted">·</span> ${r?.classes?.length??0} classes <span class="muted">·</span> ${r?.spec?.sources.length??0} declared sources`,actions);
 html+=`<div class="stats">${stat(d?.images??'—','Images in this revision','Train, validation and test stay separate')}${stat(d?.splits.train.annotations??'—','Observed training boxes','Only labels the learner is allowed to see')}${stat(ready?pct(known/Math.max(1,known+unknown)):'—','Declared class coverage','Training image × class cells')}${stat(ready?unknown:'—','Cells without negative evidence','Unknown or positives-only supervision')}</div>`;
 html+=`<div class="tabs">${[['overview','Overview'],['policy','Coverage policy'],['versions','Versions & diff']].map(([id,label])=>`<button class="${S.tab===id?'selected':''}" data-tab="${id}">${label}${id==='policy'&&Object.keys(pending()).length?' · unsaved':''}</button>`).join('')}</div>`;
 const job=S.project.jobs.find(j=>j.kind==='compile'&&j.revision_id===r?.id);
 if(!ready){html+=`<div class="callout ${r?.status==='queued'?'neutral':''}">${r?.status==='queued'?'Compilation is queued. The worker will verify images, annotations and coverage before publishing this revision.':`Compilation ${esc(r?.status)}: ${esc(job?.error?.message??'Inspect the source declarations and create a corrected revision.')}`}</div>`}
 if(S.tab==='policy'||!ready)html+=policyPanel();else if(S.tab==='versions')html+=versionPanel();else html+=overviewPanel();
 $('content').innerHTML=html;
 if(S.tab==='versions'&&ready){const previous=S.project.revisions.find(x=>x.status==='ready'&&x.number<r.number);if(previous)loadDiff(previous.id)}
}
function overviewPanel(){
 const r=S.revision,train=r.samples.filter(s=>s.split==='train');
 const rows=r.classes.map((name,i)=>`<div class="coverage-row"><span title="${esc(name)}">${esc(name)}</span><div class="coverage-strip">${train.map(s=>`<button style="--color:${states[s.states[i]]?.[1]??'#ddd'}" data-sample="${s.id}" title="Image ${s.id} · ${esc(name)}: ${esc(s.states[i])}" aria-label="Inspect image ${s.id}"></button>`).join('')}</div><span class="coverage-count">${r.class_stats[name].known_images}/${train.length}</span></div>`).join('');
 const sources=r.spec.sources.map(source=>{const samples=r.samples.filter(im=>im.sources.includes(source.id));return `<div class="source-row"><span class="source-icon">▧</span><div class="source-name"><strong>${esc(source.id)}</strong><small>${samples.length} images · ${esc(source.split)} · ${esc(source.revision.slice(0,8))}</small></div>${badge(source.split)}</div>`}).join('');
 const samples=r.samples.filter(s=>s.split===S.split&&(S.filter==='all'||S.filter==='unknown'&&s.states.includes('unknown')||S.filter==='empty'&&s.boxes===0));
 const unsupported=r.classes.filter(c=>r.class_stats[c].boxes===0);const warning=unsupported.length?`<div class="callout"><b>No training positives: ${unsupported.map(esc).join(', ')}.</b> These classes are declared in the ontology but have no observed training examples. Treat their evaluation scores as a data-support issue, not evidence of a learned class.</div>`:'';return warning+`<div class="two-col"><div class="card"><div class="card-heading"><div><h3>What each source knows</h3><p>One column per training image. Click a cell to inspect the evidence.</p></div></div><div class="card-body"><div class="legend">${Object.entries(states).map(([,v])=>`<span style="--color:${v[1]}">${v[0]}</span>`).join('')}</div>${rows}<div class="callout">A missing box is not a negative label. Unknown classes keep observed positives while avoiding unjustified negative classification supervision.</div></div></div><div class="card"><div class="card-heading"><h3>Source declarations</h3><button class="button small" data-tab="policy">Edit policy ↗</button></div><div class="card-body">${sources}</div></div></div>
 <div class="card"><div class="card-heading"><div><h3>Explore the dataset</h3><p>Inspect labels, withheld reference annotations and source provenance.</p></div><div class="filters"><select id="split-filter" aria-label="Dataset split">${['train','valid','test'].map(s=>`<option ${s===S.split?'selected':''} value="${s}">${s==='valid'?'Validation':s[0].toUpperCase()+s.slice(1)} · ${r.diagnostics.splits[s].images}</option>`).join('')}</select><select id="coverage-filter" aria-label="Coverage filter"><option value="all">All coverage</option><option value="unknown" ${S.filter==='unknown'?'selected':''}>Unknown coverage</option><option value="empty" ${S.filter==='empty'?'selected':''}>No observed boxes</option></select></div></div><div class="gallery">${samples.slice(0,S.limit).map(s=>`<button class="image-tile" data-sample="${s.id}" aria-label="Inspect image ${s.id}"><img loading="lazy" src="/api/revisions/${r.id}/images/${s.id}" alt="Dataset image ${s.id}"><span class="image-info"><span>Image ${s.id} · ${s.boxes} boxes</span>${s.states.includes('unknown')?'<i class="tile-dot" title="Unknown class coverage"></i>':''}</span></button>`).join('')}</div><div class="gallery-footer"><span>${Math.min(S.limit,samples.length)} of ${samples.length} images · ${esc(r.manifest.digest.slice(0,12))}</span>${samples.length>S.limit?'<button class="button small" data-action="more">Show more images</button>':''}</div></div>`;
}
function policyPanel(){
 const r=S.revision,changes=pending();if(!r)return '';
 const rows=r.spec.sources.map((source,i)=>`<tr><td><b>${esc(source.id)}</b><div class="muted small-text">${esc(source.split)}${Object.keys(source.overrides??{}).length?' · per-image overrides also apply':''}</div></td>${r.classes.map((name,j)=>{const key=i+':'+j,current=changes[key]?.state??source.coverage[name]??'unknown';return `<td><select data-policy="${key}" data-state="${current}" aria-label="${esc(source.id)} ${esc(name)} coverage">${Object.entries(states).map(([state,[label]])=>`<option value="${state}" ${current===state?'selected':''}>${label}</option>`).join('')}</select></td>`}).join('')}</tr>`).join('');
 return `<div class="card"><div class="card-heading"><div><h3>Declare annotation scope, source by source</h3><p>Saving creates a new immutable revision. Previous artifacts and experiments remain available.</p></div></div><div class="table-wrap"><table class="policy-table"><thead><tr><th>Source / split</th>${r.classes.map(c=>`<th>${esc(c)}</th>`).join('')}</tr></thead><tbody>${rows}</tbody></table></div><div class="policy-footer"><span class="muted small-text" id="policy-count">${Object.keys(changes).length} unsaved changes</span><button class="button primary" id="save-policy" ${Object.keys(changes).length?'':'disabled'}>Save & compile revision →</button></div></div><div class="card"><div class="card-body policy-key">${Object.entries(states).map(([state,[label,color]])=>`<div style="--color:${color}"><h3>${label}</h3><p>${{unknown:'No claim about annotation completeness. Observed positives still train the model.',positive_only:'Existing boxes are positive evidence; missing boxes are not negative evidence.',exhaustive:'This source declares every object of this class annotated. Negative supervision is permitted.',verified_absent:'The source explicitly certifies this class is absent. A positive box contradicting this claim fails compilation.'}[state]}</p></div>`).join('')}</div></div><div class="callout neutral">Declarations express the data owner’s knowledge. The compiler checks consistency; it cannot verify that a human found every object.</div>`;
}
function versionPanel(){
 return `<div class="card"><div class="card-heading"><h3>Dataset lineage</h3><span class="muted small-text">${S.project.revisions.length} revisions · append-only policies</span></div>${S.project.revisions.map(r=>`<div class="revision-row"><div class="revision-dot">${r.number}</div><div class="revision-body"><strong>Revision ${r.number}</strong> ${badge(r.status)}<p>${r.diagnostics?`${r.diagnostics.images} images · ${r.diagnostics.annotations} observed boxes`:'Compilation has not published an artifact'}</p><div class="revision-meta">${when(r.created)}${r.bundle?' · '+esc(r.bundle.split('/').pop().slice(0,16)):''}</div></div><button class="button small" data-revision="${r.id}">${r.id===S.revisionId?'Viewing':'Inspect'}</button></div>`).join('')}</div><div id="revision-diff"></div>`;
}
async function loadDiff(previous){
 const rid=S.revisionId;try{const d=await api(`/revisions/${rid}/diff/${previous}`);if(rid!==S.revisionId||!$('revision-diff'))return;
 $('revision-diff').innerHTML=`<div class="card"><div class="card-heading"><div><h3>What changed since the previous compiled revision</h3><p>Coverage differences are matched by image content, not filenames.</p></div></div><div class="card-body"><div class="diff-stats"><div><b>${d.changed_images}</b><span>images with policy changes</span></div><div><b>${d.changed_class_cells}</b><span>class coverage changes</span></div><div><b>${d.negative_training_cells_before} → ${d.negative_training_cells_after}</b><span>cells permitting negative supervision</span></div></div>${d.changes.length?`<div class="table-wrap"><table><thead><tr><th>Image</th><th>Class</th><th>Before</th><th>After</th></tr></thead><tbody>${d.changes.slice(0,12).map(c=>`<tr><td>${c.image_id} · ${c.split}</td><td>${esc(c.class_name)}</td><td>${stateLabel(c.before)}</td><td>${stateLabel(c.after)}</td></tr>`).join('')}</tbody></table></div>`:'<p class="muted small-text">No class coverage cells changed.</p>'}</div></div>`;
 }catch(e){toast(e.message,true)}
}
function jobCard(job){
 const total=job.payload?.steps??0,step=job.live?.step??0,active=['queued','running','cancelling'].includes(job.status);
 return `<div class="job-card"><div class="job-line">${badge(job.status)}<div class="job-title">${job.kind==='compile'?'Compile dataset revision':job.kind==='imported'?'Verified saved comparison':'Training comparison'}<small>${when(job.created)} · ${job.kind==='compile'?'image and policy verification':`${total||Object.values(job.result?.arms??{})[0]?.run.steps||'—'} updates per arm · seed ${job.payload?.seed??Object.values(job.result?.arms??{})[0]?.run.seed??'—'}`}</small></div>${active?`<button class="button small danger" data-cancel="${job.id}">Cancel</button>`:''}<button class="button small" data-log="${job.id}">Logs</button></div>${active&&job.kind==='train'?`<div class="progress"><i style="width:${Math.min(100,100*step/Math.max(1,total))}%"></i></div><div class="live-details"><span>${esc(armNames[job.live?.arm]??'Waiting for a worker')} · ${esc(job.live?.phase??job.status)}</span><span>${step} / ${total} updates${job.live?.loss!=null?' · loss '+job.live.loss.toFixed(3):''}</span></div>`:''}${job.error?`<p class="error-text">${esc(job.error.message??JSON.stringify(job.error))}</p>`:''}<div id="log-${job.id}"></div></div>`;
}
// Order the datasets the way the rest of the site does, pawns first, and open there.
const TASK_ORDER=['pawns','all-pieces','construction'];
function orderedProjects(projects){
 return [...(projects??[])].sort((a,b)=>{
  const rank=project=>{const index=TASK_ORDER.indexOf(projectTask(project));return index<0?TASK_ORDER.length:index};
  return rank(a)-rank(b);
 });
}
function defaultProject(projects){return orderedProjects(projects)[0]}
function projectTask(project){
 if(/construction/i.test(project.name))return 'construction';
 if(/all.*13|all.*chess|all.*pieces/i.test(project.name))return 'all-pieces';
 if(/pawn|partial annotations/i.test(project.name))return 'pawns';
 return null;
}
async function renderExperiments(){
 const projectId=S.projectId,task=projectTask(S.project);
 const highlight=task?(await loadPredictionHighlights()).selections[task]:null;
 if(S.page!=='experiments'||S.projectId!==projectId)return;
 const jobs=S.project.jobs.filter(job=>['train','imported'].includes(job.kind));
 const measured=[...(highlight?[highlight]:[]),...jobs.filter(job=>Object.keys(job.result?.arms??{}).length)];
 const complete=measured.filter(job=>['naive','aware','complete_reference'].every(arm=>job.result.arms[arm]));
 const ranked=[...(complete.length?complete:measured)].sort((a,b)=>Math.min(...Object.values(b.result.arms).map(x=>x.run.total_training_steps??x.run.steps))-Math.min(...Object.values(a.result.arms).map(x=>x.run.total_training_steps??x.run.steps)));
 if(!measured.some(job=>job.id===S.selectedJob)){S.selectedJob=highlight?.id??ranked[0]?.id??null;S.example=0}
 const selected=measured.find(job=>job.id===S.selectedJob),arms=selected?.result?.arms??{};
 const active=jobs.filter(job=>['queued','running','cancelling'].includes(job.status));
 const progress=active.map(jobCard).join('');
 const key=JSON.stringify([S.projectId,selected?.id,arms,measured.map(job=>job.id)]);
 const previous=$('experiment-results');
 if(previous?.dataset.signature===key){
  const slot=$('experiment-progress');if(slot.innerHTML!==progress)slot.innerHTML=progress;
  return;
 }
 let html=pageHeading('Predictions','Compare the three models on the same validation image.');
 html+=`<div class="dataset-switcher" aria-label="Prediction dataset">${orderedProjects(S.state.projects).map(project=>`<button data-experiment-project="${project.id}" aria-pressed="${project.id===S.projectId}" class="${project.id===S.projectId?'selected':''}">${esc(benchmarkTasks[projectTask(project)]?.name??project.name)}</button>`).join('')}</div><div id="experiment-progress">${progress}</div>`;
 if(!selected){$('content').innerHTML=html+'<div class="card empty"><h2>No measured predictions yet</h2><p>Run a local comparison to inspect the results here.</p></div>';return}
 const first=Object.values(arms)[0].run,isHighlight=selected.selection_kind==='cohort',cohort=selected.cohort;
 const rows=['naive','aware','complete_reference'].filter(arm=>arms[arm]).map(role=>({role,...modelRoles[role],metrics:arms[role].metrics}));
 html+=`<div id="experiment-results" data-signature="${esc(key)}"><div class="benchmark-context"><b>${esc(S.project.name)}</b>${isHighlight?`<span>${esc(cohort.recipe)}</span><span>${cohort.seeds.length===1?'single seed':`${cohort.seeds.length} seeds, averaged`} \u2014 the same comparison as Benchmarks</span><span>Boxes drawn by the seed ${cohort.seed} checkpoints</span>`:`<span>${(first.total_training_steps??first.steps).toLocaleString()} updates · one recorded run</span><span>Seed ${first.seed}</span>`}</div>`;
 if(measured.length>1&&!publicDemo)html+=`<details class="disclosure"><summary>Other saved runs</summary><div><label>Recorded comparison<select class="input" id="result-selection">${measured.map(job=>{const run=Object.values(job.result.arms)[0].run;return `<option value="${job.id}" ${job.id===selected.id?'selected':''}>${job.selection_kind==='cohort'?'Matched cohort · same as Benchmarks':`${run.total_training_steps??run.steps} updates · seed ${run.seed} · ${deviceNames[run.device]??esc(run.device)}`}</option>`}).join('')}</select></label><p>Each selection shows its own real checkpoint. Other saved comparisons retain their own original models and scores.</p></div></details>`;
 html+=`<section class="card prediction-section"><div class="card-heading"><div><h2>Compare predictions</h2><p>Colored boxes are the model's predictions on the same validation image.</p></div></div><div class="prediction-controls"><label>Validation image<select id="prediction-example" aria-label="Validation example" disabled><option>Loading images…</option></select></label><label>Display confidence <span id="confidence-label">${S.confidence.toFixed(2)}</span><input id="prediction-confidence" type="range" min=".01" max=".9" step=".01" value="${S.confidence}"></label><p>Changing confidence updates the boxes, not the benchmark scores.</p></div><div class="prediction-grid">${rows.map(row=>`<article class="prediction-card ${row.role}"><span class="role-tag">${row.tag}</span><h3>${row.label}</h3><p class="prediction-score"><b>${(row.metrics.AP*100).toFixed(2)}</b> AP50:95 <span>· ${isHighlight?(cohort.seeds.length===1?'full validation set':`mean of ${cohort.seeds.length} seeds`):'full validation set'}</span></p><canvas id="prediction-${row.role}" width="640" height="640" aria-label="${row.label} predictions"></canvas><p id="prediction-count-${row.role}">Loading recorded predictions…</p></article>`).join('')}</div></section>${methodExplainer()}</div>`;
 $('content').innerHTML=html;
 if(!exampleRequests.has(selected.id))exampleRequests.set(selected.id,api(`/jobs/${selected.id}/examples`).catch(error=>{exampleRequests.delete(selected.id);throw error}));
 exampleRequests.get(selected.id).then(examples=>{
  S.examples[selected.id]=examples;
  if(S.page!=='experiments'||S.selectedJob!==selected.id)return;
  const valid=examples.images??[];S.example=Math.min(S.example,Math.max(0,valid.length-1));
  const select=$('prediction-example');if(!select)return;
  select.innerHTML=valid.map((sample,index)=>`<option value="${index}" ${index===S.example?'selected':''}>Image ${index+1} of ${valid.length} · ${sample.annotations.length} labeled objects</option>`).join('');
  select.disabled=!valid.length;
  return drawPredictions();
 }).catch(error=>toast(error.message,true));
}
async function draw(canvas,url,image,boxes,classes){
 const token=Symbol();canvas.request=token;
 if(!imageCache.has(url))imageCache.set(url,(async()=>{const bitmap=new Image();bitmap.src=url;await bitmap.decode();return bitmap})().catch(error=>{imageCache.delete(url);throw error}));
 const bitmap=await imageCache.get(url);
 if(canvas.request!==token||!canvas.isConnected)return;
 const frame=document.createElement('canvas');frame.width=image.width;frame.height=image.height;
 const c=frame.getContext('2d');c.drawImage(bitmap,0,0,image.width,image.height);c.lineWidth=Math.max(2,image.width/250);c.font=`600 ${Math.max(11,image.width/52)}px system-ui`;
 for(const box of boxes){const index=box.category_id-1,[x,y,w,h]=box.bbox,color=colors[index%colors.length];c.strokeStyle=color;c.strokeRect(x,y,w,h);const label=classes[index]+(box.score!=null?' '+box.score.toFixed(2):'');const width=c.measureText(label).width+6;c.fillStyle=color;c.fillRect(x,Math.max(0,y-18),width,18);c.fillStyle='white';c.fillText(label,x+3,Math.max(13,y-4))}
 // Compose offscreen, then replace the pixels synchronously; refreshes never blank the canvas.
 if(canvas.width!==image.width)canvas.width=image.width;if(canvas.height!==image.height)canvas.height=image.height;
 canvas.getContext('2d').drawImage(frame,0,0);
}
async function inspectSample(id){
 S.sampleMode='observed';S.sample=await api(`/revisions/${S.revisionId}/samples/${id}`);const s=S.sample;
 $('sample-title').textContent='Image '+s.image.id;$('sample-kicker').textContent=s.coverage.split.toUpperCase()+' · '+s.provenance.observations.map(o=>o.source).join(', ');
 $('sample-provenance').textContent=s.provenance.observations.map(o=>`${o.source} · ${o.revision}\n${o.evidence}\n${o.attribution}`).join('\n\n')+'\n\nSHA-256 '+s.provenance.content_sha256;
 $('sample-states').innerHTML=S.revision.classes.map((name,i)=>`<div class="state-row"><span>${esc(name)}</span>${stateLabel(s.coverage.states[i])}</div>`).join('');
 $('sample-explanation').textContent=s.coverage.states.some(x=>['unknown','positive_only'].includes(x))?'Some classes have no negative evidence in this image. Observed positive boxes remain supervised; unannotated classes are not assumed absent.':'Every class has declared negative evidence. The coverage adapter preserves the ordinary classification loss.';
 document.querySelector('[data-mode=reference]').disabled=s.reference===null;
 $('sample-dialog').showModal();drawSample();
}
function drawSample(){const s=S.sample;document.querySelectorAll('[data-mode]').forEach(b=>b.classList.toggle('selected',b.dataset.mode===S.sampleMode));const boxes=S.sampleMode==='reference'?s.reference:s.annotations;$('sample-label-count').textContent=`${boxes.length} ${S.sampleMode==='reference'?'reference':'observed'} boxes · reference labels are excluded from partial-label training`;return draw($('sample-canvas'),s.image_url,s.image,boxes,S.revision.classes)}
async function drawPredictions(){
 const request=++predictionRequest,id=S.selectedJob,examples=S.examples[id],sample=examples?.images?.[S.example],confidence=S.confidence;
 if(!sample)return;
 const key=id+':'+sample.id;
 if(!predictionCache.has(key))predictionCache.set(key,api(`/jobs/${id}/predictions/${sample.id}`).catch(error=>{predictionCache.delete(key);throw error}));
 const predictions=await predictionCache.get(key);
 if(request!==predictionRequest||S.page!=='experiments'||id!==S.selectedJob||sample.id!==S.examples[id]?.images?.[S.example]?.id)return;
 await Promise.all(Object.entries(predictions).map(async([arm,boxes])=>{
  const canvas=$('prediction-'+arm);if(!canvas)return;
  const visible=boxes.filter(box=>box.score>=confidence);
  await draw(canvas,sample.image_url,sample,visible,examples.classes);
  if(request===predictionRequest&&canvas.isConnected)$('prediction-count-'+arm).textContent=`${visible.length} predictions · ${sample.annotations.length} labeled objects`;
 }));
}
function renderActivity(){
 const jobs=S.state.jobs;
 $('content').innerHTML=pageHeading('A record of the work.','Compile decisions, queued experiments and execution results persist across restarts.')+`<div class="card"><div class="card-heading"><h3>Recent jobs</h3><span class="muted small-text">${jobs.length} records</span></div>${jobs.length?jobs.map(j=>`<div class="activity-row"><span class="activity-icon">${j.kind==='compile'?'▦':'◈'}</span><div class="activity-main"><b>${j.kind==='compile'?'Dataset compilation':j.kind==='imported'?'Saved experiment adopted':'Training comparison'}</b><small>${esc(S.state.projects.find(p=>p.id===j.project_id)?.name??j.project_id)} · ${when(j.created)}</small>${j.error?`<div class="error-text">${esc(j.error.message)}</div>`:''}</div>${badge(j.status)}${['queued','running'].includes(j.status)?`<button class="button small danger" data-cancel="${j.id}">Cancel</button>`:''}<button class="button small" data-log="${j.id}">Logs</button></div><div class="inline-details" id="log-${j.id}"></div>`).join(''):'<div class="card-body muted">Your first import will appear here.</div>'}</div>`;
}
async function renderPlatform(){
 $('content').innerHTML=pageHeading('Connected to the real product.','Roboflow stores the versioned dataset; its round trip is checked against the compiled artifact.')+'<div class="loading"><div class="spin"></div>Reading provider evidence…</div>';
 try{const p=await api('/platform');if(S.page!=='platform')return;const d=p.dataset,v=p.verification,model=p.corrected_deployment??p.deployment,hosted=p.hosted;
 $('content').innerHTML=pageHeading('Connected to the real product.','Saved API receipts distinguish uploaded artifacts, verified data and hosted execution.')+`<div class="two-col"><div class="card"><div class="card-heading"><div class="platform-icon">roboflow</div>${badge(v?.status==='passed'?'ready':'pending')}</div><div class="card-body"><h2>Coverage-preserving dataset upload</h2><p class="muted small-text">The observed training view is uploaded, exported as COCO and checked image-by-image.</p>${[['Project',d?.project_name??'Not connected'],['Version',d?.version??'—'],['Training images',v?.splits?.train.images??'—'],['Validation images',v?.splits?.valid.images??'—'],['Coordinate verification',v?.status==='passed'?'Within 0.0001 pixels':'Not verified']].map(([k,x])=>`<div class="platform-row"><span>${k}</span><b>${esc(x)}</b></div>`).join('')}<div class="provider-links">${d?.url?`<a class="button" href="${esc(d.url)}" target="_blank" rel="noopener">Open dataset version ↗</a>`:''}</div></div></div><div class="card"><div class="card-heading"><h3>Execution evidence</h3></div><div class="card-body"><h3>Custom model upload</h3><p class="muted small-text">Version ${esc(model?.version??'—')} serves the exported coverage-aware model. The corrected export preserves class names and has been checked against native predictions.</p><div class="platform-row"><span>Provider status</span>${badge(model?.status??'unverified')}</div><div class="platform-row"><span>Hosted inference</span><b>${p.semantic_parity?.status==='passed'?'Class and box parity verified':p.hosted_inference?.status==='failed_semantic_parity'?'Label layout issue; corrected export pending':'Not yet verified'}</b></div>${p.corrected_deployment?`<div class="platform-row"><span>Corrected export · v${p.corrected_deployment.version}</span>${badge(p.corrected_deployment.status)}</div>`:''}<h3 style="margin-top:25px">Roboflow-trained baseline</h3><div class="platform-row"><span>Provider status</span>${badge(hosted?.status??'not submitted')}</div><p class="muted small-text">${hosted?.status==='request_rejected'?esc(hosted.reason):'Separate complete-label training. Its provider recipe is not treated as a matched arm in the custom-loss comparison.'}</p><div class="callout neutral">Custom training runs use our RF-DETR adapter. Uploading a coverage sidecar does not change the loss in ordinary hosted training.</div></div></div></div>`;
 }catch(e){toast(e.message,true)}
}
function openTrain(){if(S.revision?.status!=='ready'){toast('Compile a valid revision first.',true);return}const hardware=S.state?.hardware,mps=hardware?.devices?.mps===true;$('train-device').querySelector('[value=mps]').disabled=!mps;if(!mps)$('train-device').value='cpu';$('train-device-help').textContent=mps?'Apple GPU acceleration is available on this machine. CPU remains the default.':hardware?.mps_unavailable_reason??'Apple GPU acceleration is unavailable; local CPU training is available.';$('reference-option').classList.toggle('hidden',!S.project.reference_bundle);$('reference-option').querySelector('input').checked=false;$('train-error').textContent='';$('train-dialog').showModal()}
function openImport(){$('import-error').textContent='';$('import-dialog').showModal()}
async function demo(){const project=await api('/projects/demo',{});S.projectId=project.id;S.revisionId=null;S.page='datasets';S.tab='overview';localStorage.setItem('coveragecv-project',project.id);$('import-dialog').close();toast('Dataset imported. Compilation is queued.');await refresh()}
document.addEventListener('click',async e=>{const b=e.target.closest('button,a');if(!b)return;try{
 if(b.classList.contains('close-dialog')){b.closest('dialog').close();return}
 if(b.dataset.page){
  const oldPage=S.page;S.page=b.dataset.page;
  if(S.page==='experiments'&&oldPage==='research'){
   S.selectedJob=null;S.example=0;
   const project=S.state.projects.find(project=>projectTask(project)===selectedDatasetTask());
   if(project&&project.id!==S.projectId){S.projectId=project.id;S.revisionId=null;S.selectedJob=null;S.example=0;await refresh()}
   else {renderShell();render()}
  }else {renderShell();render()}
  window.scrollTo({top:0,behavior:'instant'});
 }
 if(b.dataset.researchTask){S.researchTask=b.dataset.researchTask;await renderResearch()}
 if(b.dataset.action==='research-refresh')await renderResearch(true);
 if(b.dataset.action==='how-it-works')$('how-it-works')?.scrollIntoView({behavior:'smooth'});
 if(b.dataset.experimentProject){S.projectId=b.dataset.experimentProject;S.revisionId=null;S.selectedJob=null;S.example=0;S.page='experiments';await refresh()}
 if(b.dataset.project){S.projectId=b.dataset.project;S.revisionId=null;S.page='datasets';S.tab='overview';localStorage.setItem('coveragecv-project',S.projectId);await refresh()}
 if(b.dataset.tab){S.tab=b.dataset.tab;renderDataset()}
 if(b.dataset.revision){S.revisionId=b.dataset.revision;await refresh()}
 if(b.dataset.sample!==undefined)await inspectSample(+b.dataset.sample);
 if(b.dataset.mode){S.sampleMode=b.dataset.mode;await drawSample()}
 if(b.dataset.cancel){await api('/jobs/'+b.dataset.cancel+'/cancel',{});toast('Cancellation requested. Completed artifacts are preserved.');await refresh()}
 if(b.dataset.log){const log=await api('/jobs/'+b.dataset.log+'/log');const box=$('log-'+b.dataset.log);box.innerHTML=`<pre class="log">${esc(log.text||'No worker output recorded for this job.')}</pre>`}
 if(b.dataset.action==='watch-demo'){$('video-dialog').showModal();$('video-dialog').querySelector('video').play().catch(()=>{})}
 if(b.dataset.action==='import')openImport();if(b.dataset.action==='train')openTrain();if(b.dataset.action==='demo')await demo();if(b.dataset.action==='more'){S.limit+=24;renderDataset()}
 if(b.dataset.action==='adopt'){await api('/projects/'+S.projectId+'/adopt-pilot',{});toast('Verified pilot measurements added to this project.');await refresh()}
 if(b.id==='save-policy'){const changes=Object.values(pending());const revision=await api('/projects/'+S.projectId+'/policy',{base_revision:S.revisionId,changes});delete S.pending[S.revisionId];S.revisionId=revision.id;S.tab='versions';toast('New revision saved. Compilation is queued.');await refresh()}
 }catch(error){toast(error.message,true)}});
document.addEventListener('change',e=>{const x=e.target;
 if(x.dataset.policy){const [i,j]=x.dataset.policy.split(':').map(Number),source=S.revision.spec.sources[i],name=S.revision.classes[j],old=source.coverage[name]??'unknown';S.pending[S.revisionId]??={};if(x.value===old)delete S.pending[S.revisionId][x.dataset.policy];else S.pending[S.revisionId][x.dataset.policy]={source:source.id,class_name:name,state:x.value};x.dataset.state=x.value;$('policy-count').textContent=Object.keys(pending()).length+' unsaved changes';$('save-policy').disabled=!Object.keys(pending()).length}
 if(x.id==='split-filter'){S.split=x.value;S.limit=24;renderDataset()}if(x.id==='coverage-filter'){S.filter=x.value;S.limit=24;renderDataset()}
 if(x.id==='result-selection'){S.selectedJob=x.value;S.example=0;renderExperiments()}if(x.id==='prediction-example'){S.example=+x.value;drawPredictions().catch(e=>toast(e.message,true))}
});
document.addEventListener('input',e=>{if(e.target.id==='prediction-confidence'){S.confidence=+e.target.value;$('confidence-label').textContent=S.confidence.toFixed(2);drawPredictions().catch(e=>toast(e.message,true))}});
$('video-dialog').addEventListener('close',()=>{$('video-dialog').querySelector('video').pause()});
$('import-button').onclick=openImport;$('sidebar-import').onclick=openImport;$('load-demo').onclick=()=>demo().catch(e=>$('import-error').textContent=e.message);
$('import-form').onsubmit=async e=>{e.preventDefault();try{let project;const name=$('import-name').value,file=$('import-file').files[0];if(file){const form=new FormData();form.append('file',file);project=await api('/projects/upload?name='+encodeURIComponent(name),form)}else{project=await api('/projects/import',{name,spec_path:$('import-path').value})}S.projectId=project.id;S.revisionId=null;S.page='datasets';S.tab='overview';$('import-dialog').close();toast('Import accepted. Compilation is queued.');await refresh()}catch(error){$('import-error').textContent=error.message}};
$('train-form').onsubmit=async e=>{e.preventDefault();try{const arms=[...document.querySelectorAll('input[name=arm]:checked')].map(x=>x.value);await api('/revisions/'+S.revisionId+'/train',{steps:+$('train-steps').value,seed:+$('train-seed').value,batch:+$('train-batch').value,timeout_seconds:+$('train-timeout').value,recipe:$('train-recipe').value,device:$('train-device').value,arms});$('train-dialog').close();S.page='experiments';toast('Experiment queued. Progress and results will appear here.');await refresh()}catch(error){$('train-error').textContent=error.message}};
if(publicDemo){
 document.body.classList.add('public-demo');
 for(const page of ['datasets','activity','platform'])document.querySelector(`[data-page="${page}"]`)?.remove();
 $('import-button').remove();$('sidebar-import').remove();$('project-list').hidden=true;
 $('queue-label').textContent='Recorded results';
}else{
 const events=new EventSource('/api/events');events.onopen=()=>{$('connection').style.color='#2b9a71'};events.onerror=()=>{$('connection').style.color='#c79a4d'};events.onmessage=scheduleRefresh;
 setInterval(()=>{if(document.visibilityState==='visible'&&S.state?.jobs.some(job=>['queued','running','cancelling'].includes(job.status)))scheduleRefresh()},2500);
}
render();refresh().catch(e=>{toast(e.message,true);$('content').innerHTML='<div class="empty"><h2>Could not open the workspace</h2><p>'+esc(e.message)+'</p></div>'});
