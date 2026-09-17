#!/usr/bin/env python3
"""Build an offline idea explorer and a printable research report."""
from pathlib import Path
import json
from urllib.parse import urlparse

ROOT = Path(__file__).resolve().parents[1]
ideas = json.loads((ROOT / 'ideas.json').read_text())
stats = json.loads((ROOT / 'research_stats.json').read_text())

def source_label(url):
    parsed = urlparse(url)
    tail = parsed.path.rstrip('/').split('/')[-1] or parsed.netloc
    return f'{parsed.netloc}: {tail.replace("-", " ")}'

report = [(ROOT / 'tools/report_preamble.md').read_text().rstrip(),
          '\n| # | Project | Main field | Recommendation |',
          '| --- | --- | --- | --- |']
for idea in ideas:
    report.append(f"| {idea['id']} | [{idea['name']}](#idea-{idea['id']}) | {idea['category']} | {idea['tier']} |")
for idea in ideas:
    report += [f'\n<a id="idea-{idea["id"]}"></a>', f"\n### {idea['id']}. {idea['name']}",
               f"\n**{idea['pitch']}**", f"\n*{idea['tier']} · {idea['category']}*"]
    for field, label in [('user','Who uses it'), ('need','Why it matters'), ('mechanism','What to build'),
                         ('demo','The demonstration'), ('ambition','Full ambition'), ('attack','The strongest attack'),
                         ('proof','Evidence that would make it impressive'), ('kill','What would disprove the idea'),
                         ('constraints','External constraints'), ('evidence','Current evidence status')]:
        report.append(f'\n**{label}.** {idea[field]}')
    report.append('\n**Sources:** ' + ' · '.join(f'[{source_label(url)}]({url})' for url in idea['sources']) + '.')
    report.append('\n**Detailed investigation:** ' + ' · '.join(f'[{name.removesuffix(".md").replace("_", " ")}](lanes/{name})' for name in idea['lanes']) + '.')
report.append('\n' + (ROOT / 'tools/report_recommendation.md').read_text())
(ROOT / 'REPORT.md').write_text('\n'.join(report))

template = r'''<!doctype html>
<html lang="en"><head><meta charset="utf-8"><meta name="viewport" content="width=device-width,initial-scale=1">
<title>Roboflow — surviving project ideas</title>
<style>
:root{color-scheme:light;--ink:#10253b;--muted:#506477;--line:#d6e0e8;--accent:#125bba;--bg:#f4f7fa;--card:#fff}*{box-sizing:border-box}body{margin:0;background:var(--bg);color:var(--ink);font:16px/1.55 system-ui,-apple-system,sans-serif}main{max-width:1260px;margin:auto;padding:40px 24px 80px}a{color:var(--accent)}h1{font-size:clamp(30px,5vw,52px);line-height:1.08;letter-spacing:-1.6px;max-width:840px;margin:14px 0 22px}h2{font-size:23px;line-height:1.25;margin:8px 0 12px}h3{font-size:18px;margin:22px 0 6px}p{margin:8px 0 15px}.eyebrow{color:var(--accent);font-weight:750;font-size:13px;text-transform:uppercase;letter-spacing:1.5px}.lead{font-size:19px;max-width:900px}.muted{color:var(--muted)}.meta{display:flex;gap:9px;flex-wrap:wrap;margin:20px 0}.pill{font-size:13px;background:#e9eff6;border:1px solid var(--line);border-radius:20px;padding:5px 11px}.recommend{background:#eaf3ff;border-left:4px solid var(--accent);padding:18px 22px;margin:26px 0}.controls{display:grid;grid-template-columns:2fr 1fr 1.3fr;gap:14px;background:white;border:1px solid var(--line);padding:18px;border-radius:14px;margin:28px 0 12px}label{display:block;font-size:13px;font-weight:650}input[type=search],select{width:100%;margin-top:5px;padding:11px;border:1px solid #aebecb;border-radius:7px;font:inherit;background:white;color:var(--ink)}button{border:1px solid #9db2c4;background:white;color:var(--ink);padding:8px 14px;border-radius:7px;font:inherit;cursor:pointer}button:hover{background:#eaf3ff}button:focus-visible,input:focus-visible,select:focus-visible,summary:focus-visible,a:focus-visible{outline:3px solid #ed9c19;outline-offset:3px}.resultline{display:flex;justify-content:space-between;gap:16px;align-items:center;margin:12px 0 22px}.grid{display:grid;grid-template-columns:repeat(2,minmax(0,1fr));gap:20px}.card{background:var(--card);border:1px solid var(--line);border-radius:14px;padding:24px;box-shadow:0 3px 14px #14345606}.card.top{border-top:4px solid var(--accent)}.cardhead{display:flex;justify-content:space-between;gap:12px;align-items:start}.number{color:#758ca1;font:700 14px ui-monospace,monospace}.category{font-size:13px;color:var(--muted);margin-bottom:12px}.tier{display:inline-block;font-size:12px;letter-spacing:.1px;font-weight:700;padding:4px 9px;border-radius:6px;background:#eaf3ff}.tier.experimental{background:#fff2db;color:#7b4b00}.select{font-size:12px;white-space:nowrap;cursor:pointer}.select input{width:16px;height:16px;vertical-align:middle;margin:0 5px 2px 0}.pitch{font-size:17px}.summary-demo{background:#f5f8fc;padding:12px 14px;border-radius:8px;font-size:14px}details{border-top:1px solid var(--line);margin-top:18px;padding-top:14px}summary{cursor:pointer;color:var(--accent);font-weight:650}.detail{font-size:14px}.detail p{margin:6px 0 13px}.links{display:flex;flex-wrap:wrap;gap:8px 14px;font-size:13px}.compare{background:white;border:1px solid #a9bfd1;border-radius:14px;margin:22px 0;padding:22px;overflow:auto}.compare table{border-collapse:collapse;width:100%;min-width:650px;font-size:14px}.compare th,.compare td{vertical-align:top;text-align:left;padding:13px;border-bottom:1px solid var(--line)}.compare th:first-child{width:140px}.compare h2{margin-top:0}.empty{grid-column:1/-1;background:white;padding:30px;border-radius:12px}footer{margin-top:35px;padding-top:22px;border-top:1px solid var(--line);font-size:14px}.noscript{padding:24px;background:#fff2db}.visually-hidden{position:absolute;width:1px;height:1px;overflow:hidden;clip:rect(0,0,0,0)}@media(max-width:760px){main{padding:24px 16px 50px}.grid,.controls{grid-template-columns:1fr}.card{padding:20px}.resultline{align-items:start;flex-direction:column}}@media print{.controls,.resultline,.select,button{display:none}.grid{display:block}.card{break-inside:avoid;margin-bottom:16px}body{background:white}main{padding:0}details{display:block}.compare{display:none}}
</style></head><body><main>
<div class="eyebrow">Research dossier / September 17, 2026</div>
<h1>Projects that could earn Roboflow’s attention.</h1>
<p class="lead">Twelve ideas that survived company research, existing-feature checks, and independent novelty and technical attacks. Infrastructure and backend first, with serious CV/ML alternatives.</p>
<div class="meta"><span class="pill">7 research lanes</span><span class="pill">2 independent critics</span><span class="pill">71 Exa queries</span><span class="pill">44 named candidates → 12 choices</span><span class="pill">No coding-time cutoff</span></div>
<p class="muted">Search counts describe discovery coverage, not independently verified sources. A surviving proposal is not a proven product: each card identifies what is established, what must be measured, and what would disprove it.</p>
<div class="recommend"><strong>My first choice: Workflow Counterexample Engine.</strong> It addresses an explicit release-quality priority with a reusable engineering contribution. Choose <strong>EventLab</strong> for the most tangible systems demo, <strong>Coverage-Aware Dataset Compiler</strong> for a data/ML bridge, or <strong>Fair Admission</strong> for an ambitious runtime-performance result.</div>
<nav class="links" aria-label="Research files"><a href="REPORT.md">Full report</a><a href="AUDIT.md">Selection and rejection audit</a><a href="critiques/novelty.md">Novelty critique</a><a href="critiques/technical.md">Technical critique</a><a href="ideas.json">Structured cards</a></nav>
<div class="controls"><label>Search ideas<input id="query" type="search" placeholder="e.g. state, GPU, annotations, physical"></label><label>Field<select id="category"><option value="">All fields</option></select></label><label>Recommendation<select id="tier"><option value="">All surviving ideas</option><option>Top recommendation</option><option>Strong alternative</option><option>Ambitious experimental option</option></select></label></div>
<div class="resultline"><div><strong id="count" aria-live="polite"></strong> <span class="muted">Select up to three to compare.</span></div><button id="clear" type="button">Reset filters and selection</button></div>
<div id="selection-note" class="muted" role="status" aria-live="polite"></div>
<section id="compare" class="compare" aria-label="Selected idea comparison" hidden></section>
<section id="cards" class="grid" aria-label="Surviving ideas"></section>
<noscript><p class="noscript">Enable JavaScript for filters and comparison, or read the <a href="REPORT.md">complete Markdown report</a>.</p></noscript>
<footer><p>Local, standalone artifact. No analytics, remote scripts, external fonts, or network requests. External citations open only when selected.</p><p>Every idea has a concrete user, mechanism, prior-art attack, proof requirement and failure criterion. Hardware, training, labels and user validation remain real constraints; estimated coding duration is intentionally omitted.</p></footer>
</main><script>
const ideas = __IDEAS__;
const selected = new Set();
const q = document.querySelector('#query'), category = document.querySelector('#category'), tier = document.querySelector('#tier');
const esc = s => String(s).replace(/[&<>"']/g, c => ({'&':'&amp;','<':'&lt;','>':'&gt;','"':'&quot;',"'":'&#39;'}[c]));
const fields = [['user','Who uses it'],['need','Why it matters'],['mechanism','What to build'],['ambition','Full ambition'],['attack','The strongest attack'],['proof','Evidence that would make it impressive'],['kill','What would disprove it'],['constraints','External constraints'],['evidence','Current evidence status']];
for (const tag of [...new Set(ideas.flatMap(x=>x.tags))].sort()) { const o=document.createElement('option');o.value=tag;o.textContent=tag;category.append(o); }
function render(){
  const query=q.value.toLowerCase().trim();
  const shown=ideas.filter(x=>(!query||JSON.stringify(x).toLowerCase().includes(query))&&(!category.value||x.tags.includes(category.value))&&(!tier.value||x.tier===tier.value));
  document.querySelector('#count').textContent=`${shown.length} of ${ideas.length} ideas`;
  document.querySelector('#cards').innerHTML=shown.map(x=>`<article class="card ${x.tier==='Top recommendation'?'top':''}" id="idea-${x.id}"><div class="cardhead"><span class="number">${x.id}</span><label class="select"><input type="checkbox" data-select="${x.id}" ${selected.has(x.id)?'checked':''} aria-label="Compare ${esc(x.name)}">Compare</label></div><h2>${esc(x.name)}</h2><div class="category">${esc(x.category)}</div><span class="tier ${x.tier.includes('experimental')?'experimental':''}">${esc(x.tier)}</span><p class="pitch">${esc(x.pitch)}</p><p class="summary-demo"><strong>The demo:</strong> ${esc(x.demo)}</p><details><summary>Mechanism, evidence, attacks and validation</summary><div class="detail">${fields.map(([key,label])=>`<h3>${label}</h3><p>${esc(x[key])}</p>`).join('')}<h3>Sources</h3><div class="links">${x.sources.map((url,i)=>`<a href="${esc(url)}" target="_blank" rel="noopener noreferrer">${i+1}. ${esc(new URL(url).hostname)}</a>`).join('')}</div><h3>Detailed research</h3><div class="links">${x.lanes.map(name=>`<a href="lanes/${esc(name)}">${esc(name.replace('.md','').replaceAll('_',' '))}</a>`).join('')}</div></div></details></article>`).join('') || '<p class="empty">No ideas match these filters. Reset the filters to see all twelve.</p>';
  for(const input of document.querySelectorAll('[data-select]'))input.addEventListener('change',()=>{
    const id=input.dataset.select;
    if(input.checked&&selected.size===3){input.checked=false;document.querySelector('#selection-note').textContent='Three ideas are already selected. Deselect one to compare another.';return;}
    input.checked?selected.add(id):selected.delete(id);document.querySelector('#selection-note').textContent='';renderCompare();
  });
  renderCompare();
}
function renderCompare(){
  const el=document.querySelector('#compare'), rows=ideas.filter(x=>selected.has(x.id));el.hidden=rows.length===0;
  if(!rows.length){el.innerHTML='';return;}
  const compareFields=[['pitch','Purpose'],['mechanism','Mechanism'],['demo','Demonstration'],['proof','Proof required'],['attack','Main attack'],['constraints','External constraints']];
  el.innerHTML=`<h2>Compare ${rows.length} selected ${rows.length===1?'idea':'ideas'}</h2><table><thead><tr><th scope="col">Dimension</th>${rows.map(x=>`<th scope="col">${esc(x.name)}<br><span class="muted">${esc(x.tier)}</span></th>`).join('')}</tr></thead><tbody>${compareFields.map(([key,label])=>`<tr><th scope="row">${label}</th>${rows.map(x=>`<td>${esc(x[key])}</td>`).join('')}</tr>`).join('')}</tbody></table>`;
}
for(const el of [q,category,tier])el.addEventListener('input',render);
document.querySelector('#clear').addEventListener('click',()=>{q.value='';category.value='';tier.value='';selected.clear();document.querySelector('#selection-note').textContent='';render();});
render();
</script></body></html>'''
safe_data = json.dumps(ideas, ensure_ascii=False).replace('<', '\\u003c')
(ROOT / 'ideas.html').write_text(template.replace('__IDEAS__', safe_data))
print(f'Built REPORT.md and ideas.html from {len(ideas)} cards.')
