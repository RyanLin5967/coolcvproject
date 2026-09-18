"""Read-only browser regression checks for local and standalone public presentations."""
import json
from pathlib import Path

from playwright.sync_api import sync_playwright

ROOT = Path(__file__).resolve().parents[1]
OUT = ROOT / "artifacts/qa/presentation"
OUT.mkdir(parents=True, exist_ok=True)

with sync_playwright() as playwright:
    browser = playwright.chromium.launch(
        executable_path="/Applications/Google Chrome.app/Contents/MacOS/Google Chrome", headless=True)
    evidence = []
    for port in (8765, 8767):
        for width in (1440, 390):
            page = browser.new_page(viewport={"width": width, "height": 1000})
            errors, requests = [], []
            page.on("pageerror", lambda error, errors=errors: errors.append(str(error)))
            page.on("request", lambda request, requests=requests: requests.append((request.method, request.url)))
            page.add_init_script("""const Native = window.EventSource;
                window.EventSource = class extends Native {constructor(...args){super(...args);window.qaStream=this}};""")
            page.goto(f"http://127.0.0.1:{port}")
            page.locator(".merge-story").wait_for()
            page.locator(".merge-photo img").evaluate("image=>image.decode()")
            page.evaluate("window.qaImage=document.querySelector('.merge-photo img')")
            page.locator('[data-story-mode="coverage"]').click()
            assert page.locator("[data-story-outcome]").inner_text().startswith("CoverageCV")
            page.locator('[data-story-source="1"]').click()
            assert page.evaluate("window.qaImage===document.querySelector('.merge-photo img')")
            page.screenshot(path=str(OUT / f"merge-{port}-{width}.png"), full_page=True)
            page.locator('#navigation [data-page="research"]').click()
            expected = {"pawns": [61.40, 79.20, 77.72], "all-pieces": [56.73, 73.68, 72.48],
                        "construction": [42.31, 51.62, 51.82]}
            for task, values in expected.items():
                page.locator(f'[data-research-task="{task}"]').click()
                page.locator(f'#benchmark-comparison[data-task="{task}"]').wait_for()
                actual = [float(text.split("AP")[0]) for text in page.locator(".score-value").all_text_contents()]
                assert actual == values, (task, actual)
                assert page.locator("details[open]").count() == 0
            page.screenshot(path=str(OUT / f"benchmarks-{port}-{width}.png"), full_page=True)
            page.locator('#navigation [data-page="experiments"]').click()
            page.locator('#prediction-example:not([disabled])').wait_for()
            page.wait_for_function("document.querySelector('#prediction-count-aware').textContent.match(/^\\d+ predictions/)")
            labels = page.locator('[data-experiment-project]').all_text_contents()
            assert len(labels) == len(set(labels)), labels
            highlights = json.loads((ROOT / "public-demo/static/prediction-highlights.json").read_text())
            task_labels = {"pawns": "Chess pawns", "all-pieces": "All chess pieces", "construction": "Construction safety"}
            for task, label in task_labels.items():
                page.locator('[data-experiment-project]').filter(has_text=label).click()
                page.wait_for_function("task=>document.querySelector('#experiment-results')?.dataset.signature.includes('highlights-'+task)", arg=task)
                page.locator('#prediction-example:not([disabled])').wait_for()
                page.wait_for_function("document.querySelector('#prediction-count-aware').textContent.match(/^\\d+ predictions/)")
                scores = [float(value) for value in page.locator('.prediction-score b').all_text_contents()]
                assert scores == expected[task], (task, scores)
                selection_id = highlights['selections'][task]['id']
                image_id = highlights['routes'][f'/jobs/{selection_id}/examples']['images'][0]['id']
                rows = highlights['routes'][f'/jobs/{selection_id}/predictions/{image_id}']
                for arm, boxes in rows.items():
                    count = sum(box['score'] >= .25 for box in boxes)
                    assert page.locator(f'#prediction-count-{arm}').inner_text().startswith(f'{count} predictions'), (task, arm)
            page.evaluate("""window.qaCanvas=document.querySelector('#prediction-aware');
                window.qaPixels=qaCanvas.toDataURL();window.qaMutations=0;
                new MutationObserver(rows=>{qaMutations+=rows.filter(r=>[...r.removedNodes].some(n=>n===qaCanvas||n.contains?.(qaCanvas))).length})
                .observe(document.querySelector('#content'),{childList:true,subtree:true});
                if(window.qaStream)for(let i=0;i<6;i++)window.qaStream.dispatchEvent(new MessageEvent('message',{data:'{}'}));""")
            page.wait_for_timeout(900)
            assert page.evaluate("qaCanvas===document.querySelector('#prediction-aware')&&qaCanvas.toDataURL()===qaPixels&&qaMutations===0")
            # Stress rapid selection changes; only the final response may draw.
            page.locator('#prediction-example').select_option('1')
            page.locator('#prediction-example').select_option('0')
            page.locator('#prediction-confidence').fill('0.5')
            page.locator('#prediction-confidence').dispatch_event('input')
            page.wait_for_timeout(500)
            assert page.locator('#confidence-label').inner_text() == '0.50'
            assert page.evaluate("qaCanvas===document.querySelector('#prediction-aware')")
            page.screenshot(path=str(OUT / f"predictions-{port}-{width}.png"), full_page=True)
            assert page.evaluate("document.documentElement.scrollWidth<=window.innerWidth"), (port, width)
            assert page.locator('#navigation').evaluate("nav=>nav.scrollWidth<=nav.clientWidth"), (port, width, 'nav')
            assert not errors, errors
            assert all(method == 'GET' for method, _ in requests)
            if port == 8767:
                assert not any('/api/' in url for _, url in requests)
                assert page.locator('.workbench-nav').is_hidden()
            evidence.append({"port": port, "width": width, "status": "passed", "errors": errors,
                             "canvas_stable_on_refresh": True, "benchmark_values": expected})
            page.close()
    browser.close()
    (OUT / "evidence.json").write_text(json.dumps(evidence, indent=2))
    print(json.dumps(evidence, indent=2))
