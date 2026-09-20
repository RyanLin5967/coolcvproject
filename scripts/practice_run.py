"""Dress rehearsal: read every score the site displays and say whether it is checkable.

The Verify page proves its own rows. This asks the harder question: of all the numbers a
visitor actually sees while walking the site, which ones can they recompute, and which are
still bare assertions? Anything in the second list is something to avoid quoting on camera.
"""
import json
import re
from pathlib import Path

from playwright.sync_api import sync_playwright

ROOT = Path(__file__).resolve().parents[1]
BASE = "http://127.0.0.1:8767"
MANIFEST = json.loads((ROOT / "public-demo/static/verify/manifest.json").read_text())

# Every AP a visitor can recompute in the browser, as it would be rendered (2dp of AP*100).
# That includes each run's own score and the seed averages the pages display, since the
# Verify page derives those averages from the same recomputed runs.
VERIFIABLE = {}
COHORT_MEANS = {}
for cohort in MANIFEST["cohorts"]:
    for run in cohort["runs"]:
        for field in ("AP", "AP50"):
            VERIFIABLE.setdefault(f"{100 * run['expected'][field]:.2f}", []).append(f"{run['id']}.{field}")
    for role in ("naive", "aware", "complete_reference"):
        values = [run["expected"] for run in cohort["runs"] if run["role"] == role]
        for field in ("AP", "AP50"):
            mean = sum(entry[field] for entry in values) / len(values)
            VERIFIABLE.setdefault(f"{100 * mean:.2f}", []).append(f"{cohort['key']}/{role}.mean.{field}")
            if field == "AP":
                COHORT_MEANS.setdefault(cohort["key"], {})[role] = mean


def caption_scores():
    """Scores quoted in the walkthrough captions are on-screen claims too.

    The embedded video is a surface a visitor sees, so a number in its caption track has
    to be recomputable for the same reason a number on a card does.
    """
    track = ROOT / "public-demo/media/walkthrough.vtt"
    if not track.exists():
        return []
    quoted = re.findall(r"\b\d{2}\.\d{1,2}\b", track.read_text())
    return [value for value in quoted if value not in VERIFIABLE]


def main():
    findings = {"verifiable": [], "unverifiable": [], "pages": {}}
    with sync_playwright() as playwright:
        browser = playwright.chromium.launch()
        page = browser.new_page(viewport={"width": 1440, "height": 1000})
        page.goto(BASE, wait_until="networkidle")

        # The landing table is the first scores a visitor sees; it was missed by the first
        # version of this rehearsal, which is how it kept showing cross-recipe extrema.
        page.wait_for_selector(".merge-results-table")
        landing = []
        rows = page.locator(".merge-results-table tbody tr")
        for index in range(rows.count()):
            cells = rows.nth(index).locator("td").all_inner_texts()
            landing.append([cell.strip() for cell in cells])
            for role, value in zip(("naive", "aware", "complete_reference"), landing[-1]):
                entry = {"page": "merge/landing", "role": role, "field": "AP", "value": value}
                if value in VERIFIABLE:
                    findings["verifiable"].append({**entry, "matches": VERIFIABLE[value]})
                else:
                    findings["unverifiable"].append(entry)
        findings["pages"]["merge/landing"] = {"rows": landing}

        page.click('[data-page="research"]')
        page.wait_for_selector(".score-cards")
        for cohort_spec in MANIFEST["cohorts"]:
            task = cohort_spec["key"]
            page.click(f'[data-research-task="{task}"]')
            page.wait_for_selector("#benchmark-comparison")
            page.wait_for_function("t => document.querySelector('#benchmark-comparison')?.dataset.task === t", arg=task)
            cards = []
            for index in range(page.locator(".score-card").count()):
                card = page.locator(".score-card").nth(index)
                role = (card.get_attribute("class") or "").replace("score-card", "").strip()
                ap = card.locator(".score-value").inner_text().split("\n")[0].strip()
                ap50 = card.locator(".score-secondary").inner_text().replace("AP50", "").strip()
                cards.append({"role": role, "AP": ap, "AP50": ap50})
            gain = page.locator(".benchmark-takeaway strong").inner_text().split("\n")[0].strip()
            findings["pages"][f"benchmarks/{task}"] = {"cards": cards, "headline_gain": gain}
            for card in cards:
                for field in ("AP", "AP50"):
                    value = card[field]
                    entry = {"page": f"benchmarks/{task}", "role": card["role"], "field": field, "value": value}
                    if value in VERIFIABLE:
                        findings["verifiable"].append({**entry, "matches": VERIFIABLE[value]})
                    else:
                        findings["unverifiable"].append(entry)

        # The prediction gallery is the third place a visitor reads scores. It used to show
        # the extrema while claiming to show the benchmark runs, so it is checked explicitly.
        page.click('[data-page="experiments"]')
        page.wait_for_selector("#experiment-results")
        gallery = {}
        buttons = page.locator("[data-experiment-project]")
        for index in range(buttons.count()):
            buttons.nth(index).click()
            page.wait_for_selector("#experiment-results")
            page.wait_for_timeout(600)
            name = page.locator(".benchmark-context b").inner_text().strip()
            values = [value.strip() for value in page.locator(".prediction-score b").all_inner_texts()]
            gallery[name] = values
            for role, value in zip(("naive", "aware", "complete_reference"), values):
                entry = {"page": "predictions", "role": role, "field": "AP", "value": value}
                if value in VERIFIABLE:
                    findings["verifiable"].append({**entry, "matches": VERIFIABLE[value]})
                else:
                    findings["unverifiable"].append(entry)
        findings["pages"]["predictions"] = gallery

        page.click('[data-page="verify"]')
        page.wait_for_selector("#verify-run")
        for cohort in MANIFEST["cohorts"]:
            page.click(f'[data-verify-cohort="{cohort["key"]}"]')
            page.wait_for_selector("#verify-run:not([disabled])")
            page.click("#verify-run")
            page.wait_for_selector(".verify-summary", timeout=180_000)
            summary = [cell.strip() for cell in page.locator(".verify-summary-rows b").all_inner_texts()]
            gain = page.locator(".verify-summary-main strong").inner_text().split("\n")[0].strip()
            findings["pages"][f"verify/{cohort['key']}"] = {"means": summary, "headline_gain": gain}

        browser.close()

    output = ROOT / "artifacts/qa/practice-run/findings.json"
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(json.dumps(findings, indent=1, sort_keys=True))

    print(f"Recomputable AP/AP50 values available to a visitor: {len(VERIFIABLE)}\n")
    for name, data in findings["pages"].items():
        print(f"  {name:34} {data}")
    print(f"\nDisplayed values that ARE recomputable:   {len(findings['verifiable'])}")
    print(f"Displayed values that are NOT recomputable: {len(findings['unverifiable'])}")
    for entry in findings["unverifiable"]:
        print(f"   ! {entry['page']:24} {entry['role']:20} {entry['field']:5} {entry['value']}")
    # The point of the rehearsal: the headline a viewer reads on Benchmarks must be the
    # same number the Verify page recomputes for that cohort. A mismatch here means the
    # site argues with itself on camera.
    mismatches = [f"walkthrough.vtt quotes {value}, which is not recomputable"
                  for value in caption_scores()]
    for key, means in COHORT_MEANS.items():
        cards = findings["pages"].get(f"benchmarks/{key}", {}).get("cards", [])
        shown = {card["role"]: card["AP"] for card in cards}
        for role, mean in means.items():
            if shown.get(role) != f"{100 * mean:.2f}":
                mismatches.append(f"benchmarks/{key} {role}: page {shown.get(role)} vs bundle {100 * mean:.2f}")
        gain = f"+{100 * (means['aware'] - means['naive']):.2f} AP points"
        for page in (f"benchmarks/{key}", f"verify/{key}"):
            actual = findings["pages"].get(page, {}).get("headline_gain")
            if actual != gain:
                mismatches.append(f"{page} headline: page {actual} vs bundle {gain}")
        gallery_rows = list(findings["pages"].get("predictions", {}).values())
        if not any(row == [shown.get(r) for r in ("naive", "aware", "complete_reference")]
                   for row in gallery_rows) and key in {"pawns-base", "all-pieces-base", "construction"}:
            mismatches.append(f"{key}: no prediction gallery matches the Benchmarks cards "
                              f"{[shown.get(r) for r in ('naive', 'aware', 'complete_reference')]}")
        landing_rows = findings["pages"].get("merge/landing", {}).get("rows", [])
        card_values = [shown.get(role) for role in ("naive", "aware", "complete_reference")]
        if not any(row == card_values for row in landing_rows):
            mismatches.append(f"{key}: no landing-table row matches the Benchmarks cards {card_values}")
        verify_means = findings["pages"].get(f"verify/{key}", {}).get("means", [])
        card_means = [shown.get(role) for role in ("naive", "aware", "complete_reference")]
        if verify_means != card_means:
            mismatches.append(f"{key}: Benchmarks {card_means} disagrees with Verify {verify_means}")
    findings["cross_page_mismatches"] = mismatches
    output.write_text(json.dumps(findings, indent=1, sort_keys=True))

    print(f"\nCross-page consistency (Benchmarks vs Verify): "
          f"{'OK, every headline matches the recomputed bundle' if not mismatches else 'MISMATCH'}")
    for line in mismatches:
        print(f"   ! {line}")
    print(f"\nFindings: {output.relative_to(ROOT)}")
    if findings["unverifiable"] or mismatches:
        raise SystemExit(f"{len(findings['unverifiable'])} unverifiable value(s), {len(mismatches)} mismatch(es)")
    print("PASSED: every score displayed on the site can be recomputed from the shipped bundle.")


if __name__ == "__main__":
    main()
