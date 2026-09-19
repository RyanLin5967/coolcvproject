"""Drive the published Verify page in a real browser and confirm it recomputes the numbers.

This is the check behind the claim the page makes. It clicks the button a visitor
clicks, waits for every row to settle, and asserts the browser's recomputed AP
equals the published AP exactly -- not approximately -- for every run in a cohort.
"""
import json
import sys
from pathlib import Path

from playwright.sync_api import sync_playwright

ROOT = Path(__file__).resolve().parents[1]
BASE = sys.argv[1] if len(sys.argv) > 1 else "http://127.0.0.1:8767"
MANIFEST = json.loads((ROOT / "public-demo/static/verify/manifest.json").read_text())


def main():
    evidence = {"base": BASE, "cohorts": {}}
    with sync_playwright() as playwright:
        browser = playwright.chromium.launch()
        for width, height, label in ((1440, 900, "desktop"), (390, 844, "mobile")):
            page = browser.new_page(viewport={"width": width, "height": height})
            errors = []
            page.on("pageerror", lambda error, sink=errors: sink.append(str(error)))
            page.on("console",
                    lambda message, sink=errors: sink.append(message.text) if message.type == "error" else None)
            page.goto(BASE, wait_until="networkidle")
            page.click('[data-page="verify"]')
            page.wait_for_selector("#verify-run")

            for cohort in MANIFEST["cohorts"]:
                page.click(f'[data-verify-cohort="{cohort["key"]}"]')
                page.wait_for_selector("#verify-run:not([disabled])")
                page.click("#verify-run")
                page.wait_for_selector(".verify-banner.pass, .verify-banner.fail", timeout=180_000)
                page.wait_for_function(
                    "count => document.querySelectorAll('.verify-good, .verify-bad').length === count",
                    arg=len(cohort["runs"]), timeout=180_000)
                banner_class = page.get_attribute(".verify-banner", "class")
                identical = page.locator(".verify-good").count()
                published = [f"{100 * run['expected']['AP']:.4f}" for run in cohort["runs"]]
                shown = page.locator(".verify-table tbody tr td:nth-child(3) b").all_inner_texts()
                if "fail" in banner_class:
                    raise AssertionError(f"{label}/{cohort['key']}: the page reported a mismatch")
                if identical != len(cohort["runs"]):
                    raise AssertionError(f"{label}/{cohort['key']}: {identical}/{len(cohort['runs'])} rows verified")
                if shown != published:
                    raise AssertionError(f"{label}/{cohort['key']}: recomputed {shown} != published {published}")
                evidence["cohorts"].setdefault(cohort["key"], {})[label] = {
                    "runs": len(cohort["runs"]), "identical": identical, "recomputed_AP": shown}
                print(f"{label:8} {cohort['key']:20} {identical}/{len(cohort['runs'])} identical · {shown}")

            # The self-test is a claim the page makes, so exercise it: one deliberately
            # broken detection must turn the verdict red, and clearing it must restore green.
            page.click('[data-verify-cohort="pawns-base"]')
            page.wait_for_selector("#verify-run:not([disabled])")
            page.check("#verify-tamper")
            page.wait_for_selector(".verify-banner.fail", timeout=180_000)
            page.wait_for_function("() => !document.querySelector('#verify-run').disabled", timeout=180_000)
            if page.locator(".verify-bad").count() == 0:
                raise AssertionError(f"{label}: the self-test did not mark any row as differing")
            page.uncheck("#verify-tamper")
            page.wait_for_selector(".verify-banner.pass", timeout=180_000)
            page.wait_for_function(
                "count => document.querySelectorAll('.verify-good').length === count", arg=9, timeout=180_000)
            evidence[f"{label}_self_test"] = "red when tampered, green when restored"
            print(f"{label:8} self-test            goes red on one broken detection, green again when cleared")

            overflow = page.evaluate("document.scrollingElement.scrollWidth > document.scrollingElement.clientWidth + 1")
            if overflow:
                raise AssertionError(f"{label}: the page scrolls horizontally")
            if errors:
                raise AssertionError(f"{label}: browser reported {errors[:3]}")
            evidence[f"{label}_clean"] = True
            page.close()
        browser.close()
    output = ROOT / "artifacts/qa/verify-page/evidence.json"
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(json.dumps(evidence, indent=1, sort_keys=True))
    print(f"\nPASSED. Every cohort recomputed exactly at 1440px and 390px. Evidence: {output.relative_to(ROOT)}")


if __name__ == "__main__":
    main()
