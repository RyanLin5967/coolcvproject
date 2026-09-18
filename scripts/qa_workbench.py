"""Exercise the actual browser, worker process, server restart and cancellation."""
import json
import os
import subprocess
import time
import urllib.request
from pathlib import Path

from playwright.sync_api import sync_playwright

ROOT = Path(__file__).resolve().parents[1]
STATE = ROOT / "artifacts/qa/lifecycle"
STATE.mkdir(parents=True, exist_ok=True)
BASE = "http://127.0.0.1:8766"


def request(path, body=None):
    data = None if body is None else json.dumps(body).encode()
    req = urllib.request.Request(BASE+"/api"+path, data=data,
        headers={"Content-Type": "application/json", "X-CoverageCV": "workbench"})
    with urllib.request.urlopen(req, timeout=15) as response:
        return json.load(response)


def wait_for(fn, timeout=120):
    until = time.monotonic()+timeout
    while time.monotonic() < until:
        try:
            result = fn()
            if result:
                return result
        except (ConnectionError, OSError):
            pass
        time.sleep(.5)
    raise AssertionError("QA condition timed out")


log = (STATE / "server.log").open("ab")


def start():
    p = subprocess.Popen([str(ROOT / ".venv/bin/coveragecv"), "serve", "--port", "8766",
                          "--root", str(STATE), "--no-seed-demo"], cwd=ROOT, stdout=log, stderr=log)
    wait_for(lambda: request("/state"), 30)
    return p


server = start()
errors = []
evidence = {}
try:
    with sync_playwright() as p:
        browser = p.chromium.launch(executable_path="/Applications/Google Chrome.app/Contents/MacOS/Google Chrome", headless=True)
        page = browser.new_page(viewport={"width": 1440, "height": 1000})
        page.on("pageerror", lambda error: errors.append(str(error)))
        page.goto(BASE)
        page.locator("#import-button").click()
        page.locator("#import-name").fill("Lifecycle validation · warehouse")
        page.locator("#import-file").set_input_files(str(ROOT / "artifacts/qa/dataset.zip"))
        page.locator("#import-form button[type=submit]").click()
        project = wait_for(lambda: next((p for p in request("/state")["projects"]
                                       if p["name"] == "Lifecycle validation · warehouse"), None))
        rid = project["active_revision"]
        wait_for(lambda: request("/revisions/"+rid)["status"] == "ready")
        page.locator("[data-tab=policy]").click()
        page.locator('[data-policy="0:0"]').select_option("verified_absent")
        page.locator("#save-policy").click()
        bad = wait_for(lambda: (r if (r := request("/projects/"+project["id"])["active_revision"]) != rid else None))
        wait_for(lambda: request("/revisions/"+bad)["status"] == "failed")
        assert request("/revisions/"+rid)["status"] == "ready"
        fixed = request("/projects/"+project["id"]+"/policy", {"base_revision": bad,
            "changes": [{"source": "train", "class_name": "forklift", "state": "exhaustive"}]})
        wait_for(lambda: request("/revisions/"+fixed["id"])["status"] == "ready")
        evidence["zip_import_and_failed_policy_recovery"] = True
        short = request("/revisions/"+fixed["id"]+"/train", {"steps": 2, "seed": 42,
                        "batch": 2, "timeout_seconds": 120, "arms": ["aware"]})
        wait_for(lambda: request("/jobs/"+short["id"])["status"] == "completed", 180)
        evidence["actual_two_update_job_completed"] = True
        long = request("/revisions/"+fixed["id"]+"/train", {"steps": 1000, "seed": 42,
                        "batch": 2, "timeout_seconds": 600, "arms": ["aware"]})
        live = wait_for(lambda: (j if (j := request("/jobs/"+long["id"]))["live"].get("step", 0) >= 10 else None))
        old_pid, old_step = live["pid"], live["live"]["step"]
        server.terminate()
        server.wait(timeout=15)
        os.kill(old_pid, 0)
        server = start()
        resumed = wait_for(lambda: (j if (j := request("/jobs/"+long["id"]))["live"].get("step", 0) > old_step else None))
        assert resumed["pid"] == old_pid and resumed["status"] == "running"
        evidence["server_restart_preserved_worker_and_progress"] = True
        page.reload()
        page.locator('[data-page="experiments"]').click()
        page.locator(f'[data-cancel="{long["id"]}"]').click()
        wait_for(lambda: request("/jobs/"+long["id"])["status"] == "cancelled", 30)
        evidence["browser_cancel_stopped_worker"] = True
        page.screenshot(path=str(STATE / "cancelled.png"), full_page=True)
        evidence["page_errors"] = errors
        assert not errors
        browser.close()
    evidence["status"] = "passed"
    (STATE / "evidence.json").write_text(json.dumps(evidence, indent=2))
    print(json.dumps(evidence, indent=2), flush=True)
finally:
    server.terminate()
    server.wait(timeout=15)
    log.close()
