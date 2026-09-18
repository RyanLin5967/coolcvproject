"""A single local queue, process isolation, cancellation and restart reconciliation."""
import fcntl
import os
import signal
import subprocess
import sys
import threading
import time

import psutil

from coveragecv.artifacts import read_json, write_json


class Scheduler:
    def __init__(self, store):
        self.store = store
        self.stop = threading.Event()
        self.processes = {}
        self.cancel_sent = {}
        self.thread = None

    def start(self):
        self.lock = (self.store.root / "scheduler.lock").open("a")
        try:
            fcntl.flock(self.lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
        except BlockingIOError:
            self.lock.close()
            raise RuntimeError("Another workbench already owns this state directory") from None
        self.thread = threading.Thread(target=self.loop, name="coveragecv-queue", daemon=True)
        self.thread.start()

    def close(self):
        self.stop.set()
        if self.thread:
            self.thread.join(timeout=5)
        self.lock.close()
        # Workers deliberately survive server restarts. Use explicit Cancel to stop a run.

    def job_path(self, job):
        return self.store.root / "jobs" / job["id"] / "job.json"

    def alive(self, job):
        if job["id"] in self.processes:
            return self.processes[job["id"]].poll() is None
        if not job.get("pid"):
            return False
        try:
            process = psutil.Process(job["pid"])
            command = process.cmdline()
            return process.status() != psutil.STATUS_ZOMBIE and str(self.job_path(job)) in command and (
                "coveragecv.workbench.worker" in command)
        except (psutil.NoSuchProcess, psutil.AccessDenied):
            return False

    def tick(self):
        for job in self.store.jobs():
            if job["status"] not in ("running", "cancelling"):
                continue
            result_path = self.job_path(job).parent / "result.json"
            if result_path.exists():
                self.store.finish(job["id"], read_json(result_path))
                process = self.processes.pop(job["id"], None)
                if process:
                    process.wait(timeout=5)
                continue
            alive = self.alive(job)
            if job["status"] == "cancelling" and alive:
                sent = self.cancel_sent.get(job["id"])
                if sent is None:
                    os.killpg(job["pid"], signal.SIGTERM)
                    self.cancel_sent[job["id"]] = time.monotonic()
                elif time.monotonic()-sent > 10:
                    os.killpg(job["pid"], signal.SIGKILL)
            elif not alive and time.time()-(job["started"] or job["created"]) > 5:
                self.store.finish(job["id"], {"status": "cancelled" if job["status"] == "cancelling" else "interrupted",
                    "error": {"message": "The worker exited without a completed result. Start a new attempt."}})
        job = self.store.claim()
        if job:
            path = self.job_path(job)
            write_json(path, {**job, "root": str(self.store.root)})
            with (path.parent / "worker.log").open("ab") as log:
                process = subprocess.Popen([sys.executable, "-m", "coveragecv.workbench.worker", str(path)],
                    stdin=subprocess.DEVNULL, stdout=log, stderr=subprocess.STDOUT, start_new_session=True)
            self.processes[job["id"]] = process
            self.store.set_pid(job["id"], process.pid)

    def loop(self):
        while not self.stop.is_set():
            try:
                self.tick()
            except Exception as exc:  # noqa: BLE001 -- queue health is surfaced; no silent daemon death
                write_json(self.store.root / "scheduler_error.json", {"type": type(exc).__name__, "message": str(exc)})
            self.stop.wait(0.5)
