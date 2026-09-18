"""SQLite is the source of truth; model files remain immutable filesystem artifacts."""
import json
import sqlite3
import time
import uuid
from contextlib import contextmanager
from pathlib import Path


def identifier():
    return uuid.uuid4().hex[:16]


class Store:
    def __init__(self, root: Path):
        self.root = root.resolve()
        self.root.mkdir(parents=True, exist_ok=True)
        self.path = self.root / "workbench.sqlite3"
        with self.connection() as db:
            db.executescript("""
                PRAGMA journal_mode=WAL;
                CREATE TABLE IF NOT EXISTS projects (
                    id TEXT PRIMARY KEY, name TEXT NOT NULL, created REAL NOT NULL,
                    reference_bundle TEXT, active_revision TEXT);
                CREATE TABLE IF NOT EXISTS revisions (
                    id TEXT PRIMARY KEY, project_id TEXT NOT NULL REFERENCES projects(id),
                    number INTEGER NOT NULL, spec_json TEXT NOT NULL, base_dir TEXT NOT NULL,
                    created REAL NOT NULL, status TEXT NOT NULL DEFAULT 'queued',
                    bundle TEXT, view TEXT, diagnostics_json TEXT, UNIQUE(project_id,number));
                CREATE TABLE IF NOT EXISTS jobs (
                    id TEXT PRIMARY KEY, project_id TEXT NOT NULL REFERENCES projects(id),
                    revision_id TEXT NOT NULL REFERENCES revisions(id), kind TEXT NOT NULL,
                    payload_json TEXT NOT NULL, status TEXT NOT NULL DEFAULT 'queued',
                    created REAL NOT NULL, started REAL, finished REAL, pid INTEGER,
                    result_json TEXT, error_json TEXT);
                CREATE TABLE IF NOT EXISTS events (
                    id INTEGER PRIMARY KEY AUTOINCREMENT, job_id TEXT, created REAL NOT NULL,
                    data_json TEXT NOT NULL);
                CREATE INDEX IF NOT EXISTS jobs_status ON jobs(status,created);
            """)

    @contextmanager
    def connection(self):
        db = sqlite3.connect(self.path, timeout=10)
        db.row_factory = sqlite3.Row
        db.execute("PRAGMA foreign_keys=ON")
        try:
            yield db
            db.commit()
        except BaseException:
            db.rollback()
            raise
        finally:
            db.close()

    @staticmethod
    def decode(row):
        if row is None:
            return None
        result = dict(row)
        for key in list(result):
            if key.endswith("_json"):
                value = result.pop(key)
                result[key[:-5]] = json.loads(value) if value is not None else None
        return result

    def get(self, table, identity):
        if table not in ("projects", "revisions", "jobs"):
            raise ValueError("invalid table")
        with self.connection() as db:
            return self.decode(db.execute(f"SELECT * FROM {table} WHERE id=?", (identity,)).fetchone())

    def projects(self):
        with self.connection() as db:
            return [self.decode(r) for r in db.execute("SELECT * FROM projects ORDER BY created DESC")]

    def revisions(self, project_id):
        with self.connection() as db:
            return [self.decode(r) for r in db.execute(
                "SELECT * FROM revisions WHERE project_id=? ORDER BY number DESC", (project_id,))]

    def jobs(self, project_id=None):
        with self.connection() as db:
            query = "SELECT * FROM jobs" + (" WHERE project_id=?" if project_id else "") + " ORDER BY created DESC"
            return [self.decode(r) for r in db.execute(query, (project_id,) if project_id else ())]

    def create_project(self, name, spec, base_dir, reference_bundle=None):
        pid = identifier()
        with self.connection() as db:
            db.execute("INSERT INTO projects(id,name,created,reference_bundle) VALUES(?,?,?,?)",
                       (pid, name, time.time(), reference_bundle))
        self.create_revision(pid, spec, base_dir)
        return self.get("projects", pid)

    def create_revision(self, project_id, spec, base_dir, expected_revision=None):
        rid, jid, now = identifier(), identifier(), time.time()
        with self.connection() as db:
            db.execute("BEGIN IMMEDIATE")
            project = db.execute("SELECT * FROM projects WHERE id=?", (project_id,)).fetchone()
            if project is None:
                raise KeyError(project_id)
            if expected_revision is not None and project["active_revision"] != expected_revision:
                raise ValueError("This project changed in another tab. Refresh before saving your policy.")
            number = db.execute("SELECT COALESCE(MAX(number),0)+1 FROM revisions WHERE project_id=?",
                                (project_id,)).fetchone()[0]
            db.execute("INSERT INTO revisions(id,project_id,number,spec_json,base_dir,created) VALUES(?,?,?,?,?,?)",
                       (rid, project_id, number, json.dumps(spec), str(base_dir), now))
            payload = {"spec": spec, "base_dir": str(base_dir)}
            db.execute("INSERT INTO jobs(id,project_id,revision_id,kind,payload_json,created) VALUES(?,?,?,?,?,?)",
                       (jid, project_id, rid, "compile", json.dumps(payload), now))
            db.execute("UPDATE projects SET active_revision=? WHERE id=?", (rid, project_id))
            self._event(db, jid, {"type": "queued", "project_id": project_id, "revision_id": rid})
        return self.get("revisions", rid)

    def create_job(self, project_id, revision_id, kind, payload):
        jid = identifier()
        with self.connection() as db:
            db.execute("INSERT INTO jobs(id,project_id,revision_id,kind,payload_json,created) VALUES(?,?,?,?,?,?)",
                       (jid, project_id, revision_id, kind, json.dumps(payload), time.time()))
            self._event(db, jid, {"type": "queued", "project_id": project_id})
        return self.get("jobs", jid)

    def claim(self):
        with self.connection() as db:
            db.execute("BEGIN IMMEDIATE")
            if db.execute("SELECT 1 FROM jobs WHERE status IN ('running','cancelling')").fetchone():
                return None
            row = db.execute("SELECT * FROM jobs WHERE status='queued' AND kind IN ('compile','train') ORDER BY created LIMIT 1").fetchone()
            if row is None:
                return None
            db.execute("UPDATE jobs SET status='running',started=? WHERE id=?", (time.time(), row["id"]))
            self._event(db, row["id"], {"type": "running", "project_id": row["project_id"]})
            return self.decode(db.execute("SELECT * FROM jobs WHERE id=?", (row["id"],)).fetchone())

    def set_pid(self, job_id, pid):
        with self.connection() as db:
            db.execute("UPDATE jobs SET pid=? WHERE id=? AND status IN ('running','cancelling')", (pid, job_id))

    def finish(self, job_id, result):
        status = result["status"]
        if status not in ("completed", "failed", "cancelled", "interrupted"):
            raise ValueError("invalid terminal job state")
        with self.connection() as db:
            row = db.execute("SELECT * FROM jobs WHERE id=?", (job_id,)).fetchone()
            if row["status"] in ("completed", "failed", "cancelled", "interrupted"):
                return
            db.execute("UPDATE jobs SET status=?,finished=?,result_json=?,error_json=? WHERE id=?",
                       (status, time.time(), json.dumps(result.get("result")), json.dumps(result.get("error")), job_id))
            if row["kind"] == "compile":
                compiled = result.get("result") or {}
                db.execute("UPDATE revisions SET status=?,bundle=?,view=?,diagnostics_json=? WHERE id=?",
                           ("ready" if status == "completed" else status, compiled.get("bundle"),
                            compiled.get("view"), json.dumps(compiled.get("diagnostics")), row["revision_id"]))
            self._event(db, job_id, {"type": status, "project_id": row["project_id"], "error": result.get("error")})

    def request_cancel(self, job_id):
        with self.connection() as db:
            row = db.execute("SELECT * FROM jobs WHERE id=?", (job_id,)).fetchone()
            if row is None:
                raise KeyError(job_id)
            if row["status"] == "queued":
                db.execute("UPDATE jobs SET status='cancelled',finished=? WHERE id=?", (time.time(), job_id))
                if row["kind"] == "compile":
                    db.execute("UPDATE revisions SET status='cancelled' WHERE id=?", (row["revision_id"],))
            elif row["status"] == "running":
                db.execute("UPDATE jobs SET status='cancelling' WHERE id=?", (job_id,))
            self._event(db, job_id, {"type": "cancel_requested", "project_id": row["project_id"]})
        return self.get("jobs", job_id)

    @staticmethod
    def _event(db, job_id, payload):
        db.execute("INSERT INTO events(job_id,created,data_json) VALUES(?,?,?)",
                   (job_id, time.time(), json.dumps(payload)))

    def events(self, after=0):
        with self.connection() as db:
            return [self.decode(r) for r in db.execute("SELECT * FROM events WHERE id>? ORDER BY id LIMIT 100", (after,))]

    def latest_event_id(self):
        with self.connection() as db:
            return db.execute("SELECT COALESCE(MAX(id),0) FROM events").fetchone()[0]
