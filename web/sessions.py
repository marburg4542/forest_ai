"""Per-browser session state.

The pipeline result is ~700 MB of numpy arrays that cannot be serialised, so it
has to live in the server process.  On a single-user desktop run one global
would do, but the moment the app is on a public URL that global becomes both a
bug and a privacy leak: whoever runs second overwrites the first user's result,
and the first user's page silently starts showing someone else's forest.

So results, jobs and uploaded files are all keyed by a session id, and the
number of live sessions is capped — two results already cost ~1.4 GB.

The session id arrives in an `X-Session-Id` header rather than a cookie.
Hugging Face Spaces can embed the page in an iframe on huggingface.co, where
third-party cookies may be blocked; a header the client sends explicitly is not
subject to that.  It is a privacy boundary between casual users, not a security
credential — anyone who guesses another id can read that session.
"""

from __future__ import annotations

import os
import re
import shutil
import threading
import time
import uuid
from collections import OrderedDict
from dataclasses import dataclass, field

SID_RE = re.compile(r"^[A-Za-z0-9_-]{8,64}$")

DATA_DIR = "data"
OUT_DIR = "outputs"


def env_int(name: str, default: int) -> int:
    try:
        return int(os.environ.get(name, default))
    except ValueError:
        return default


def env_bool(name: str, default: bool) -> bool:
    v = os.environ.get(name)
    return default if v is None else v.strip().lower() in ("1", "true", "yes", "on")


MAX_SESSIONS = env_int("FAI_MAX_SESSIONS", 2)
MAX_UPLOAD_MB = env_int("FAI_MAX_UPLOAD_MB", 300)
# on a shared deployment nobody should see the point clouds sitting next to the
# server; locally it is the whole convenience of the tool
ALLOW_LOCAL_CLOUDS = env_bool("FAI_ALLOW_LOCAL_CLOUDS", True)
ALLOW_SEGMENTED_LAS = env_bool("FAI_ALLOW_SEGMENTED_LAS", True)


def new_job() -> dict:
    return {"state": "idle", "progress": 0.0, "message": "", "error": None}


@dataclass
class Session:
    sid: str
    result: object | None = None          # forest_ai.pipeline.Result
    job: dict = field(default_factory=new_job)
    last_seen: float = field(default_factory=time.time)

    @property
    def upload_dir(self) -> str:
        return os.path.join(DATA_DIR, self.sid)

    @property
    def out_dir(self) -> str:
        return os.path.join(OUT_DIR, self.sid)

    def clouds(self) -> list[str]:
        """Point clouds this session is allowed to open."""
        out = []
        if ALLOW_LOCAL_CLOUDS and os.path.isdir("."):
            out += [f for f in sorted(os.listdir("."))
                    if f.lower().endswith((".las", ".laz"))]
        if os.path.isdir(self.upload_dir):
            out += [f"{self.upload_dir}/{f}".replace("\\", "/")
                    for f in sorted(os.listdir(self.upload_dir))
                    if f.lower().endswith((".las", ".laz"))]
        return out

    def dispose(self) -> None:
        """Free the result and delete anything this session put on disk."""
        self.result = None
        for d in (self.upload_dir, self.out_dir):
            shutil.rmtree(d, ignore_errors=True)


class SessionStore:
    """LRU-capped session table.  Safe to call from the request threads."""

    def __init__(self, max_sessions: int = MAX_SESSIONS):
        self.max = max(1, max_sessions)
        self._d: OrderedDict[str, Session] = OrderedDict()
        self._lock = threading.Lock()

    def get(self, sid: str | None) -> Session:
        """Fetch or create.  An unusable id is replaced with a fresh one."""
        if not sid or not SID_RE.match(sid):
            sid = uuid.uuid4().hex
        with self._lock:
            s = self._d.get(sid)
            if s is None:
                s = Session(sid=sid)
                self._d[sid] = s
            s.last_seen = time.time()
            self._d.move_to_end(sid)
            evicted = []
            while len(self._d) > self.max:
                # never evict a session whose job is still on the worker
                victim_sid = next(
                    (k for k, v in self._d.items()
                     if k != sid and v.job["state"] not in ("running", "queued")),
                    None)
                if victim_sid is None:
                    break
                evicted.append(self._d.pop(victim_sid))
            for v in evicted:
                v.dispose()
        return s

    def snapshot(self) -> list[Session]:
        with self._lock:
            return list(self._d.values())


STORE = SessionStore()
