"""Analysis sessions: data sources, kernel, conversation history and pinned insights."""

from __future__ import annotations

import secrets
import shutil
import threading
from dataclasses import dataclass, field
from datetime import UTC, datetime, timedelta
from pathlib import Path
from typing import Any

from fastapi import HTTPException

from adsci.config import Settings
from adsci.data.base import DataSource
from adsci.data.files import FileDataSource

FILES_SOURCE = "files"


def _now() -> datetime:
    return datetime.now(UTC)


@dataclass
class Session:
    id: str
    dir: Path
    settings: Settings
    sources: dict[str, DataSource] = field(default_factory=dict)
    history: list[Any] = field(default_factory=list)
    insights: list[Any] = field(default_factory=list)
    questions: list[Any] = field(default_factory=list)
    model_id: str | None = None
    notebook: Any = None
    kernel: Any = None
    internal_token: str = field(default_factory=lambda: secrets.token_urlsafe(32))
    last_active: datetime = field(default_factory=_now)

    @property
    def files(self) -> FileDataSource:
        src = self.sources.get(FILES_SOURCE)
        if src is None:
            src = FileDataSource(
                FILES_SOURCE,
                self.dir / "files.duckdb",
                row_limit=self.settings.sql_row_limit,
                max_limit=self.settings.analysis_row_limit,
            )
            self.sources[FILES_SOURCE] = src
        return src  # type: ignore[return-value]

    def get_source(self, name: str) -> DataSource:
        try:
            return self.sources[name]
        except KeyError:
            available = ", ".join(sorted(self.sources)) or "none"
            raise KeyError(f"Unknown data source '{name}'. Available sources: {available}.") from None

    def touch(self) -> None:
        self.last_active = _now()

    def close(self) -> None:
        if self.kernel is not None:
            try:
                self.kernel.close_sync()
            except Exception:  # noqa: BLE001 - closing is best-effort
                pass
            self.kernel = None
        for src in self.sources.values():
            try:
                src.close()
            except Exception:  # noqa: BLE001 - closing is best-effort
                pass
        self.sources.clear()


class SessionManager:
    def __init__(self, settings: Settings) -> None:
        self.settings = settings
        self._sessions: dict[str, Session] = {}
        self._lock = threading.Lock()

    def create(self) -> Session:
        sid = secrets.token_hex(8)
        d = self.settings.data_dir / "sessions" / sid
        d.mkdir(parents=True, exist_ok=True)
        s = Session(id=sid, dir=d, settings=self.settings)
        with self._lock:
            self._sessions[sid] = s
        return s

    def get(self, sid: str) -> Session:
        with self._lock:
            s = self._sessions.get(sid)
        if s is None:
            raise HTTPException(status_code=404, detail="Session not found. Start a new session.")
        s.touch()
        return s

    def close(self, sid: str) -> None:
        with self._lock:
            s = self._sessions.pop(sid, None)
        if s is not None:
            s.close()
            shutil.rmtree(s.dir, ignore_errors=True)

    def close_all(self) -> None:
        for sid in list(self._sessions):
            self.close(sid)

    def reap_idle(self, now: datetime) -> int:
        cutoff = now - timedelta(minutes=self.settings.session_idle_minutes)
        with self._lock:
            stale = [sid for sid, s in self._sessions.items() if s.last_active < cutoff]
        for sid in stale:
            self.close(sid)
        return len(stale)
