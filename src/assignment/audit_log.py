"""
Assignment 11 — Audit Log starter (TODO).

Records every interaction for forensics. Never blocks by itself —
other layers catch attacks; this layer makes them reviewable.
"""
from __future__ import annotations

import json
import time
from datetime import datetime, timezone
from pathlib import Path


def default_audit_log_path() -> str:
    """Always resolve to <repo>/outputs/… (safe when cwd is src/)."""
    repo_root = Path(__file__).resolve().parents[2]
    return str(repo_root / "outputs" / "audit_log.json")


class AuditLogPlugin:
    """Framework-agnostic audit logger (wire into ADK callbacks or your pipeline)."""

    def __init__(self):
        self.name = "audit_log"
        self.logs: list[dict] = []
        self._open: dict[str, float] = {}

    def record_input(self, *, user_id: str, text: str, request_id: str | None = None):
        """Store input and start time; return the correlation id."""
        # With no explicit request id, user_id is the documented fallback so
        # record_input() and record_output() can still be paired by callers.
        correlation_id = request_id or user_id
        self._open[correlation_id] = {
            "request_id": correlation_id,
            "user_id": user_id,
            "input": text,
            "started_at": utc_now_iso(),
            "started_monotonic": time.perf_counter(),
        }
        return correlation_id

    def record_output(
        self,
        *,
        user_id: str,
        text: str,
        blocked: bool = False,
        layer: str | None = None,
        request_id: str | None = None,
    ):
        """Close an interaction and append a reviewable audit entry."""
        correlation_id = request_id or user_id
        opened = self._open.pop(correlation_id, None)
        now = utc_now_iso()
        latency_ms = None
        if opened and opened.get("started_monotonic") is not None:
            latency_ms = round((time.perf_counter() - opened["started_monotonic"]) * 1000, 3)
        entry = {
            "request_id": correlation_id,
            "user_id": user_id,
            "input": opened.get("input", "") if opened else "",
            "output": text,
            "blocked": bool(blocked),
            "layer": layer,
            "started_at": opened.get("started_at") if opened else now,
            "finished_at": now,
            "latency_ms": latency_ms,
        }
        self.logs.append(entry)
        return entry

    def export_json(self, filepath: str | None = None):
        """Write logs to disk (JSON array) under repo-root ``outputs/`` by default."""
        path = Path(filepath or default_audit_log_path())
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(
            json.dumps(self.logs, ensure_ascii=False, indent=2), encoding="utf-8"
        )
        return str(path)


def utc_now_iso() -> str:
    return datetime.now(timezone.utc).isoformat()
