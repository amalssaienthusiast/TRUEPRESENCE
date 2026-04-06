"""
database.py  —  SQLite attendance database (zero-dependency, no Docker needed)

Drop-in replacement for the old PostgreSQL/psycopg2 implementation.
The database file is created automatically; no setup is required.

Thread safety:
    sqlite3 connections are opened per-call.  A module-level Lock serialises
    writes so concurrent threads never corrupt the database.

Resilience:
    Every public function is wrapped in try/except.  On failure it logs a
    warning and returns a safe fallback so the rest of the app keeps running.
"""

import datetime
import logging
import os
import sqlite3
import threading
from contextlib import contextmanager
from pathlib import Path
from typing import Generator

from dotenv import load_dotenv

load_dotenv()

logger = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# Configuration
# ---------------------------------------------------------------------------

# Path to the SQLite database file.  Override via DB_PATH env var.
DB_PATH: str = os.environ.get("DB_PATH", "data/attendance.db")

# Module-level write lock — prevents concurrent INSERT/UPDATE corruption.
_write_lock = threading.Lock()

# Track whether the schema has been initialised in this process.
_schema_ready: bool = False


# ---------------------------------------------------------------------------
# Connection helper
# ---------------------------------------------------------------------------

def _connect() -> sqlite3.Connection:
    """Open a new SQLite connection with sensible defaults.

    * Creates the parent directory if it does not exist.
    * Row factory = sqlite3.Row so callers can use column names.
    * WAL journal mode for safe concurrent reads.
    * Foreign keys enforced.
    """
    Path(DB_PATH).parent.mkdir(parents=True, exist_ok=True)
    conn = sqlite3.connect(DB_PATH, check_same_thread=False, timeout=10)
    conn.row_factory = sqlite3.Row
    conn.execute("PRAGMA journal_mode=WAL")
    conn.execute("PRAGMA foreign_keys=ON")
    return conn


@contextmanager
def _get_conn() -> Generator[sqlite3.Connection, None, None]:
    """Context manager: open a connection, commit on exit, rollback + log on error."""
    conn = _connect()
    try:
        yield conn
        conn.commit()
    except sqlite3.Error as exc:
        conn.rollback()
        logger.error("SQLite error: %s", exc)
        raise
    finally:
        conn.close()


# ---------------------------------------------------------------------------
# Schema
# ---------------------------------------------------------------------------

def init_db() -> bool:
    """Create the attendance table (idempotent).

    Returns True on success, False on failure (app continues either way).
    """
    global _schema_ready
    if _schema_ready:
        return True
    try:
        with _write_lock, _get_conn() as conn:
            conn.execute("""
                CREATE TABLE IF NOT EXISTS attendance (
                    id      INTEGER  PRIMARY KEY AUTOINCREMENT,
                    name    TEXT     NOT NULL,
                    time    TEXT     NOT NULL,
                    date    TEXT     NOT NULL,
                    status  TEXT     NOT NULL DEFAULT 'UNKNOWN',
                    UNIQUE(name, date)
                )
            """)
            # Index for fast date lookups
            conn.execute("""
                CREATE INDEX IF NOT EXISTS idx_attendance_date
                ON attendance(date DESC)
            """)
        _schema_ready = True
        logger.info("SQLite schema ready at %s", DB_PATH)
        return True
    except Exception as exc:
        logger.warning("init_db failed: %s  (DB features degraded)", exc)
        return False


# ---------------------------------------------------------------------------
# CRUD helpers
# ---------------------------------------------------------------------------

def record_attendance(name: str, status: str) -> str:
    """Insert or upgrade today's attendance record for *name*.

    Returns
    -------
    'inserted'  — new record created today
    'updated'   — existing record upgraded (non-VALID → VALID)
    'exists'    — record already present, no change needed
    'error'     — database error (logged; caller can degrade gracefully)
    """
    now      = datetime.datetime.now()
    date_str = now.strftime("%Y-%m-%d")
    time_str = now.strftime("%H:%M:%S")

    try:
        with _write_lock, _get_conn() as conn:
            row = conn.execute(
                "SELECT id, status FROM attendance WHERE name = ? AND date = ?",
                (name, date_str),
            ).fetchone()

            if row:
                rec_id, existing_status = row["id"], row["status"]
                if existing_status != "VALID" and status == "VALID":
                    conn.execute(
                        "UPDATE attendance SET time = ?, status = ? WHERE id = ?",
                        (time_str, status, rec_id),
                    )
                    logger.info("%s: UPDATED → %s at %s", name, status, time_str)
                    return "updated"
                logger.info("%s: already %s — no change", name, existing_status)
                return "exists"

            conn.execute(
                "INSERT INTO attendance (name, time, date, status) VALUES (?, ?, ?, ?)",
                (name, time_str, date_str, status),
            )
            logger.info("%s: INSERTED as %s at %s", name, status, time_str)
            return "inserted"

    except Exception as exc:
        logger.warning("record_attendance failed for %s: %s", name, exc)
        return "error"


def get_all_records() -> list[dict]:
    """Return all records ordered newest first.

    Returns an empty list on DB error (app keeps running).
    """
    try:
        with _get_conn() as conn:
            rows = conn.execute(
                "SELECT name, time, date, status FROM attendance ORDER BY date DESC, time DESC"
            ).fetchall()
            return [dict(r) for r in rows]
    except Exception as exc:
        logger.warning("get_all_records failed: %s", exc)
        return []


def get_records_by_date() -> dict[str, list[dict]]:
    """Return records grouped by date (newest date first).

    Returns an empty dict on DB error.
    """
    records = get_all_records()
    grouped: dict[str, list[dict]] = {}
    for rec in records:
        grouped.setdefault(rec["date"], []).append(rec)
    return grouped


def get_stats() -> dict:
    """Return quick summary statistics for the dashboard.

    Returns a dict with zero counts on DB error.
    """
    _default = {"total": 0, "today": 0, "valid": 0, "invalid": 0, "db_ok": False}
    try:
        today = datetime.datetime.now().strftime("%Y-%m-%d")
        with _get_conn() as conn:
            total   = conn.execute("SELECT COUNT(*) FROM attendance").fetchone()[0]
            today_n = conn.execute(
                "SELECT COUNT(*) FROM attendance WHERE date = ?", (today,)
            ).fetchone()[0]
            valid   = conn.execute(
                "SELECT COUNT(*) FROM attendance WHERE status = 'VALID'"
            ).fetchone()[0]
        return {
            "total":   total,
            "today":   today_n,
            "valid":   valid,
            "invalid": total - valid,
            "db_ok":   True,
        }
    except Exception as exc:
        logger.warning("get_stats failed: %s", exc)
        return _default


def export_csv_rows() -> list[list]:
    """Return [header] + data rows suitable for CSV export.

    Returns just the header on DB error.
    """
    header = [["Name", "Time", "Date", "Status"]]
    try:
        records = get_all_records()
        rows = [[r["name"], r["time"], r["date"], r["status"]] for r in records]
        return header + rows
    except Exception as exc:
        logger.warning("export_csv_rows failed: %s", exc)
        return header


# ---------------------------------------------------------------------------
# CLI smoke test
# ---------------------------------------------------------------------------

if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO)
    assert init_db(), "init_db() failed"
    r = record_attendance("SmokeTest_User", "VALID")
    print(f"record_attendance → {r}")
    recs = get_all_records()
    print(f"Total records: {len(recs)}")
    stats = get_stats()
    print(f"Stats: {stats}")
    print("✓ SQLite smoke test passed")
