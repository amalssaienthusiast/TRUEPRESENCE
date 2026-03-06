"""
database.py — PostgreSQL attendance database abstraction.

Provides connection pooling, schema initialization, and CRUD helpers
for the attendance system. Configuration is read from environment
variables (or a .env file via python-dotenv).
"""

import os
import datetime
import logging
from contextlib import contextmanager
from typing import Generator

import psycopg2
import psycopg2.extras
from psycopg2.pool import ThreadedConnectionPool
from dotenv import load_dotenv

load_dotenv()

logger = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# Connection configuration
# ---------------------------------------------------------------------------

DB_CONFIG: dict = {
    "host":     os.environ.get("DB_HOST", "localhost"),
    "port":     int(os.environ.get("DB_PORT", "5432")),
    "dbname":   os.environ.get("DB_NAME", "attendance_db"),
    "user":     os.environ.get("DB_USER", "attendance_user"),
    "password": os.environ.get("DB_PASSWORD", "attendance_pass"),
}

_pool: ThreadedConnectionPool | None = None


def _get_pool() -> ThreadedConnectionPool:
    global _pool
    if _pool is None or _pool.closed:
        _pool = ThreadedConnectionPool(1, 20, **DB_CONFIG)
        logger.info("PostgreSQL connection pool created (host=%s, db=%s)",
                    DB_CONFIG["host"], DB_CONFIG["dbname"])
    return _pool


@contextmanager
def get_connection() -> Generator:
    """Yield a psycopg2 connection from the pool; commit on exit or rollback on error."""
    pool = _get_pool()
    conn = pool.getconn()
    try:
        yield conn
        conn.commit()
    except Exception:
        conn.rollback()
        raise
    finally:
        pool.putconn(conn)


# ---------------------------------------------------------------------------
# Schema
# ---------------------------------------------------------------------------

def init_db() -> None:
    """Create the attendance table if it does not already exist."""
    with get_connection() as conn:
        with conn.cursor() as cur:
            cur.execute("""
                CREATE TABLE IF NOT EXISTS attendance (
                    id      SERIAL  PRIMARY KEY,
                    name    TEXT    NOT NULL,
                    time    TEXT    NOT NULL,
                    date    DATE    NOT NULL,
                    status  TEXT    NOT NULL DEFAULT 'UNKNOWN',
                    UNIQUE(name, date)
                )
            """)
    logger.info("Database schema ready.")


# ---------------------------------------------------------------------------
# CRUD helpers
# ---------------------------------------------------------------------------

def record_attendance(name: str, status: str) -> str:
    """
    Insert or update today's attendance record for *name*.

    Returns
    -------
    'inserted'  — new record created
    'updated'   — existing record upgraded from non-VALID → VALID
    'exists'    — record already present and not upgraded
    """
    now      = datetime.datetime.now()
    date_str = now.strftime("%Y-%m-%d")
    time_str = now.strftime("%H:%M:%S")

    with get_connection() as conn:
        with conn.cursor() as cur:
            cur.execute(
                "SELECT id, status FROM attendance WHERE name = %s AND date = %s",
                (name, date_str),
            )
            row = cur.fetchone()

            if row:
                rec_id, existing_status = row
                if existing_status != "VALID" and status == "VALID":
                    cur.execute(
                        "UPDATE attendance SET time = %s, status = %s WHERE id = %s",
                        (time_str, status, rec_id),
                    )
                    logger.info("%s: attendance UPDATED to %s at %s", name, status, time_str)
                    return "updated"
                logger.info("%s: already recorded as %s — no change", name, existing_status)
                return "exists"

            cur.execute(
                "INSERT INTO attendance (name, time, date, status) VALUES (%s, %s, %s, %s)",
                (name, time_str, date_str, status),
            )
            logger.info("%s: attendance INSERTED as %s at %s", name, status, time_str)
            return "inserted"


def get_all_records() -> list[dict]:
    """Return all attendance records ordered by date desc, time desc."""
    with get_connection() as conn:
        with conn.cursor(cursor_factory=psycopg2.extras.RealDictCursor) as cur:
            cur.execute("""
                SELECT name, time, date::text AS date, status
                FROM   attendance
                ORDER  BY date DESC, time DESC
            """)
            return [dict(row) for row in cur.fetchall()]


def get_records_by_date() -> dict[str, list[dict]]:
    """Return attendance records grouped by date (newest first)."""
    records = get_all_records()
    grouped: dict[str, list[dict]] = {}
    for rec in records:
        grouped.setdefault(rec["date"], []).append(rec)
    return grouped
