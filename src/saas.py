from __future__ import annotations

import hashlib
import secrets
import sqlite3
from datetime import datetime, timezone
from pathlib import Path

import pandas as pd


SCHEMA = """
CREATE TABLE IF NOT EXISTS workspaces (
    id INTEGER PRIMARY KEY AUTOINCREMENT,
    name TEXT NOT NULL,
    email TEXT NOT NULL UNIQUE,
    password_hash TEXT NOT NULL,
    plan TEXT NOT NULL DEFAULT 'Growth',
    created_at TEXT NOT NULL
);

CREATE TABLE IF NOT EXISTS scoring_runs (
    id INTEGER PRIMARY KEY AUTOINCREMENT,
    workspace_id INTEGER NOT NULL,
    run_type TEXT NOT NULL,
    rows_scored INTEGER NOT NULL,
    contacts_recommended INTEGER NOT NULL,
    avg_churn_probability REAL NOT NULL,
    threshold REAL NOT NULL,
    created_at TEXT NOT NULL,
    FOREIGN KEY (workspace_id) REFERENCES workspaces(id)
);

CREATE TABLE IF NOT EXISTS api_keys (
    id INTEGER PRIMARY KEY AUTOINCREMENT,
    workspace_id INTEGER NOT NULL,
    key_prefix TEXT NOT NULL,
    key_hash TEXT NOT NULL,
    created_at TEXT NOT NULL,
    revoked_at TEXT,
    FOREIGN KEY (workspace_id) REFERENCES workspaces(id)
);
"""


PLAN_LIMITS = {
    "Starter": {"monthly_rows": 500, "seats": 1, "price": 0},
    "Growth": {"monthly_rows": 5000, "seats": 3, "price": 49},
    "Scale": {"monthly_rows": 25000, "seats": 10, "price": 199},
}


def hash_password(password: str) -> str:
    return hashlib.sha256(password.encode("utf-8")).hexdigest()


def connect(db_path: Path) -> sqlite3.Connection:
    db_path.parent.mkdir(parents=True, exist_ok=True)
    connection = sqlite3.connect(db_path)
    connection.row_factory = sqlite3.Row
    return connection


def initialize_database(db_path: Path) -> None:
    with connect(db_path) as connection:
        connection.executescript(SCHEMA)
        seed_demo_workspace(connection)


def seed_demo_workspace(connection: sqlite3.Connection) -> None:
    existing = connection.execute(
        "SELECT id FROM workspaces WHERE email = ?",
        ("demo@churnai.com",),
    ).fetchone()
    if existing:
        return

    connection.execute(
        """
        INSERT INTO workspaces (name, email, password_hash, plan, created_at)
        VALUES (?, ?, ?, ?, ?)
        """,
        (
            "Demo Telecom Co.",
            "demo@churnai.com",
            hash_password("demo123"),
            "Growth",
            utc_now(),
        ),
    )


def utc_now() -> str:
    return datetime.now(timezone.utc).isoformat(timespec="seconds")


def create_workspace(db_path: Path, name: str, email: str, password: str) -> dict:
    with connect(db_path) as connection:
        connection.execute(
            """
            INSERT INTO workspaces (name, email, password_hash, plan, created_at)
            VALUES (?, ?, ?, ?, ?)
            """,
            (name.strip(), email.strip().lower(), hash_password(password), "Starter", utc_now()),
        )
        workspace = connection.execute(
            "SELECT id, name, email, plan, created_at FROM workspaces WHERE email = ?",
            (email.strip().lower(),),
        ).fetchone()
    return dict(workspace)


def fetch_workspace(db_path: Path, workspace_id: int) -> dict | None:
    with connect(db_path) as connection:
        workspace = connection.execute(
            "SELECT id, name, email, plan, created_at FROM workspaces WHERE id = ?",
            (workspace_id,),
        ).fetchone()
    return dict(workspace) if workspace else None


def authenticate_workspace(db_path: Path, email: str, password: str) -> dict | None:
    with connect(db_path) as connection:
        workspace = connection.execute(
            """
            SELECT id, name, email, plan, created_at
            FROM workspaces
            WHERE email = ? AND password_hash = ?
            """,
            (email.strip().lower(), hash_password(password)),
        ).fetchone()
    return dict(workspace) if workspace else None


def record_scoring_run(
    db_path: Path,
    workspace_id: int,
    run_type: str,
    scored_df: pd.DataFrame,
    threshold: float,
) -> None:
    contacts = int((scored_df["recommended_decision"] == "Contact").sum())
    avg_probability = float(scored_df["churn_probability"].mean())

    with connect(db_path) as connection:
        connection.execute(
            """
            INSERT INTO scoring_runs (
                workspace_id,
                run_type,
                rows_scored,
                contacts_recommended,
                avg_churn_probability,
                threshold,
                created_at
            )
            VALUES (?, ?, ?, ?, ?, ?, ?)
            """,
            (
                workspace_id,
                run_type,
                int(len(scored_df)),
                contacts,
                avg_probability,
                float(threshold),
                utc_now(),
            ),
        )


def update_workspace_plan(db_path: Path, workspace_id: int, plan: str) -> dict:
    if plan not in PLAN_LIMITS:
        raise ValueError(f"Unknown plan: {plan}")

    with connect(db_path) as connection:
        connection.execute(
            "UPDATE workspaces SET plan = ? WHERE id = ?",
            (plan, workspace_id),
        )
    workspace = fetch_workspace(db_path, workspace_id)
    if workspace is None:
        raise ValueError("Workspace not found.")
    return workspace


def fetch_scoring_runs(db_path: Path, workspace_id: int) -> pd.DataFrame:
    with connect(db_path) as connection:
        rows = connection.execute(
            """
            SELECT created_at, run_type, rows_scored, contacts_recommended,
                   avg_churn_probability, threshold
            FROM scoring_runs
            WHERE workspace_id = ?
            ORDER BY id DESC
            """,
            (workspace_id,),
        ).fetchall()

    return pd.DataFrame([dict(row) for row in rows])


def summarize_usage(runs_df: pd.DataFrame) -> dict[str, int | float]:
    if runs_df.empty:
        return {
            "runs": 0,
            "rows_scored": 0,
            "contacts_recommended": 0,
            "avg_churn_probability": 0.0,
        }

    return {
        "runs": int(len(runs_df)),
        "rows_scored": int(runs_df["rows_scored"].sum()),
        "contacts_recommended": int(runs_df["contacts_recommended"].sum()),
        "avg_churn_probability": float(runs_df["avg_churn_probability"].mean()),
    }


def get_plan_limit(plan: str) -> dict[str, int]:
    return PLAN_LIMITS.get(plan, PLAN_LIMITS["Starter"])


def can_score_rows(workspace: dict, runs_df: pd.DataFrame, incoming_rows: int) -> tuple[bool, str]:
    usage = summarize_usage(runs_df)
    limit = get_plan_limit(workspace["plan"])["monthly_rows"]
    projected_rows = int(usage["rows_scored"]) + int(incoming_rows)
    if projected_rows <= limit:
        return True, f"{projected_rows:,} of {limit:,} monthly rows used."
    return False, f"This run would use {projected_rows:,} rows, above the {limit:,}-row {workspace['plan']} plan limit."


def create_api_key(db_path: Path, workspace_id: int) -> str:
    raw_key = f"churn_live_{secrets.token_urlsafe(24)}"
    key_prefix = raw_key[:18]
    with connect(db_path) as connection:
        connection.execute(
            """
            INSERT INTO api_keys (workspace_id, key_prefix, key_hash, created_at)
            VALUES (?, ?, ?, ?)
            """,
            (workspace_id, key_prefix, hash_password(raw_key), utc_now()),
        )
    return raw_key


def fetch_api_keys(db_path: Path, workspace_id: int) -> pd.DataFrame:
    with connect(db_path) as connection:
        rows = connection.execute(
            """
            SELECT id, key_prefix, created_at, revoked_at
            FROM api_keys
            WHERE workspace_id = ?
            ORDER BY id DESC
            """,
            (workspace_id,),
        ).fetchall()
    return pd.DataFrame([dict(row) for row in rows])


def revoke_api_key(db_path: Path, workspace_id: int, api_key_id: int) -> None:
    with connect(db_path) as connection:
        connection.execute(
            """
            UPDATE api_keys
            SET revoked_at = ?
            WHERE id = ? AND workspace_id = ? AND revoked_at IS NULL
            """,
            (utc_now(), api_key_id, workspace_id),
        )


def authenticate_api_key(db_path: Path, raw_key: str) -> dict | None:
    key_prefix = raw_key[:18]
    with connect(db_path) as connection:
        row = connection.execute(
            """
            SELECT w.id, w.name, w.email, w.plan, w.created_at
            FROM api_keys k
            JOIN workspaces w ON w.id = k.workspace_id
            WHERE k.key_prefix = ? AND k.key_hash = ? AND k.revoked_at IS NULL
            """,
            (key_prefix, hash_password(raw_key)),
        ).fetchone()
    return dict(row) if row else None
