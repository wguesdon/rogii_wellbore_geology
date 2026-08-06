"""Track Kaggle kernel scores in a local SQLite database.

Kaggle authentication follows the standard Kaggle CLI configuration. This
script does not read credentials from the repository.
"""

from __future__ import annotations

import argparse
import sqlite3
from datetime import UTC, datetime
from pathlib import Path


PROJECT_ROOT = Path(__file__).resolve().parent.parent
DB_PATH = PROJECT_ROOT / "experiments.db"


def ensure_table() -> None:
    """Create the run tracking table when it does not already exist."""
    with sqlite3.connect(DB_PATH) as connection:
        connection.execute(
            """
            CREATE TABLE IF NOT EXISTS kaggle_runs (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                logged_at TEXT NOT NULL,
                kernel_slug TEXT NOT NULL,
                kernel_version INTEGER,
                cv_score REAL,
                public_score REAL,
                private_score REAL,
                submission_id INTEGER,
                notes TEXT
            )
            """
        )
        connection.execute(
            """
            CREATE UNIQUE INDEX IF NOT EXISTS ix_kaggle_run_kernel_version
            ON kaggle_runs (kernel_slug, COALESCE(kernel_version, -1))
            """
        )


def log_run(args: argparse.Namespace) -> None:
    """Insert or update one tracked Kaggle kernel run.

    Args:
        args: Parsed command line arguments for a run record.
    """
    ensure_table()
    logged_at = datetime.now(UTC).replace(microsecond=0).isoformat()
    values = (
        logged_at,
        args.kernel,
        args.version,
        args.cv,
        args.public_score,
        args.private_score,
        args.submission_id,
        args.notes,
    )
    with sqlite3.connect(DB_PATH) as connection:
        existing = connection.execute(
            """
            SELECT id FROM kaggle_runs
            WHERE kernel_slug = ?
              AND COALESCE(kernel_version, -1) = COALESCE(?, -1)
            """,
            (args.kernel, args.version),
        ).fetchone()
        if existing:
            connection.execute(
                """
                UPDATE kaggle_runs
                SET logged_at = ?, cv_score = COALESCE(?, cv_score),
                    public_score = COALESCE(?, public_score),
                    private_score = COALESCE(?, private_score),
                    submission_id = COALESCE(?, submission_id),
                    notes = COALESCE(?, notes)
                WHERE id = ?
                """,
                (*values[0:7], values[7], existing[0]),
            )
        else:
            connection.execute(
                """
                INSERT INTO kaggle_runs (
                    logged_at, kernel_slug, kernel_version, cv_score,
                    public_score, private_score, submission_id, notes
                ) VALUES (?, ?, ?, ?, ?, ?, ?, ?)
                """,
                values,
            )


def list_runs() -> None:
    """Print recorded Kaggle runs sorted by private then public score."""
    ensure_table()
    with sqlite3.connect(DB_PATH) as connection:
        rows = connection.execute(
            """
            SELECT kernel_slug, kernel_version, cv_score, public_score,
                   private_score, submission_id, notes
            FROM kaggle_runs
            ORDER BY COALESCE(private_score, public_score, 999.0), kernel_slug
            """
        ).fetchall()
    if not rows:
        print("No runs logged. Use the log subcommand to add one.")
        return
    print("kernel\tversion\tcv\tpublic\tprivate\tsubmission\tnotes")
    for row in rows:
        print("\t".join("" if value is None else str(value) for value in row))


def build_parser() -> argparse.ArgumentParser:
    """Build the command line parser.

    Returns:
        Configured parser for the run tracker.
    """
    parser = argparse.ArgumentParser(description=__doc__)
    commands = parser.add_subparsers(dest="command", required=True)
    log_parser = commands.add_parser("log", help="Insert or update one run.")
    log_parser.add_argument("--kernel", required=True)
    log_parser.add_argument("--version", type=int)
    log_parser.add_argument("--cv", type=float)
    log_parser.add_argument("--public-score", type=float)
    log_parser.add_argument("--private-score", type=float)
    log_parser.add_argument("--submission-id", type=int)
    log_parser.add_argument("--notes")
    commands.add_parser("list", help="List recorded runs.")
    return parser


def main() -> None:
    """Run the command line interface."""
    args = build_parser().parse_args()
    if args.command == "log":
        log_run(args)
    else:
        list_runs()


if __name__ == "__main__":
    main()
