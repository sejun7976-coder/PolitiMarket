"""Periodic validated keyword-tuning scheduler for real-time feedback labels.

This module prepares training datasets from reviewed feedback. It is careful by
default: without enough newly validated samples, or without an evaluation path
that beats the deployed baseline, it records a skipped run instead of promoting
BGE/NER/FinBERT keyword tuning artifacts.
"""

from __future__ import annotations

import argparse
import json
import os
import sqlite3
from datetime import datetime, timezone
from pathlib import Path
from typing import Any


BASE_DIR = Path(__file__).resolve().parents[1]
CRAWLING_DIR = BASE_DIR / "Crawling"
DB_PATH = CRAWLING_DIR / "crawler.db"
OUTPUT_DIR = Path(__file__).resolve().parent / "output"
TRAINING_DIR = Path(__file__).resolve().parent / "training_data"
TRAINING_STATUS = OUTPUT_DIR / "training_status.json"

MIN_VALIDATED_SAMPLES = int(os.environ.get("TRAIN_MIN_VALIDATED_SAMPLES", "100"))
MIN_HOURS_BETWEEN_RUNS = float(os.environ.get("TRAIN_MIN_HOURS_BETWEEN_RUNS", "6"))
AUTO_LABEL_CONFIDENCE = float(os.environ.get("TRAIN_AUTO_LABEL_CONFIDENCE", "0.9"))
BASELINE_METRIC = float(os.environ.get("TRAIN_BASELINE_METRIC", "0.0"))


def utc_now_dt() -> datetime:
    return datetime.now(timezone.utc)


def utc_now() -> str:
    return utc_now_dt().isoformat(timespec="seconds")


def connect() -> sqlite3.Connection:
    conn = sqlite3.connect(DB_PATH)
    conn.row_factory = sqlite3.Row
    conn.execute("PRAGMA journal_mode=WAL")
    return conn


def init_db(conn: sqlite3.Connection) -> None:
    conn.executescript(
        """
        CREATE TABLE IF NOT EXISTS label_feedback (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            item_id TEXT NOT NULL,
            content_hash TEXT NOT NULL,
            original_tag TEXT,
            original_sentiment TEXT,
            corrected_tag TEXT,
            corrected_sentiment TEXT,
            approved INTEGER NOT NULL DEFAULT 0,
            reviewer TEXT,
            notes TEXT,
            created_at TEXT NOT NULL,
            used_for_training INTEGER NOT NULL DEFAULT 0,
            used_for_training_at TEXT DEFAULT ''
        );

        CREATE TABLE IF NOT EXISTS training_runs (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            status TEXT NOT NULL,
            started_at TEXT NOT NULL,
            finished_at TEXT,
            sample_count INTEGER NOT NULL DEFAULT 0,
            validated_count INTEGER NOT NULL DEFAULT 0,
            auto_label_count INTEGER NOT NULL DEFAULT 0,
            baseline_metric REAL,
            candidate_metric REAL,
            promoted INTEGER NOT NULL DEFAULT 0,
            model_path TEXT,
            notes TEXT
        );
        """
    )
    ensure_columns(
        conn,
        "label_feedback",
        {
            "original_tag": "TEXT DEFAULT ''",
            "original_sentiment": "TEXT DEFAULT ''",
            "reviewer": "TEXT DEFAULT ''",
            "used_for_training": "INTEGER NOT NULL DEFAULT 0",
            "used_for_training_at": "TEXT DEFAULT ''",
        },
    )
    ensure_columns(
        conn,
        "training_runs",
        {
            "sample_count": "INTEGER NOT NULL DEFAULT 0",
            "validated_count": "INTEGER NOT NULL DEFAULT 0",
            "auto_label_count": "INTEGER NOT NULL DEFAULT 0",
            "promoted": "INTEGER NOT NULL DEFAULT 0",
            "model_path": "TEXT DEFAULT ''",
        },
    )
    conn.commit()


def ensure_columns(conn: sqlite3.Connection, table: str, columns: dict[str, str]) -> None:
    existing = {
        row["name"]
        for row in conn.execute(f"PRAGMA table_info({table})").fetchall()
    }
    for name, definition in columns.items():
        if name not in existing:
            conn.execute(f"ALTER TABLE {table} ADD COLUMN {name} {definition}")


def row_to_dict(row: sqlite3.Row) -> dict[str, Any]:
    return {key: row[key] for key in row.keys()}


def last_run(conn: sqlite3.Connection) -> sqlite3.Row | None:
    return conn.execute(
        "SELECT * FROM training_runs ORDER BY started_at DESC LIMIT 1"
    ).fetchone()


def hours_since_last_run(conn: sqlite3.Connection) -> float | None:
    row = last_run(conn)
    if not row:
        return None
    started_raw = row["started_at"]
    try:
        started = datetime.fromisoformat(started_raw.replace("Z", "+00:00"))
    except ValueError:
        return None
    return (utc_now_dt() - started).total_seconds() / 3600


def fetch_validated_feedback(conn: sqlite3.Connection) -> list[dict[str, Any]]:
    rows = conn.execute(
        """
        SELECT f.id AS feedback_id, f.*, c.title, c.content, c.source, c.language, c.country, c.raw_url
        FROM label_feedback f
        JOIN crawled_items c ON c.id = f.item_id
        WHERE f.approved = 1
          AND f.used_for_training = 0
          AND COALESCE(f.corrected_tag, '') <> ''
        ORDER BY f.created_at ASC
        """
    ).fetchall()
    return [row_to_dict(row) for row in rows]


def fetch_high_confidence_auto_labels(conn: sqlite3.Connection) -> list[dict[str, Any]]:
    rows = conn.execute(
        """
        SELECT c.id AS item_id, c.content_hash, c.title, c.content, c.source, c.language,
               c.country, c.raw_url, r.primary_tag AS corrected_tag,
               r.sentiment_label AS corrected_sentiment, r.confidence
        FROM tag_results r
        JOIN crawled_items c ON c.id = r.item_id
        LEFT JOIN label_feedback f ON f.content_hash = c.content_hash
        WHERE r.excluded = 0
          AND r.confidence >= ?
          AND f.id IS NULL
        GROUP BY c.content_hash
        ORDER BY r.tagged_at DESC
        LIMIT 300
        """,
        (AUTO_LABEL_CONFIDENCE,),
    ).fetchall()
    return [row_to_dict(row) | {"sample_weight": 0.25} for row in rows]


def make_sample(row: dict[str, Any], weight: float = 1.0) -> dict[str, Any]:
    return {
        "item_id": row["item_id"],
        "content_hash": row["content_hash"],
        "text": " ".join(part for part in [row.get("title"), row.get("content")] if part),
        "label": row.get("corrected_tag"),
        "sentiment": row.get("corrected_sentiment"),
        "source": row.get("source"),
        "language": row.get("language"),
        "country": row.get("country"),
        "raw_url": row.get("raw_url"),
        "sample_weight": float(row.get("sample_weight", weight)),
    }


def write_status(status: dict[str, Any]) -> None:
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    TRAINING_STATUS.write_text(json.dumps(status, ensure_ascii=False, indent=2), encoding="utf-8")


def insert_run(
    conn: sqlite3.Connection,
    status: str,
    started_at: str,
    sample_count: int,
    validated_count: int,
    auto_label_count: int,
    baseline_metric: float | None = None,
    candidate_metric: float | None = None,
    promoted: bool = False,
    model_path: str | None = None,
    notes: str | None = None,
) -> int:
    conn.execute(
        """
        INSERT INTO training_runs (
            status, started_at, finished_at, sample_count, validated_count,
            auto_label_count, baseline_metric, candidate_metric, promoted,
            model_path, notes
        )
        VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
        """,
        (
            status,
            started_at,
            utc_now(),
            sample_count,
            validated_count,
            auto_label_count,
            baseline_metric,
            candidate_metric,
            1 if promoted else 0,
            model_path,
            notes,
        ),
    )
    conn.commit()
    return int(conn.execute("SELECT last_insert_rowid() AS id").fetchone()["id"])


def export_jsonl(samples: list[dict[str, Any]]) -> Path:
    TRAINING_DIR.mkdir(parents=True, exist_ok=True)
    path = TRAINING_DIR / f"keyword_tuning_feedback_{utc_now_dt().strftime('%Y%m%d_%H%M%S')}.jsonl"
    with path.open("w", encoding="utf-8") as handle:
        for sample in samples:
            handle.write(json.dumps(sample, ensure_ascii=False) + "\n")
    return path


def export_keyword_tuning_artifacts(samples: list[dict[str, Any]], stem: str) -> dict[str, str]:
    TRAINING_DIR.mkdir(parents=True, exist_ok=True)
    artifacts = {
        "bge_category_keywords": TRAINING_DIR / f"bge_category_keywords_{stem}.jsonl",
        "ner_entity_keywords": TRAINING_DIR / f"ner_entity_keywords_{stem}.jsonl",
        "finbert_sentiment_keywords": TRAINING_DIR / f"finbert_sentiment_keywords_{stem}.jsonl",
    }
    with artifacts["bge_category_keywords"].open("w", encoding="utf-8") as handle:
        for sample in samples:
            handle.write(json.dumps({
                "text": sample.get("text"),
                "category": sample.get("label"),
                "source": sample.get("source"),
                "sample_weight": sample.get("sample_weight"),
            }, ensure_ascii=False) + "\n")
    with artifacts["ner_entity_keywords"].open("w", encoding="utf-8") as handle:
        for sample in samples:
            handle.write(json.dumps({
                "text": sample.get("text"),
                "category": sample.get("label"),
                "country": sample.get("country"),
                "source": sample.get("source"),
                "sample_weight": sample.get("sample_weight"),
            }, ensure_ascii=False) + "\n")
    with artifacts["finbert_sentiment_keywords"].open("w", encoding="utf-8") as handle:
        for sample in samples:
            handle.write(json.dumps({
                "text": sample.get("text"),
                "sentiment": sample.get("sentiment"),
                "source": sample.get("source"),
                "sample_weight": sample.get("sample_weight"),
            }, ensure_ascii=False) + "\n")
    return {key: str(path) for key, path in artifacts.items()}


def transformers_available() -> bool:
    try:
        import transformers  # noqa: F401
        import torch  # noqa: F401
    except Exception:
        return False
    return True


def mark_feedback_used(conn: sqlite3.Connection, rows: list[dict[str, Any]]) -> None:
    ids = [row["feedback_id"] for row in rows if row.get("feedback_id")]
    if not ids:
        return
    placeholders = ",".join("?" for _ in ids)
    conn.execute(
        f"UPDATE label_feedback SET used_for_training = 1, used_for_training_at = ? WHERE id IN ({placeholders})",
        [utc_now(), *ids],
    )
    conn.commit()


def run_scheduler(force: bool = False) -> dict[str, Any]:
    started = utc_now()
    if not DB_PATH.exists():
        status = {
            "status": "skipped",
            "started_at": started,
            "finished_at": utc_now(),
            "reason": "crawler.db not found",
        }
        write_status(status)
        return status

    conn = connect()
    init_db(conn)
    try:
        validated_rows = fetch_validated_feedback(conn)
        validated_count = len(validated_rows)
        hours_elapsed = hours_since_last_run(conn)
        enough_time = hours_elapsed is None or hours_elapsed >= MIN_HOURS_BETWEEN_RUNS
        enough_samples = validated_count >= MIN_VALIDATED_SAMPLES

        if not force and not (enough_time and enough_samples):
            reason = (
                f"conditions not met: validated={validated_count}/{MIN_VALIDATED_SAMPLES}, "
                f"hours_since_last_run={hours_elapsed if hours_elapsed is not None else 'none'}"
            )
            run_id = insert_run(
                conn,
                "skipped",
                started,
                0,
                validated_count,
                0,
                notes=reason,
            )
            status = {
                "run_id": run_id,
                "status": "skipped",
                "started_at": started,
                "finished_at": utc_now(),
                "reason": reason,
                "target_models": ["BGE category keyword profiles", "NER entity/rule dictionary", "FinBERT sentiment labels"],
                "validated_count": validated_count,
                "minimum_validated_samples": MIN_VALIDATED_SAMPLES,
                "minimum_hours_between_runs": MIN_HOURS_BETWEEN_RUNS,
            }
            write_status(status)
            return status

        auto_rows = fetch_high_confidence_auto_labels(conn)
        validated_samples = [make_sample(row, 1.0) for row in validated_rows]
        auto_samples = [make_sample(row, 0.25) for row in auto_rows]
        samples = validated_samples + auto_samples
        stem = utc_now_dt().strftime('%Y%m%d_%H%M%S')
        dataset_path = export_jsonl(samples)
        artifact_paths = export_keyword_tuning_artifacts(samples, stem)

        if not transformers_available():
            notes = (
                "dataset exported; transformers/torch runtime not installed, "
                "so model training and promotion were skipped"
            )
            run_id = insert_run(
                conn,
                "dataset_exported",
                started,
                len(samples),
                len(validated_samples),
                len(auto_samples),
                baseline_metric=BASELINE_METRIC,
                promoted=False,
                model_path=str(dataset_path),
                notes=notes,
            )
            mark_feedback_used(conn, validated_rows)
            status = {
                "run_id": run_id,
                "status": "dataset_exported",
                "started_at": started,
                "finished_at": utc_now(),
                "dataset_path": str(dataset_path),
                "artifact_paths": artifact_paths,
                "target_models": ["BGE category keyword profiles", "NER entity/rule dictionary", "FinBERT sentiment labels"],
                "sample_count": len(samples),
                "validated_count": len(validated_samples),
                "auto_label_count": len(auto_samples),
                "promoted": False,
                "reason": notes,
            }
            write_status(status)
            return status

        # The fine-tuning entry point is deliberately guarded. Real promotion
        # requires a held-out validation script to set candidate_metric.
        candidate_metric = None
        promoted = bool(candidate_metric is not None and candidate_metric >= BASELINE_METRIC)
        notes = (
            "keyword-tuning artifacts exported for BGE/NER/FinBERT; "
            "no validation harness configured, so runtime keyword/model promotion was not applied"
        )
        run_id = insert_run(
            conn,
            "not_promoted",
            started,
            len(samples),
            len(validated_samples),
            len(auto_samples),
            baseline_metric=BASELINE_METRIC,
            candidate_metric=candidate_metric,
            promoted=promoted,
            model_path=str(dataset_path),
            notes=notes,
        )
        mark_feedback_used(conn, validated_rows)
        status = {
            "run_id": run_id,
            "status": "not_promoted",
            "started_at": started,
            "finished_at": utc_now(),
            "dataset_path": str(dataset_path),
            "artifact_paths": artifact_paths,
            "target_models": ["BGE category keyword profiles", "NER entity/rule dictionary", "FinBERT sentiment labels"],
            "sample_count": len(samples),
            "validated_count": len(validated_samples),
            "auto_label_count": len(auto_samples),
            "baseline_metric": BASELINE_METRIC,
            "candidate_metric": candidate_metric,
            "promoted": promoted,
            "reason": notes,
        }
        write_status(status)
        return status
    finally:
        conn.close()


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--force", action="store_true", help="Ignore time/sample thresholds")
    args = parser.parse_args()
    status = run_scheduler(force=args.force)
    print(json.dumps(status, ensure_ascii=False, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
