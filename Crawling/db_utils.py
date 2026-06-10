import hashlib
import json
import os
import sqlite3
from datetime import datetime, timezone


BASE_DIR = os.path.dirname(os.path.abspath(__file__))
WEB_DIR = os.path.abspath(os.path.join(BASE_DIR, "../Web"))
DB_PATH = os.path.join(BASE_DIR, "crawler.db")


CATEGORY_ORDER = [
    "IT",
    "Energy",
    "Finance",
    "Healthcare",
    "Commodities",
    "Defense",
    "Chemicals",
    "Shipbuilding",
]


def utc_now():
    return datetime.now(timezone.utc).isoformat()


def normalize_text(value):
    return " ".join(str(value or "").split())


def content_hash(source, title, content, url=""):
    # Source is intentionally excluded so the same article from overlapping RSS
    # feeds is treated as one item.
    body = "\n".join(
        [
            normalize_text(title),
            normalize_text(content),
            normalize_text(url),
        ]
    )
    return hashlib.sha256(body.encode("utf-8", errors="ignore")).hexdigest()


def item_id_for(source, url, digest):
    raw = f"{source}|{url or digest}"
    return hashlib.sha1(raw.encode("utf-8", errors="ignore")).hexdigest()[:16]


def is_title_like_body(title, content):
    title_value = normalize_text(title).lower()
    content_value = normalize_text(content).lower()
    if not content_value:
        return True
    if not title_value:
        return False
    if len(content_value) <= len(title_value) + 20:
        return content_value in title_value or title_value in content_value
    return False


def body_quality_tuple(title, content, crawl_quality_score=0):
    body = normalize_text(content)
    score = float(crawl_quality_score or 0)
    return (score, len(body), 0 if is_title_like_body(title, body) else 1)


def should_preserve_existing_body(existing, incoming):
    existing_quality = body_quality_tuple(
        existing["title"],
        existing["content"],
        existing["crawl_quality_score"],
    )
    incoming_quality = body_quality_tuple(
        incoming.get("title", ""),
        incoming.get("content", ""),
        incoming.get("crawl_quality_score", 0),
    )
    existing_len = existing_quality[1]
    incoming_len = incoming_quality[1]
    if existing_len < 120:
        return False
    if incoming_quality[2] == 0 and existing_quality[2] == 1:
        return True
    if incoming_len + 120 < existing_len and incoming_quality[0] <= existing_quality[0]:
        return True
    if incoming_quality[0] + 0.2 < existing_quality[0] and incoming_len < existing_len:
        return True
    return False


def should_queue_for_tagging(item):
    source_group = str(item.get("source_group", ""))
    if source_group in {"truth", "x", "market", "test"}:
        return True
    title = item.get("title", "")
    content = item.get("content", "")
    source = str(item.get("source", ""))
    url = str(item.get("url", ""))
    status = str(item.get("crawl_status", ""))
    score = float(item.get("crawl_quality_score") or 0)
    content_len = len(normalize_text(content))
    if source.startswith("Gov_KoreaNet") and "articleId=" not in url and content_len < 300:
        return False
    if status == "no_body_title_only":
        return False
    if is_title_like_body(title, content):
        return False
    if score <= 0.2 and content_len < 80:
        return False
    return True


def connect(db_path=DB_PATH):
    os.makedirs(os.path.dirname(db_path), exist_ok=True)
    conn = sqlite3.connect(db_path)
    conn.row_factory = sqlite3.Row
    conn.execute("PRAGMA journal_mode=WAL")
    conn.execute("PRAGMA busy_timeout=30000")
    return conn


def init_db(db_path=DB_PATH):
    with connect(db_path) as conn:
        conn.executescript(
            """
            CREATE TABLE IF NOT EXISTS crawled_items (
                id TEXT PRIMARY KEY,
                content_hash TEXT NOT NULL UNIQUE,
                source TEXT NOT NULL,
                title TEXT DEFAULT '',
                content TEXT DEFAULT '',
                url TEXT DEFAULT '',
                raw_url TEXT DEFAULT '',
                published_at TEXT DEFAULT '',
                language TEXT DEFAULT '',
                country TEXT DEFAULT '',
                source_group TEXT DEFAULT '',
                created_at TEXT NOT NULL,
                updated_at TEXT NOT NULL
            );

            CREATE TABLE IF NOT EXISTS tagging_queue (
                item_id TEXT PRIMARY KEY,
                status TEXT NOT NULL DEFAULT 'pending',
                attempts INTEGER NOT NULL DEFAULT 0,
                last_error TEXT DEFAULT '',
                queued_at TEXT NOT NULL,
                started_at TEXT DEFAULT '',
                finished_at TEXT DEFAULT '',
                FOREIGN KEY(item_id) REFERENCES crawled_items(id)
            );

            CREATE TABLE IF NOT EXISTS tag_results (
                item_id TEXT PRIMARY KEY,
                content_hash TEXT NOT NULL,
                model_version TEXT NOT NULL,
                language TEXT DEFAULT '',
                tags_json TEXT NOT NULL DEFAULT '[]',
                primary_tag TEXT DEFAULT '',
                relevance_score REAL NOT NULL DEFAULT 0,
                sentiment_score REAL NOT NULL DEFAULT 0,
                sentiment_label TEXT DEFAULT 'Neutral',
                impact_type TEXT DEFAULT '중립',
                confidence REAL NOT NULL DEFAULT 0,
                matching_keywords_json TEXT NOT NULL DEFAULT '[]',
                excluded INTEGER NOT NULL DEFAULT 0,
                exclude_reason TEXT DEFAULT '',
                inference_ms INTEGER NOT NULL DEFAULT 0,
                cache_hit INTEGER NOT NULL DEFAULT 0,
                tagged_at TEXT NOT NULL,
                FOREIGN KEY(item_id) REFERENCES crawled_items(id)
            );

            CREATE TABLE IF NOT EXISTS tag_cache (
                content_hash TEXT NOT NULL,
                model_version TEXT NOT NULL,
                payload_json TEXT NOT NULL,
                created_at TEXT NOT NULL,
                PRIMARY KEY(content_hash, model_version)
            );

            CREATE TABLE IF NOT EXISTS label_feedback (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                item_id TEXT,
                content_hash TEXT,
                original_tag TEXT DEFAULT '',
                original_sentiment TEXT DEFAULT '',
                corrected_tag TEXT DEFAULT '',
                corrected_sentiment TEXT DEFAULT '',
                approved INTEGER NOT NULL DEFAULT 0,
                reviewer TEXT DEFAULT '',
                notes TEXT DEFAULT '',
                source TEXT DEFAULT '',
                created_at TEXT NOT NULL,
                used_for_training INTEGER NOT NULL DEFAULT 0,
                used_for_training_at TEXT DEFAULT ''
            );

            CREATE TABLE IF NOT EXISTS training_runs (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                status TEXT NOT NULL,
                sample_count INTEGER NOT NULL DEFAULT 0,
                validated_count INTEGER NOT NULL DEFAULT 0,
                auto_label_count INTEGER NOT NULL DEFAULT 0,
                candidate_count INTEGER NOT NULL DEFAULT 0,
                promoted_count INTEGER NOT NULL DEFAULT 0,
                baseline_metric REAL,
                candidate_metric REAL,
                promoted INTEGER NOT NULL DEFAULT 0,
                model_path TEXT DEFAULT '',
                model_version TEXT DEFAULT '',
                started_at TEXT NOT NULL,
                finished_at TEXT DEFAULT '',
                notes TEXT DEFAULT ''
            );

            CREATE TABLE IF NOT EXISTS entity_dictionary (
                entity_name TEXT PRIMARY KEY,
                category TEXT NOT NULL DEFAULT '',
                synonyms_json TEXT NOT NULL DEFAULT '[]',
                association_rules_json TEXT NOT NULL DEFAULT '[]',
                rationale TEXT DEFAULT '',
                estimated_market_impact TEXT DEFAULT 'MEDIUM',
                source TEXT DEFAULT 'builtin',
                updated_at TEXT NOT NULL
            );

            CREATE TABLE IF NOT EXISTS ner_filter_events (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                item_id TEXT,
                content_hash TEXT,
                decision TEXT NOT NULL,
                reason TEXT DEFAULT '',
                matched_entities_json TEXT NOT NULL DEFAULT '[]',
                matched_terms_json TEXT NOT NULL DEFAULT '[]',
                created_at TEXT NOT NULL
            );

            CREATE TABLE IF NOT EXISTS item_embeddings (
                item_id TEXT PRIMARY KEY,
                content_hash TEXT NOT NULL,
                model_version TEXT NOT NULL,
                text_hash TEXT NOT NULL,
                vector_json TEXT NOT NULL,
                created_at TEXT NOT NULL
            );

            CREATE TABLE IF NOT EXISTS dedup_groups (
                group_id TEXT PRIMARY KEY,
                representative_item_id TEXT NOT NULL,
                model_version TEXT NOT NULL,
                item_count INTEGER NOT NULL DEFAULT 1,
                created_at TEXT NOT NULL,
                updated_at TEXT NOT NULL
            );

            CREATE TABLE IF NOT EXISTS dedup_group_members (
                item_id TEXT PRIMARY KEY,
                group_id TEXT NOT NULL,
                representative_item_id TEXT NOT NULL,
                similarity REAL NOT NULL DEFAULT 1,
                is_representative INTEGER NOT NULL DEFAULT 0,
                created_at TEXT NOT NULL
            );

            CREATE TABLE IF NOT EXISTS dedup_candidate_pairs (
                left_item_id TEXT NOT NULL,
                right_item_id TEXT NOT NULL,
                model_version TEXT NOT NULL,
                similarity REAL NOT NULL,
                bge_similarity REAL NOT NULL DEFAULT 0,
                lsa_similarity REAL NOT NULL DEFAULT 0,
                title_similarity REAL NOT NULL DEFAULT 0,
                url_similarity REAL NOT NULL DEFAULT 0,
                source_match INTEGER NOT NULL DEFAULT 0,
                time_delta_hours REAL NOT NULL DEFAULT 0,
                composite_score REAL NOT NULL DEFAULT 0,
                audit_status TEXT NOT NULL DEFAULT 'pending',
                audit_model TEXT DEFAULT '',
                gpt_is_duplicate INTEGER,
                gpt_confidence REAL,
                gpt_rationale TEXT DEFAULT '',
                audited_at TEXT DEFAULT '',
                within_group INTEGER NOT NULL DEFAULT 0,
                created_at TEXT NOT NULL,
                PRIMARY KEY(left_item_id, right_item_id, model_version)
            );

            CREATE TABLE IF NOT EXISTS engine_state (
                key TEXT PRIMARY KEY,
                value TEXT DEFAULT '',
                updated_at TEXT NOT NULL
            );

            CREATE TABLE IF NOT EXISTS llm_excluded_reviews (
                item_id TEXT PRIMARY KEY,
                content_hash TEXT NOT NULL,
                model_id TEXT NOT NULL,
                decision TEXT NOT NULL,
                corrected_tag TEXT DEFAULT '',
                corrected_sentiment TEXT DEFAULT '',
                confidence REAL NOT NULL DEFAULT 0,
                rationale TEXT DEFAULT '',
                payload_json TEXT NOT NULL DEFAULT '{}',
                created_at TEXT NOT NULL
            );

            CREATE INDEX IF NOT EXISTS idx_crawled_source ON crawled_items(source);
            CREATE INDEX IF NOT EXISTS idx_crawled_published ON crawled_items(published_at);
            CREATE INDEX IF NOT EXISTS idx_queue_status ON tagging_queue(status);
            CREATE INDEX IF NOT EXISTS idx_tag_excluded ON tag_results(excluded);
            CREATE INDEX IF NOT EXISTS idx_ner_filter_item ON ner_filter_events(item_id);
            CREATE INDEX IF NOT EXISTS idx_dedup_members_group ON dedup_group_members(group_id);
            CREATE INDEX IF NOT EXISTS idx_dedup_candidates_model ON dedup_candidate_pairs(model_version, similarity);
            CREATE INDEX IF NOT EXISTS idx_llm_excluded_decision ON llm_excluded_reviews(decision);
            """
        )
        ensure_columns(
            conn,
            "crawled_items",
            {
                "content_origin": "TEXT DEFAULT ''",
                "crawl_status": "TEXT DEFAULT ''",
                "crawl_error": "TEXT DEFAULT ''",
                "crawl_quality_score": "REAL NOT NULL DEFAULT 0",
                "crawl_quality_json": "TEXT NOT NULL DEFAULT '{}'",
                "last_seen_at": "TEXT DEFAULT ''",
            },
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
        ensure_columns(
            conn,
            "dedup_candidate_pairs",
            {
                "bge_similarity": "REAL NOT NULL DEFAULT 0",
                "lsa_similarity": "REAL NOT NULL DEFAULT 0",
                "title_similarity": "REAL NOT NULL DEFAULT 0",
                "url_similarity": "REAL NOT NULL DEFAULT 0",
                "source_match": "INTEGER NOT NULL DEFAULT 0",
                "time_delta_hours": "REAL NOT NULL DEFAULT 0",
                "composite_score": "REAL NOT NULL DEFAULT 0",
                "audit_status": "TEXT NOT NULL DEFAULT 'pending'",
                "audit_model": "TEXT DEFAULT ''",
                "gpt_is_duplicate": "INTEGER",
                "gpt_confidence": "REAL",
                "gpt_rationale": "TEXT DEFAULT ''",
                "audited_at": "TEXT DEFAULT ''",
            },
        )


def ensure_columns(conn, table, columns):
    existing = {
        row["name"]
        for row in conn.execute(f"PRAGMA table_info({table})").fetchall()
    }
    for name, definition in columns.items():
        if name not in existing:
            conn.execute(f"ALTER TABLE {table} ADD COLUMN {name} {definition}")


def upsert_crawled_item(item, db_path=DB_PATH):
    init_db(db_path)
    source = item.get("source", "")
    title = item.get("title", "")
    content = item.get("content", "")
    url = item.get("url", "")
    digest = item.get("content_hash") or content_hash(source, title, content, url)
    item_id = item.get("id") or item_id_for(source, url, digest)
    now = utc_now()
    created = False
    updated = False
    with connect(db_path) as conn:
        existing = conn.execute(
            """
            SELECT id, content_hash, title, content, crawl_status, crawl_quality_score
              FROM crawled_items
             WHERE id = ?
                OR content_hash = ?
                OR (url <> '' AND url = ?)
             ORDER BY CASE
                 WHEN id = ? THEN 0
                 WHEN content_hash = ? THEN 1
                 ELSE 2
             END
             LIMIT 1
            """,
            (item_id, digest, url, item_id, digest),
        ).fetchone()
        queue_item = item
        if existing:
            item_id = existing["id"]
            if should_preserve_existing_body(existing, item):
                digest = existing["content_hash"]
                queue_item = {
                    "title": existing["title"],
                    "content": existing["content"],
                    "source_group": item.get("source_group", ""),
                    "crawl_status": existing["crawl_status"],
                    "crawl_quality_score": existing["crawl_quality_score"],
                }
                conn.execute(
                    """
                    UPDATE crawled_items
                       SET raw_url = ?, last_seen_at = ?
                     WHERE id = ?
                    """,
                    (item.get("raw_url", url), now, item_id),
                )
            else:
                updated = existing["content_hash"] != digest
                conn.execute(
                    """
                    UPDATE crawled_items
                       SET content_hash = ?, title = ?, content = ?, url = ?, raw_url = ?,
                           published_at = ?, language = ?, country = ?,
                           source_group = ?, content_origin = ?, crawl_status = ?,
                           crawl_error = ?, crawl_quality_score = ?,
                           crawl_quality_json = ?, updated_at = ?, last_seen_at = ?
                     WHERE id = ?
                    """,
                    (
                        digest,
                        title,
                        content,
                        url,
                        item.get("raw_url", url),
                        item.get("published_at", ""),
                        item.get("language", ""),
                        item.get("country", ""),
                        item.get("source_group", ""),
                        item.get("content_origin", ""),
                        item.get("crawl_status", ""),
                        item.get("crawl_error", ""),
                        float(item.get("crawl_quality_score") or 0),
                        item.get("crawl_quality_json", "{}"),
                        now,
                        now,
                        item_id,
                    ),
                )
        else:
            created = True
            conn.execute(
                """
                INSERT INTO crawled_items (
                    id, content_hash, source, title, content, url, raw_url,
                    published_at, language, country, source_group, content_origin,
                    crawl_status, crawl_error, crawl_quality_score, crawl_quality_json,
                    created_at, updated_at, last_seen_at
                ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
                """,
                (
                    item_id,
                    digest,
                    source,
                    title,
                    content,
                    url,
                    item.get("raw_url", url),
                    item.get("published_at", ""),
                    item.get("language", ""),
                    item.get("country", ""),
                    item.get("source_group", ""),
                    item.get("content_origin", ""),
                    item.get("crawl_status", ""),
                    item.get("crawl_error", ""),
                    float(item.get("crawl_quality_score") or 0),
                    item.get("crawl_quality_json", "{}"),
                    now,
                    now,
                    now,
                ),
            )
        cached = conn.execute(
            "SELECT 1 FROM tag_results WHERE item_id = ? AND content_hash = ?",
            (item_id, digest),
        ).fetchone()
        if not cached and should_queue_for_tagging(queue_item):
            conn.execute(
                """
                INSERT INTO tagging_queue (item_id, status, attempts, queued_at, started_at, finished_at, last_error)
                VALUES (?, 'pending', 0, ?, '', '', '')
                ON CONFLICT(item_id) DO UPDATE SET
                    status = 'pending',
                    attempts = 0,
                    queued_at = excluded.queued_at,
                    started_at = '',
                    finished_at = '',
                    last_error = ''
                """,
                (item_id, now),
            )
    return {"id": item_id, "content_hash": digest, "created": created, "updated": updated}


def add_feedback(
    item_id=None,
    content_hash=None,
    content_hash_value=None,
    original_tag="",
    original_sentiment="",
    corrected_tag="",
    corrected_sentiment="",
    approved=False,
    reviewer="",
    notes="",
    source="user",
):
    init_db()
    digest = content_hash if content_hash is not None else content_hash_value
    with connect() as conn:
        conn.execute(
            """
            INSERT INTO label_feedback (
                item_id, content_hash, original_tag, original_sentiment,
                corrected_tag, corrected_sentiment, approved, reviewer,
                notes, source, created_at
            ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
            """,
            (
                item_id,
                digest,
                original_tag,
                original_sentiment,
                corrected_tag,
                corrected_sentiment,
                1 if approved else 0,
                reviewer,
                notes,
                source,
                utc_now(),
            ),
        )
        return conn.execute("SELECT last_insert_rowid()").fetchone()[0]


def rows_to_dicts(rows):
    return [dict(row) for row in rows]


def json_dumps(data):
    return json.dumps(data, ensure_ascii=False, indent=2)
