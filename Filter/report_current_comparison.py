"""Create a report-ready comparison of current crawl tagging/sentiment results.

This report joins:
- the current production crawler/tagger results stored in crawler.db,
- the latest model comparison export,
- an independent Codex audit pass based on local TF-IDF/LSA embeddings plus
  transparent Korean/English market lexicons.

The accuracy values in this file are agreement/proxy metrics because no human
gold labels are stored in the database.
"""

from __future__ import annotations

import csv
import html
import json
import math
import os
import re
import sqlite3
import sys
from collections import Counter, defaultdict
from datetime import datetime, timedelta, timezone
from email.utils import parsedate_to_datetime
from pathlib import Path
from typing import Any
from zoneinfo import ZoneInfo

import numpy as np
from sklearn.decomposition import TruncatedSVD
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity
from sklearn.preprocessing import normalize


BASE_DIR = Path(__file__).resolve().parents[1]
DB_PATH = BASE_DIR / "Crawling" / "crawler.db"
EXPORTS_DIR = BASE_DIR / "Exports"


def env_int(name: str, default: str, minimum: int | None = None) -> int:
    try:
        value = int(os.environ.get(name, default))
    except Exception:
        value = int(default)
    if minimum is not None:
        value = max(minimum, value)
    return value


LOOKBACK_DAYS = env_int("CRAWLER_LOOKBACK_DAYS", "2", minimum=1)
LOOKBACK_ENABLED = os.environ.get("CRAWLER_LOOKBACK_DAYS", "2") != "0"
LOCAL_TIMEZONE_NAME = os.environ.get("CRAWLER_DAY_TIMEZONE", "Asia/Seoul")
try:
    LOCAL_TIMEZONE = ZoneInfo(LOCAL_TIMEZONE_NAME)
except Exception:
    LOCAL_TIMEZONE = ZoneInfo("Asia/Seoul")

CATEGORIES = [
    "IT",
    "Energy",
    "Finance",
    "Healthcare",
    "Commodities",
    "Defense",
    "Chemicals",
    "Shipbuilding",
]

CATEGORY_TERMS: dict[str, list[str]] = {
    "IT": [
        "ai", "artificial intelligence", "openai", "chatgpt", "codex", "llm",
        "semiconductor", "chip", "cloud", "data center", "cyber", "software",
        "technology", "nvidia", "robot", "인공지능", "반도체", "칩", "데이터센터",
        "클라우드", "사이버", "기술", "소프트웨어", "엔비디아", "로봇",
    ],
    "Energy": [
        "oil", "crude", "gas", "lng", "energy", "power", "electricity",
        "nuclear power", "solar", "wind power", "원유", "유가", "석유", "가스",
        "에너지", "전력", "전기", "원전", "태양광", "풍력", "발전",
    ],
    "Finance": [
        "market", "economy", "inflation", "rate", "bank", "treasury", "dollar",
        "finance", "stock", "bond", "tariff", "trade", "gdp", "tax", "export",
        "경제", "금융", "금리", "환율", "달러", "은행", "증시", "주식", "채권",
        "물가", "인플레이션", "관세", "무역", "세금", "수출", "수입",
    ],
    "Healthcare": [
        "health", "pharma", "vaccine", "hospital", "bio", "drug", "medical",
        "의료", "보건", "병원", "제약", "바이오", "백신", "건강", "신약",
    ],
    "Commodities": [
        "gold", "silver", "copper", "wheat", "steel", "commodity", "rare earth",
        "aluminum", "coal", "gold price", "silver price", "금값", "금 가격",
        "은값", "은 가격", "구리", "밀 가격", "철강", "원자재", "희토류",
        "광물", "알루미늄", "석탄", "곡물",
    ],
    "Defense": [
        "war", "weapon", "missile", "nuclear", "security", "military", "defense",
        "drone", "sanction", "conflict", "alliance", "전쟁", "무기", "미사일",
        "핵", "안보", "군사", "국방", "방위", "드론", "제재", "충돌", "동맹",
    ],
    "Chemicals": [
        "chemical", "battery", "petrochemical", "fertilizer", "lithium",
        "화학", "배터리", "석유화학", "비료", "리튬", "소재",
    ],
    "Shipbuilding": [
        "ship", "vessel", "shipping", "shipbuilding", "naval", "port",
        "조선", "선박", "해운", "항만", "lng선", "함정",
    ],
}

POSITIVE_TERMS = [
    "growth", "deal", "agreement", "investment", "cooperation", "support",
    "profit", "record", "expand", "recovery", "improve", "export", "rally",
    "surge", "rise", "gain", "strong demand", "relief", "cut tariffs",
    "성장", "상승", "증가", "투자", "협력", "합의", "지원", "회복", "개선",
    "확대", "호조", "수출", "기록", "강세", "완화", "인하", "급등", "수주",
    "실적", "흑자", "기회", "활성화", "개방",
]

NEGATIVE_TERMS = [
    "risk", "war", "sanction", "crisis", "fall", "decline", "attack",
    "conflict", "inflation", "default", "volatility", "tariff", "restriction",
    "uncertainty", "lawsuit", "fine", "breach", "violation", "block", "rebuke",
    "위험", "전쟁", "제재", "위기", "하락", "감소", "공격", "갈등", "충돌",
    "인플레이션", "관세", "규제", "불확실", "침해", "위반", "과징금", "차단",
    "반발", "압박", "긴장", "악재", "폭락", "파산", "경고", "불안", "논란",
]

STOPLIKE_TITLES = {"<p></p>", "", "none", "null"}


def now_stamp() -> str:
    return datetime.now().strftime("%Y%m%d_%H%M%S")


def safe_text(value: Any) -> str:
    return " ".join(str(value or "").replace("\x00", " ").split())


def parse_item_datetime(value: Any) -> datetime:
    if not value:
        return datetime.now(timezone.utc)
    raw = str(value).strip()
    try:
        parsed = parsedate_to_datetime(raw)
    except Exception:
        try:
            parsed = datetime.fromisoformat(raw.replace("Z", "+00:00"))
        except Exception:
            parsed = datetime.now(timezone.utc)
    if parsed.tzinfo is None:
        parsed = parsed.replace(tzinfo=timezone.utc)
    return parsed.astimezone(timezone.utc)


def parse_people_url_datetime(url: Any) -> datetime | None:
    match = re.search(r"/n3/(\d{4})/(\d{4})/", str(url or ""))
    if not match:
        return None
    year, month_day = match.groups()
    try:
        parsed = datetime(
            int(year),
            int(month_day[:2]),
            int(month_day[2:]),
            tzinfo=LOCAL_TIMEZONE,
        )
        return parsed.astimezone(timezone.utc)
    except Exception:
        return None


def lookback_start_local() -> datetime:
    start_date = datetime.now(LOCAL_TIMEZONE).date() - timedelta(days=LOOKBACK_DAYS - 1)
    return datetime.combine(start_date, datetime.min.time(), tzinfo=LOCAL_TIMEZONE)


def is_within_lookback_item(item: dict[str, Any]) -> bool:
    if not LOOKBACK_ENABLED:
        return True
    published = None
    if item.get("source") == "News_PeopleCN_KO":
        published = parse_people_url_datetime(item.get("url") or item.get("raw_url"))
    if published is None:
        published = parse_item_datetime(item.get("published_at") or item.get("created_at"))
    return published.astimezone(LOCAL_TIMEZONE) >= lookback_start_local()


def compact(value: Any, limit: int = 90) -> str:
    text = safe_text(value)
    return text if len(text) <= limit else text[: limit - 3].rstrip() + "..."


def write_csv(path: Path, rows: list[dict[str, Any]], fieldnames: list[str] | None = None) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    if fieldnames is None:
        fieldnames = list(rows[0].keys()) if rows else []
    with path.open("w", encoding="utf-8-sig", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)


def write_json(path: Path, data: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(data, ensure_ascii=False, indent=2), encoding="utf-8")


def latest_comparison_export() -> Path:
    candidates = [
        p for p in EXPORTS_DIR.glob("model_comparison_current_*")
        if p.is_dir() and (p / "sentiment" / "sentiment_predictions.csv").exists()
    ]
    if not candidates:
        raise FileNotFoundError("No model_comparison_current_* export folder found.")
    return max(candidates, key=lambda p: p.stat().st_mtime)


def load_items() -> list[dict[str, Any]]:
    conn = sqlite3.connect(DB_PATH)
    conn.row_factory = sqlite3.Row
    try:
        rows = conn.execute(
            """
            SELECT c.id AS item_id, c.content_hash, c.source, c.title, c.content, c.url,
                   c.raw_url, c.published_at, c.language, c.country, c.source_group,
                   c.created_at, c.updated_at, c.content_origin, c.crawl_status,
                   c.crawl_error, c.crawl_quality_score, q.status AS queue_status,
                   r.model_version, r.tags_json, r.primary_tag, r.relevance_score,
                   r.sentiment_score, r.sentiment_label, r.impact_type, r.confidence,
                   r.excluded, r.exclude_reason, r.tagged_at
            FROM crawled_items c
            LEFT JOIN tagging_queue q ON q.item_id = c.id
            LEFT JOIN tag_results r ON r.item_id = c.id
            WHERE COALESCE(c.source_group, '') <> 'test'
            ORDER BY COALESCE(c.published_at, c.created_at) DESC, c.created_at DESC
            """
        ).fetchall()
    finally:
        conn.close()

    items: list[dict[str, Any]] = []
    for row in rows:
        item = dict(row)
        if not is_within_lookback_item(item):
            continue
        title = safe_text(item.get("title"))
        content = safe_text(item.get("content"))
        if title.lower() in STOPLIKE_TITLES:
            title = ""
        model_text = content
        if title and title not in content[:240]:
            model_text = f"{title}. {content}"
        item["clean_title"] = title
        item["clean_content"] = content
        item["model_text"] = model_text[:2200]
        item["content_len"] = len(content)
        try:
            item["current_tags"] = json.loads(item.get("tags_json") or "[]")
        except Exception:
            item["current_tags"] = []
        items.append(item)
    return items


def term_hits(title: str, content: str, terms: list[str]) -> float:
    title_l = title.lower()
    content_l = content.lower()
    score = 0.0
    for term in terms:
        t = term.lower()
        if not t:
            continue
        title_count = count_term(title_l, t)
        content_count = count_term(content_l, t)
        if title_count:
            score += title_count * 2.5
        if content_count:
            score += content_count
    return score


def count_term(text: str, term: str) -> int:
    if re.fullmatch(r"[a-z0-9+#_.-]+", term):
        pattern = rf"(?<![a-z0-9]){re.escape(term)}(?![a-z0-9])"
        return len(re.findall(pattern, text))
    if len(term) <= 1:
        return 0
    return text.count(term)


def codex_audit_item(item: dict[str, Any]) -> dict[str, Any]:
    title = str(item.get("clean_title") or "")
    content = str(item.get("clean_content") or "")
    tag_scores = {
        category: term_hits(title, content, terms)
        for category, terms in CATEGORY_TERMS.items()
    }
    top_tag, top_score = max(tag_scores.items(), key=lambda row: row[1])
    sorted_tags = sorted(tag_scores.items(), key=lambda row: row[1], reverse=True)
    tags = [
        {
            "tag": tag,
            "raw_score": round(score, 3),
            "score": round(min(1.0, score / max(top_score, 1.0)), 4),
        }
        for tag, score in sorted_tags
        if score > 0
    ]
    primary_tag = top_tag if top_score >= 1.0 else "Unclear"
    relevance = round(min(1.0, top_score / 6.0), 4) if primary_tag != "Unclear" else 0.0

    pos = term_hits(title, content, POSITIVE_TERMS)
    neg = term_hits(title, content, NEGATIVE_TERMS)
    raw = pos - neg
    denom = max(pos + neg, 2.0)
    score = max(-1.0, min(1.0, raw / denom))
    if score <= -0.55:
        label = "Panic"
        label3 = "Negative"
    elif score <= -0.15:
        label = "Warning"
        label3 = "Negative"
    elif score >= 0.75:
        label = "Overheated"
        label3 = "Positive"
    elif score >= 0.15:
        label = "Positive"
        label3 = "Positive"
    else:
        label = "Neutral"
        label3 = "Neutral"

    content_len = int(item.get("content_len") or 0)
    excluded = primary_tag == "Unclear" or (relevance < 0.2 and content_len < 160)
    exclude_reason = ""
    if excluded:
        exclude_reason = "codex_low_market_relevance" if content_len >= 160 else "codex_short_or_low_relevance"

    return {
        "codex_primary_tag": primary_tag,
        "codex_relevance_score": relevance,
        "codex_tags_json": json.dumps(tags, ensure_ascii=False),
        "codex_sentiment_label": label,
        "codex_sentiment_3": label3,
        "codex_sentiment_score": round(score, 4),
        "codex_positive_hits": round(pos, 3),
        "codex_negative_hits": round(neg, 3),
        "codex_excluded": int(excluded),
        "codex_exclude_reason": exclude_reason,
    }


def sentiment3(label: Any, score: Any = None) -> str:
    value = str(label or "").strip().lower()
    if value in {"positive", "overheated", "optimistic"}:
        return "Positive"
    if value in {"panic", "warning", "negative"}:
        return "Negative"
    if value == "skipped":
        return "Skipped"
    try:
        numeric = float(score)
    except Exception:
        numeric = 0.0
    if numeric > 0.15:
        return "Positive"
    if numeric < -0.15:
        return "Negative"
    return "Neutral"


def read_comparison_predictions(export_dir: Path) -> dict[str, dict[str, dict[str, Any]]]:
    path = export_dir / "sentiment" / "sentiment_predictions.csv"
    by_item: dict[str, dict[str, dict[str, Any]]] = defaultdict(dict)
    with path.open("r", encoding="utf-8-sig", newline="") as f:
        for row in csv.DictReader(f):
            by_item[row["item_id"]][row["model_short"]] = row
    return by_item


def pick_language_finbert(item: dict[str, Any], preds: dict[str, dict[str, Any]]) -> tuple[str, str]:
    language = str(item.get("language") or "").lower()
    preferred = "kr_finbert_sc" if language.startswith("ko") else "prosus_finbert"
    row = preds.get(preferred)
    if row and row.get("canonical_label") not in {"", "Skipped"}:
        return sentiment3(row.get("canonical_label"), row.get("sentiment_score")), preferred
    for model in ("prosus_finbert", "kr_finbert_sc", "distilbert_sst2", "nlptown_multilingual_stars"):
        row = preds.get(model)
        if row and row.get("canonical_label") not in {"", "Skipped"}:
            return sentiment3(row.get("canonical_label"), row.get("sentiment_score")), model
    return "Skipped", ""


def pick_comparison_consensus(item: dict[str, Any], preds: dict[str, dict[str, Any]]) -> tuple[str, str, str]:
    lang_label, lang_model = pick_language_finbert(item, preds)
    keyword_row = preds.get("financial_keyword_baseline")
    keyword_label = (
        sentiment3(keyword_row.get("canonical_label"), keyword_row.get("sentiment_score"))
        if keyword_row else "Skipped"
    )
    labels = [
        sentiment3(row.get("canonical_label"), row.get("sentiment_score"))
        for row in preds.values()
        if row.get("canonical_label") not in {"", "Skipped"}
    ]
    counts = Counter(label for label in labels if label != "Skipped")
    if counts:
        label, count = counts.most_common(1)[0]
        tied = [name for name, value in counts.items() if value == count]
        if len(tied) == 1:
            return label, "model_majority", keyword_label
    if lang_label != "Skipped":
        return lang_label, lang_model, keyword_label
    if keyword_label != "Skipped":
        return keyword_label, "financial_keyword_baseline", keyword_label
    return "Neutral", "fallback_neutral", keyword_label


def build_embeddings(items: list[dict[str, Any]], out_dir: Path) -> tuple[list[dict[str, Any]], dict[str, Any]]:
    texts = [str(item.get("model_text") or " ") for item in items]
    emb_dir = out_dir / "codex_audit" / "embeddings"
    emb_dir.mkdir(parents=True, exist_ok=True)

    vectorizer = TfidfVectorizer(
        analyzer="word",
        ngram_range=(1, 2),
        min_df=1,
        max_df=0.93,
        lowercase=True,
    )
    sparse = vectorizer.fit_transform(texts).astype(np.float32)
    n_components = max(2, min(64, sparse.shape[0] - 1, sparse.shape[1] - 1))
    svd = TruncatedSVD(n_components=n_components, random_state=42)
    dense = svd.fit_transform(sparse).astype(np.float32)
    dense = normalize(dense, norm="l2").astype(np.float32)
    sim = cosine_similarity(dense)
    np.save(emb_dir / "codex_lsa_embeddings.npy", dense)

    item_ids = [item["item_id"] for item in items]
    neighbors: list[dict[str, Any]] = []
    top_pairs: list[dict[str, Any]] = []
    for i, item in enumerate(items):
        scores = sim[i].copy()
        scores[i] = -2
        j = int(np.argmax(scores))
        neighbors.append({
            "item_id": item["item_id"],
            "nearest_item_id": item_ids[j],
            "similarity": round(float(scores[j]), 6),
            "source": item.get("source"),
            "nearest_source": items[j].get("source"),
            "source_group": item.get("source_group"),
            "nearest_source_group": items[j].get("source_group"),
            "title": item.get("clean_title"),
            "nearest_title": items[j].get("clean_title"),
        })
    for i in range(len(items)):
        for j in range(i + 1, len(items)):
            top_pairs.append({
                "item_id_a": item_ids[i],
                "item_id_b": item_ids[j],
                "similarity": round(float(sim[i, j]), 6),
                "source_a": items[i].get("source"),
                "source_b": items[j].get("source"),
                "title_a": items[i].get("clean_title"),
                "title_b": items[j].get("clean_title"),
            })
    top_pairs.sort(key=lambda row: row["similarity"], reverse=True)
    write_csv(emb_dir / "codex_nearest_neighbors.csv", neighbors)
    write_csv(emb_dir / "codex_top_similarity_pairs.csv", top_pairs[:80])
    metrics = {
        "embedding_model": "codex_tfidf_lsa_word_1_2",
        "item_count": len(items),
        "embedding_dim": int(dense.shape[1]),
        "vocabulary_size": len(vectorizer.vocabulary_),
        "explained_variance_ratio_sum": round(float(np.sum(svd.explained_variance_ratio_)), 6),
        "avg_nearest_neighbor_similarity": round(
            sum(row["similarity"] for row in neighbors) / len(neighbors), 6
        ) if neighbors else 0,
        "pairs_ge_0_90": sum(1 for row in top_pairs if row["similarity"] >= 0.90),
        "pairs_ge_0_80": sum(1 for row in top_pairs if row["similarity"] >= 0.80),
        "pairs_ge_0_70": sum(1 for row in top_pairs if row["similarity"] >= 0.70),
    }
    write_json(emb_dir / "codex_embedding_manifest.json", metrics)
    return neighbors, metrics


def pairwise_agreement(rows: list[dict[str, Any]], model_cols: dict[str, str]) -> list[dict[str, Any]]:
    out: list[dict[str, Any]] = []
    models = list(model_cols)
    for a in models:
        for b in models:
            common = [
                row for row in rows
                if row.get(model_cols[a]) not in {"", "Skipped"} and row.get(model_cols[b]) not in {"", "Skipped"}
            ]
            matches = sum(1 for row in common if row.get(model_cols[a]) == row.get(model_cols[b]))
            out.append({
                "model_a": a,
                "model_b": b,
                "common_items": len(common),
                "matches": matches,
                "agreement_rate": round(matches / len(common), 6) if common else "",
            })
    return out


def consensus_proxy_accuracy(rows: list[dict[str, Any]], model_cols: dict[str, str]) -> list[dict[str, Any]]:
    results: list[dict[str, Any]] = []
    for model, col in model_cols.items():
        compared = 0
        matched = 0
        for row in rows:
            own = row.get(col)
            if own in {"", "Skipped"}:
                continue
            other_labels = [
                row.get(other_col) for other_model, other_col in model_cols.items()
                if other_model != model and row.get(other_col) not in {"", "Skipped"}
            ]
            counts = Counter(other_labels)
            if not counts:
                continue
            label, count = counts.most_common(1)[0]
            if sum(1 for value in counts.values() if value == count) > 1:
                continue
            compared += 1
            if own == label:
                matched += 1
        results.append({
            "metric": "sentiment_consensus_proxy_accuracy",
            "model": model,
            "compared_items": compared,
            "matches": matched,
            "rate": round(matched / compared, 6) if compared else "",
            "note": "Agreement with the majority label of the other available models; not human-label accuracy.",
        })
    return results


def count_rows(rows: list[dict[str, Any]], column: str, order: list[str] | None = None) -> list[dict[str, Any]]:
    counts = Counter(str(row.get(column) or "") for row in rows)
    keys = order or sorted(counts)
    return [{"label": key, "count": counts.get(key, 0)} for key in keys if counts.get(key, 0)]


def escape_xml(value: Any) -> str:
    return html.escape(str(value or ""), quote=True)


def svg_grouped_bars(path: Path, title: str, rows: list[dict[str, Any]], series: list[str]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    width = 980
    left = 180
    top = 56
    row_h = 38
    height = top + max(1, len(rows)) * row_h + 42
    colors = ["#2563eb", "#16a34a", "#dc2626", "#7c3aed", "#f59e0b"]
    max_value = max([float(row.get(name, 0) or 0) for row in rows for name in series] + [1])
    parts = [
        f'<svg xmlns="http://www.w3.org/2000/svg" width="{width}" height="{height}" viewBox="0 0 {width} {height}">',
        '<rect width="100%" height="100%" fill="#ffffff"/>',
        f'<text x="24" y="30" font-family="Inter, Arial" font-size="18" font-weight="700" fill="#111827">{escape_xml(title)}</text>',
    ]
    for idx, name in enumerate(series):
        x = left + idx * 140
        parts.append(f'<rect x="{x}" y="16" width="12" height="12" rx="2" fill="{colors[idx % len(colors)]}"/>')
        parts.append(f'<text x="{x + 18}" y="27" font-family="Inter, Arial" font-size="11" fill="#475467">{escape_xml(name)}</text>')
    for i, row in enumerate(rows):
        y = top + i * row_h
        parts.append(f'<text x="24" y="{y + 20}" font-family="Inter, Arial" font-size="12" font-weight="600" fill="#344054">{escape_xml(row.get("label"))}</text>')
        for j, name in enumerate(series):
            value = float(row.get(name, 0) or 0)
            bar_w = value / max_value * 600
            by = y + j * 8
            parts.append(f'<rect x="{left}" y="{by}" width="{bar_w:.2f}" height="6" rx="3" fill="{colors[j % len(colors)]}"/>')
            parts.append(f'<text x="{left + bar_w + 6:.2f}" y="{by + 6}" font-family="Inter, Arial" font-size="9" fill="#475467">{int(value)}</text>')
    parts.append("</svg>")
    path.write_text("\n".join(parts), encoding="utf-8")


def svg_rate_bars(path: Path, title: str, rows: list[dict[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    width = 900
    left = 270
    top = 54
    row_h = 34
    height = top + max(1, len(rows)) * row_h + 32
    parts = [
        f'<svg xmlns="http://www.w3.org/2000/svg" width="{width}" height="{height}" viewBox="0 0 {width} {height}">',
        '<rect width="100%" height="100%" fill="#ffffff"/>',
        f'<text x="24" y="30" font-family="Inter, Arial" font-size="18" font-weight="700" fill="#111827">{escape_xml(title)}</text>',
    ]
    for i, row in enumerate(rows):
        y = top + i * row_h
        rate = row.get("rate")
        try:
            value = float(rate)
        except Exception:
            value = 0.0
        label = row.get("model") or row.get("metric")
        parts.append(f'<text x="24" y="{y + 17}" font-family="Inter, Arial" font-size="12" font-weight="600" fill="#344054">{escape_xml(label)}</text>')
        parts.append(f'<rect x="{left}" y="{y}" width="560" height="16" rx="8" fill="#eef2f7"/>')
        parts.append(f'<rect x="{left}" y="{y}" width="{560 * value:.2f}" height="16" rx="8" fill="#2563eb"/>')
        parts.append(f'<text x="{left + 570}" y="{y + 13}" font-family="Inter, Arial" font-size="11" fill="#111827">{value * 100:.1f}%</text>')
    parts.append("</svg>")
    path.write_text("\n".join(parts), encoding="utf-8")


def svg_heatmap(path: Path, title: str, rows: list[dict[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    models = sorted({row["model_a"] for row in rows} | {row["model_b"] for row in rows})
    cell = 86
    left = 190
    top = 92
    width = left + cell * len(models) + 40
    height = top + cell * len(models) + 40
    by_pair = {(row["model_a"], row["model_b"]): row for row in rows}
    parts = [
        f'<svg xmlns="http://www.w3.org/2000/svg" width="{width}" height="{height}" viewBox="0 0 {width} {height}">',
        '<rect width="100%" height="100%" fill="#ffffff"/>',
        f'<text x="24" y="30" font-family="Inter, Arial" font-size="18" font-weight="700" fill="#111827">{escape_xml(title)}</text>',
    ]
    for idx, model in enumerate(models):
        x = left + idx * cell
        parts.append(f'<text x="{x + 6}" y="72" font-family="Inter, Arial" font-size="10" fill="#344054" transform="rotate(-25 {x + 6},72)">{escape_xml(model)}</text>')
        parts.append(f'<text x="24" y="{top + idx * cell + 48}" font-family="Inter, Arial" font-size="11" font-weight="600" fill="#344054">{escape_xml(model)}</text>')
    for y_idx, a in enumerate(models):
        for x_idx, b in enumerate(models):
            row = by_pair.get((a, b), {})
            try:
                rate = float(row.get("agreement_rate", 0) or 0)
            except Exception:
                rate = 0.0
            blue = int(240 - 120 * rate)
            fill = f"rgb({blue},{blue + 8},255)"
            x = left + x_idx * cell
            y = top + y_idx * cell
            parts.append(f'<rect x="{x}" y="{y}" width="{cell - 6}" height="{cell - 6}" rx="8" fill="{fill}" stroke="#d0d5dd"/>')
            parts.append(f'<text x="{x + 24}" y="{y + 45}" font-family="Inter, Arial" font-size="13" font-weight="700" fill="#111827">{rate:.2f}</text>')
    parts.append("</svg>")
    path.write_text("\n".join(parts), encoding="utf-8")


def markdown_table(rows: list[dict[str, Any]], columns: list[str], max_rows: int = 12) -> str:
    visible = rows[:max_rows]
    lines = ["| " + " | ".join(columns) + " |", "| " + " | ".join("---" for _ in columns) + " |"]
    for row in visible:
        values = [str(row.get(col, "")).replace("|", "/") for col in columns]
        lines.append("| " + " | ".join(values) + " |")
    return "\n".join(lines)


def create_report(
    out_dir: Path,
    comparison_dir: Path,
    rows: list[dict[str, Any]],
    summary_metrics: list[dict[str, Any]],
    proxy_rows: list[dict[str, Any]],
    embedding_metrics: dict[str, Any],
    divergence_rows: list[dict[str, Any]],
) -> None:
    report_path = out_dir / "report.md"
    source_rows = count_rows(rows, "source_group")
    current_dist = count_rows(rows, "current_sentiment_3", ["Positive", "Neutral", "Negative"])
    codex_dist = count_rows(rows, "codex_sentiment_3", ["Positive", "Neutral", "Negative"])
    comparison_dist = count_rows(rows, "comparison_sentiment_3", ["Positive", "Neutral", "Negative"])

    dist_rows: list[dict[str, Any]] = []
    for label in ["Positive", "Neutral", "Negative"]:
        dist_rows.append({
            "label": label,
            "current_model": next((r["count"] for r in current_dist if r["label"] == label), 0),
            "comparison_model": next((r["count"] for r in comparison_dist if r["label"] == label), 0),
            "codex_audit": next((r["count"] for r in codex_dist if r["label"] == label), 0),
        })

    tag_rows: list[dict[str, Any]] = []
    for tag in CATEGORIES + ["Unclear"]:
        tag_rows.append({
            "label": tag,
            "current_model": sum(1 for row in rows if row.get("current_tag") == tag),
            "codex_audit": sum(1 for row in rows if row.get("codex_primary_tag") == tag),
        })
    tag_rows = [row for row in tag_rows if row["current_model"] or row["codex_audit"]]

    write_csv(out_dir / "tables" / "sentiment_distribution_report.csv", dist_rows)
    write_csv(out_dir / "tables" / "tag_distribution_report.csv", tag_rows)
    write_csv(out_dir / "tables" / "source_distribution_report.csv", source_rows)

    lines = [
        "# 현재 크롤링 데이터 비교분석 보고서",
        "",
        f"- 생성 시각: `{datetime.now().isoformat(timespec='seconds')}`",
        f"- 현재 DB: `{DB_PATH}`",
        f"- 사용한 비교분석 export: `{comparison_dir.name}`",
        f"- 분석 대상 운영 데이터: `{len(rows)}`건",
        "",
        "## 해석 주의",
        "",
        "현재 DB에는 사람이 확정한 정답 라벨이 없으므로, 아래의 정확도는 엄밀한 정답 기준 accuracy가 아니라 모델 간 합의율과 다수결 합의 기준의 proxy accuracy입니다.",
        "보고서에는 `정답 라벨 부재 상태의 모델 일치율`로 표기하는 것이 안전합니다.",
        "",
        "## 데이터 정리 요약",
        "",
        markdown_table(source_rows, ["label", "count"], max_rows=20),
        "",
        "## 핵심 지표",
        "",
        markdown_table(summary_metrics, ["metric", "value", "denominator", "rate", "note"], max_rows=30),
        "",
        "## 감성 분포",
        "",
        "![감성 분포](graphs/sentiment_distribution.svg)",
        "",
        markdown_table(dist_rows, ["label", "current_model", "comparison_model", "codex_audit"], max_rows=10),
        "",
        "## 태그 분포",
        "",
        "![태그 분포](graphs/tag_distribution.svg)",
        "",
        markdown_table(tag_rows, ["label", "current_model", "codex_audit"], max_rows=20),
        "",
        "## 모델 간 감성 일치율",
        "",
        "![감성 일치율 히트맵](graphs/sentiment_agreement_heatmap.svg)",
        "",
        "## 합의 기준 Proxy Accuracy",
        "",
        "![Proxy accuracy](graphs/proxy_accuracy.svg)",
        "",
        markdown_table(proxy_rows, ["model", "compared_items", "matches", "rate", "note"], max_rows=20),
        "",
        "## Codex 임베딩 감사",
        "",
        markdown_table([embedding_metrics], [
            "embedding_model",
            "item_count",
            "embedding_dim",
            "vocabulary_size",
            "avg_nearest_neighbor_similarity",
            "pairs_ge_0_90",
            "pairs_ge_0_80",
            "pairs_ge_0_70",
        ], max_rows=1),
        "",
        "- Codex 감사 임베딩 파일: `codex_audit/embeddings/codex_lsa_embeddings.npy`",
        "- 최근접 이웃 표: `codex_audit/embeddings/codex_nearest_neighbors.csv`",
        "- 상위 유사도 쌍: `codex_audit/embeddings/codex_top_similarity_pairs.csv`",
        "",
        "## 주요 차이 사례",
        "",
        markdown_table(
            divergence_rows,
            [
                "item_id",
                "source_group",
                "current_tag",
                "codex_primary_tag",
                "current_sentiment_3",
                "comparison_sentiment_3",
                "codex_sentiment_3",
                "divergence_flags",
                "title",
            ],
            max_rows=18,
        ),
        "",
        "## 산출물",
        "",
        "- 정리 통합표: `tables/cleaned_model_comparison.csv`",
        "- 차이 사례 전체표: `tables/divergence_cases.csv`",
        "- 지표표: `tables/summary_metrics.csv`",
        "- 비교모델 원본 export: `" + comparison_dir.name + "`",
    ]
    report_path.write_text("\n".join(lines), encoding="utf-8")

    html_lines = [
        "<!doctype html><html lang='ko'><head><meta charset='utf-8'>",
        "<title>현재 크롤링 데이터 비교분석 보고서</title>",
        "<style>body{font-family:Inter,Arial,sans-serif;margin:32px;color:#111827;line-height:1.55}table{border-collapse:collapse;width:100%;font-size:13px}th,td{border:1px solid #d0d5dd;padding:7px;vertical-align:top}th{background:#f3f4f6}img{max-width:100%;height:auto;border:1px solid #e5e7eb;margin:8px 0 18px}.note{background:#fff7ed;border:1px solid #fed7aa;padding:12px;border-radius:8px}.small{color:#667085;font-size:12px}</style>",
        "</head><body>",
        "<h1>현재 크롤링 데이터 비교분석 보고서</h1>",
        f"<p class='small'>생성 시각: {html.escape(datetime.now().isoformat(timespec='seconds'))} / 분석 대상 {len(rows)}건</p>",
        "<p class='note'>정답 라벨이 없으므로 accuracy는 모델 간 합의율 기반 proxy accuracy입니다.</p>",
        "<h2>핵심 지표</h2>",
        html_table(summary_metrics, ["metric", "value", "denominator", "rate", "note"]),
        "<h2>감성 분포</h2><img src='graphs/sentiment_distribution.svg'>",
        html_table(dist_rows, ["label", "current_model", "comparison_model", "codex_audit"]),
        "<h2>태그 분포</h2><img src='graphs/tag_distribution.svg'>",
        html_table(tag_rows, ["label", "current_model", "codex_audit"]),
        "<h2>모델 간 감성 일치율</h2><img src='graphs/sentiment_agreement_heatmap.svg'>",
        "<h2>Proxy Accuracy</h2><img src='graphs/proxy_accuracy.svg'>",
        html_table(proxy_rows, ["model", "compared_items", "matches", "rate", "note"]),
        "<h2>주요 차이 사례</h2>",
        html_table(divergence_rows[:25], ["item_id", "source_group", "current_tag", "codex_primary_tag", "current_sentiment_3", "comparison_sentiment_3", "codex_sentiment_3", "divergence_flags", "title"]),
        "</body></html>",
    ]
    (out_dir / "report.html").write_text("\n".join(html_lines), encoding="utf-8")


def html_table(rows: list[dict[str, Any]], columns: list[str]) -> str:
    parts = ["<table><thead><tr>"]
    for col in columns:
        parts.append(f"<th>{html.escape(col)}</th>")
    parts.append("</tr></thead><tbody>")
    for row in rows:
        parts.append("<tr>")
        for col in columns:
            parts.append(f"<td>{html.escape(str(row.get(col, '')))}</td>")
        parts.append("</tr>")
    parts.append("</tbody></table>")
    return "".join(parts)


def main() -> int:
    comparison_dir = Path(sys.argv[1]) if len(sys.argv) > 1 else latest_comparison_export()
    out_dir = EXPORTS_DIR / f"report_current_comparison_{now_stamp()}"
    out_dir.mkdir(parents=True, exist_ok=False)
    (out_dir / "tables").mkdir(parents=True, exist_ok=True)
    (out_dir / "graphs").mkdir(parents=True, exist_ok=True)

    comparison_predictions = read_comparison_predictions(comparison_dir)
    snapshot_ids = set(comparison_predictions)
    items = [item for item in load_items() if item["item_id"] in snapshot_ids]
    _neighbors, embedding_metrics = build_embeddings(items, out_dir)

    cleaned_rows: list[dict[str, Any]] = []
    for item in items:
        audit = codex_audit_item(item)
        preds = comparison_predictions.get(item["item_id"], {})
        comparison_label, comparison_source, keyword_label = pick_comparison_consensus(item, preds)
        language_finbert_label, language_finbert_source = pick_language_finbert(item, preds)
        current_label3 = sentiment3(item.get("sentiment_label"), item.get("sentiment_score"))
        current_tag = str(item.get("primary_tag") or "Unclear")
        flags = []
        if current_tag != audit["codex_primary_tag"]:
            flags.append("tag_mismatch")
        if current_label3 != comparison_label:
            flags.append("current_vs_comparison_sentiment")
        if audit["codex_sentiment_3"] != comparison_label:
            flags.append("codex_vs_comparison_sentiment")
        if int(item.get("excluded") or 0) != int(audit["codex_excluded"]):
            flags.append("exclude_mismatch")

        cleaned_rows.append({
            "item_id": item["item_id"],
            "source_group": item.get("source_group") or "",
            "source": item.get("source") or "",
            "language": item.get("language") or "",
            "title": item.get("clean_title") or "",
            "url": item.get("url") or "",
            "published_at": item.get("published_at") or "",
            "content_len": item.get("content_len") or 0,
            "crawl_status": item.get("crawl_status") or "",
            "queue_status": item.get("queue_status") or "",
            "current_model_version": item.get("model_version") or "",
            "current_tag": current_tag,
            "current_relevance_score": item.get("relevance_score") or "",
            "current_sentiment_label": item.get("sentiment_label") or "",
            "current_sentiment_3": current_label3,
            "current_sentiment_score": item.get("sentiment_score") or "",
            "current_confidence": item.get("confidence") or "",
            "current_excluded": int(item.get("excluded") or 0),
            "current_exclude_reason": item.get("exclude_reason") or "",
            **audit,
            "comparison_sentiment_3": comparison_label,
            "comparison_source": comparison_source,
            "comparison_keyword_label": keyword_label,
            "language_finbert_sentiment_3": language_finbert_label,
            "language_finbert_source": language_finbert_source,
            "tag_match_current_codex": int(current_tag == audit["codex_primary_tag"]),
            "sentiment_match_current_comparison": int(current_label3 == comparison_label),
            "sentiment_match_codex_comparison": int(audit["codex_sentiment_3"] == comparison_label),
            "sentiment_match_current_codex": int(current_label3 == audit["codex_sentiment_3"]),
            "exclude_match_current_codex": int(int(item.get("excluded") or 0) == int(audit["codex_excluded"])),
            "divergence_flags": ",".join(flags),
        })

    write_csv(out_dir / "tables" / "cleaned_model_comparison.csv", cleaned_rows)

    divergence_rows = [
        {
            "item_id": row["item_id"],
            "source_group": row["source_group"],
            "source": row["source"],
            "current_tag": row["current_tag"],
            "codex_primary_tag": row["codex_primary_tag"],
            "current_sentiment_3": row["current_sentiment_3"],
            "comparison_sentiment_3": row["comparison_sentiment_3"],
            "codex_sentiment_3": row["codex_sentiment_3"],
            "current_excluded": row["current_excluded"],
            "codex_excluded": row["codex_excluded"],
            "divergence_flags": row["divergence_flags"],
            "title": compact(row["title"], 120),
            "url": row["url"],
        }
        for row in cleaned_rows
        if row.get("divergence_flags")
    ]
    divergence_rows.sort(key=lambda row: (row["source_group"], row["divergence_flags"], row["title"]))
    write_csv(out_dir / "tables" / "divergence_cases.csv", divergence_rows)

    model_cols = {
        "current_model": "current_sentiment_3",
        "comparison_model": "comparison_sentiment_3",
        "codex_audit": "codex_sentiment_3",
        "financial_keyword": "comparison_keyword_label",
        "language_finbert": "language_finbert_sentiment_3",
    }
    agreement_rows = pairwise_agreement(cleaned_rows, model_cols)
    proxy_rows = consensus_proxy_accuracy(cleaned_rows, model_cols)
    write_csv(out_dir / "tables" / "sentiment_pairwise_agreement.csv", agreement_rows)
    write_csv(out_dir / "tables" / "consensus_proxy_accuracy.csv", proxy_rows)

    total = len(cleaned_rows)
    currently_tagged = [row for row in cleaned_rows if not int(row["current_excluded"])]
    summary_metrics = [
        {
            "metric": "row_count",
            "value": total,
            "denominator": total,
            "rate": "1.000000",
            "note": "Non-test crawled rows joined with current tag_results.",
        },
        {
            "metric": "current_tagged_count",
            "value": sum(1 for row in cleaned_rows if not int(row["current_excluded"])),
            "denominator": total,
            "rate": round(sum(1 for row in cleaned_rows if not int(row["current_excluded"])) / total, 6) if total else "",
            "note": "Production tag_results not excluded.",
        },
        {
            "metric": "current_excluded_count",
            "value": sum(1 for row in cleaned_rows if int(row["current_excluded"])),
            "denominator": total,
            "rate": round(sum(1 for row in cleaned_rows if int(row["current_excluded"])) / total, 6) if total else "",
            "note": "Production tag_results excluded.",
        },
        {
            "metric": "tag_agreement_current_vs_codex_all",
            "value": sum(row["tag_match_current_codex"] for row in cleaned_rows),
            "denominator": total,
            "rate": round(sum(row["tag_match_current_codex"] for row in cleaned_rows) / total, 6) if total else "",
            "note": "Primary category exact match. This is agreement, not gold accuracy.",
        },
        {
            "metric": "tag_agreement_current_vs_codex_current_tagged",
            "value": sum(row["tag_match_current_codex"] for row in currently_tagged),
            "denominator": len(currently_tagged),
            "rate": round(sum(row["tag_match_current_codex"] for row in currently_tagged) / len(currently_tagged), 6) if currently_tagged else "",
            "note": "Computed only on production-tagged non-excluded rows.",
        },
        {
            "metric": "sentiment_agreement_current_vs_comparison",
            "value": sum(row["sentiment_match_current_comparison"] for row in cleaned_rows),
            "denominator": total,
            "rate": round(sum(row["sentiment_match_current_comparison"] for row in cleaned_rows) / total, 6) if total else "",
            "note": "3-class sentiment agreement.",
        },
        {
            "metric": "sentiment_agreement_codex_vs_comparison",
            "value": sum(row["sentiment_match_codex_comparison"] for row in cleaned_rows),
            "denominator": total,
            "rate": round(sum(row["sentiment_match_codex_comparison"] for row in cleaned_rows) / total, 6) if total else "",
            "note": "3-class sentiment agreement.",
        },
        {
            "metric": "sentiment_agreement_current_vs_codex",
            "value": sum(row["sentiment_match_current_codex"] for row in cleaned_rows),
            "denominator": total,
            "rate": round(sum(row["sentiment_match_current_codex"] for row in cleaned_rows) / total, 6) if total else "",
            "note": "3-class sentiment agreement.",
        },
        {
            "metric": "exclude_agreement_current_vs_codex",
            "value": sum(row["exclude_match_current_codex"] for row in cleaned_rows),
            "denominator": total,
            "rate": round(sum(row["exclude_match_current_codex"] for row in cleaned_rows) / total, 6) if total else "",
            "note": "Market-relevance exclusion agreement.",
        },
        {
            "metric": "divergence_case_count",
            "value": len(divergence_rows),
            "denominator": total,
            "rate": round(len(divergence_rows) / total, 6) if total else "",
            "note": "Rows with at least one tag/sentiment/exclusion mismatch.",
        },
    ]
    write_csv(out_dir / "tables" / "summary_metrics.csv", summary_metrics)

    sentiment_dist_rows = []
    for label in ["Positive", "Neutral", "Negative"]:
        sentiment_dist_rows.append({
            "label": label,
            "current_model": sum(1 for row in cleaned_rows if row["current_sentiment_3"] == label),
            "comparison_model": sum(1 for row in cleaned_rows if row["comparison_sentiment_3"] == label),
            "codex_audit": sum(1 for row in cleaned_rows if row["codex_sentiment_3"] == label),
        })
    svg_grouped_bars(out_dir / "graphs" / "sentiment_distribution.svg", "Sentiment distribution", sentiment_dist_rows, ["current_model", "comparison_model", "codex_audit"])

    tag_rows = []
    for tag in CATEGORIES + ["Unclear"]:
        tag_rows.append({
            "label": tag,
            "current_model": sum(1 for row in cleaned_rows if row["current_tag"] == tag),
            "codex_audit": sum(1 for row in cleaned_rows if row["codex_primary_tag"] == tag),
        })
    tag_rows = [row for row in tag_rows if row["current_model"] or row["codex_audit"]]
    svg_grouped_bars(out_dir / "graphs" / "tag_distribution.svg", "Primary tag distribution", tag_rows, ["current_model", "codex_audit"])
    svg_heatmap(out_dir / "graphs" / "sentiment_agreement_heatmap.svg", "Sentiment agreement heatmap", agreement_rows)
    svg_rate_bars(out_dir / "graphs" / "proxy_accuracy.svg", "Consensus proxy accuracy", proxy_rows)

    create_report(out_dir, comparison_dir, cleaned_rows, summary_metrics, proxy_rows, embedding_metrics, divergence_rows)
    write_json(out_dir / "metadata.json", {
        "generated_at": datetime.now().isoformat(timespec="seconds"),
        "database": str(DB_PATH),
        "comparison_export": str(comparison_dir),
        "row_count": total,
        "embedding_metrics": embedding_metrics,
    })
    print(str(out_dir))
    print(json.dumps({
        "row_count": total,
        "divergence_cases": len(divergence_rows),
        "tag_agreement_all": next(row["rate"] for row in summary_metrics if row["metric"] == "tag_agreement_current_vs_codex_all"),
        "sentiment_agreement_codex_vs_comparison": next(row["rate"] for row in summary_metrics if row["metric"] == "sentiment_agreement_codex_vs_comparison"),
        "exclude_agreement_current_vs_codex": next(row["rate"] for row in summary_metrics if row["metric"] == "exclude_agreement_current_vs_codex"),
    }, ensure_ascii=False, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
