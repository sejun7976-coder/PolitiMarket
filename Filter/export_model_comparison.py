"""Export raw crawl data plus multi-model embedding and sentiment comparisons."""

from __future__ import annotations

import csv
import json
import math
import os
import platform
import re
import sqlite3
import sys
import time
from collections import Counter, defaultdict
from datetime import datetime, timedelta, timezone
from pathlib import Path
from typing import Any
from email.utils import parsedate_to_datetime
from zoneinfo import ZoneInfo

import numpy as np


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

EMBEDDING_MODELS = [
    {
        "id": "BAAI/bge-m3",
        "short": "bge_m3",
        "type": "sentence_transformer",
        "family": "embedding_similarity",
        "scope": "multilingual",
        "reason": "Widely used multilingual retrieval embedding model; already used in this project for dedup.",
        "query_prefix": "",
        "local_files_only": True,
    },
    {
        "id": "intfloat/multilingual-e5-small",
        "short": "multilingual_e5_small",
        "type": "sentence_transformer",
        "family": "embedding_similarity",
        "scope": "multilingual",
        "reason": "Popular E5 multilingual embedding baseline with strong retrieval behavior.",
        "query_prefix": "passage: ",
        "local_files_only": True,
    },
    {
        "id": "sentence-transformers/paraphrase-multilingual-MiniLM-L12-v2",
        "short": "paraphrase_multilingual_minilm",
        "type": "sentence_transformer",
        "family": "embedding_similarity",
        "scope": "multilingual",
        "reason": "Classic compact SentenceTransformers multilingual semantic similarity baseline.",
        "query_prefix": "",
        "local_files_only": True,
    },
    {
        "id": "sentence-transformers/all-MiniLM-L6-v2",
        "short": "all_minilm_l6_v2",
        "type": "sentence_transformer",
        "family": "embedding_similarity",
        "scope": "english",
        "reason": "Very common English semantic similarity baseline; useful as an English-only control.",
        "query_prefix": "",
        "local_files_only": True,
    },
    {
        "id": "sklearn/tfidf-word-1-2",
        "short": "tfidf_word_1_2",
        "type": "tfidf_word",
        "family": "embedding_similarity",
        "scope": "multilingual",
        "reason": "Classic TF-IDF cosine baseline for lexical similarity and duplicate checking.",
        "query_prefix": "",
    },
    {
        "id": "sklearn/tfidf-char-wb-3-5",
        "short": "tfidf_char_wb_3_5",
        "type": "tfidf_char",
        "family": "embedding_similarity",
        "scope": "multilingual",
        "reason": "Character n-gram TF-IDF baseline, robust to mixed Korean/English tokens and boilerplate.",
        "query_prefix": "",
    },
    {
        "id": "sklearn/lsa-tfidf-svd-128",
        "short": "lsa_tfidf_svd_128",
        "type": "lsa_tfidf",
        "family": "embedding_similarity",
        "scope": "multilingual",
        "reason": "Latent Semantic Analysis baseline built from TF-IDF plus TruncatedSVD.",
        "query_prefix": "",
    },
]

SENTIMENT_MODELS = [
    {
        "id": "ProsusAI/finbert",
        "short": "prosus_finbert",
        "type": "transformer",
        "family": "financial_sentiment",
        "scope": "en",
        "reason": "Famous English financial-news sentiment classifier.",
        "local_files_only": True,
    },
    {
        "id": "snunlp/KR-FinBert-SC",
        "short": "kr_finbert_sc",
        "type": "transformer",
        "family": "financial_sentiment",
        "scope": "ko",
        "reason": "Korean finance-domain BERT sentiment classifier.",
        "local_files_only": True,
    },
    {
        "id": "distilbert-base-uncased-finetuned-sst-2-english",
        "short": "distilbert_sst2",
        "type": "transformer",
        "family": "general_sentiment",
        "scope": "en",
        "reason": "Canonical lightweight English general sentiment baseline.",
        "local_files_only": True,
    },
    {
        "id": "nlptown/bert-base-multilingual-uncased-sentiment",
        "short": "nlptown_multilingual_stars",
        "type": "transformer",
        "family": "general_sentiment",
        "scope": "multilingual",
        "reason": "Popular multilingual 1-5 star sentiment baseline.",
        "local_files_only": True,
    },
    {
        "id": "rule/financial-risk-opportunity-lexicon-v1",
        "short": "financial_keyword_baseline",
        "type": "keyword",
        "family": "financial_sentiment",
        "scope": "multilingual",
        "reason": "Transparent Korean/English financial risk-opportunity lexicon baseline for audit control.",
    },
]


def safe_name(value: str) -> str:
    return re.sub(r"[^A-Za-z0-9_.-]+", "_", value).strip("_").lower()


def now_stamp() -> str:
    return datetime.now().strftime("%Y%m%d_%H%M%S")


def json_dump(path: Path, data: Any) -> None:
    path.write_text(json.dumps(data, ensure_ascii=False, indent=2), encoding="utf-8")


def compact_text(item: dict[str, Any], limit: int = 1600) -> str:
    title = " ".join(str(item.get("title") or "").split())
    content = " ".join(str(item.get("content") or "").split())
    if title and title not in content[:200]:
        text = f"{title}. {content}"
    else:
        text = content or title
    return text[:limit].strip()


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


def load_items() -> list[dict[str, Any]]:
    conn = sqlite3.connect(DB_PATH)
    conn.row_factory = sqlite3.Row
    try:
        rows = conn.execute(
            """
            SELECT id, content_hash, source, title, content, url, raw_url, published_at,
                   language, country, source_group, created_at, updated_at
            FROM crawled_items
            ORDER BY COALESCE(published_at, created_at) DESC, created_at DESC
            """
        ).fetchall()
        items = [dict(row) for row in rows]
        items = [item for item in items if is_within_lookback_item(item)]
        for item in items:
            item["model_text"] = compact_text(item)
        return items
    finally:
        conn.close()


def write_raw_data(out_dir: Path, items: list[dict[str, Any]]) -> None:
    raw_dir = out_dir / "raw_data"
    raw_dir.mkdir(parents=True, exist_ok=True)
    raw_items = [{k: v for k, v in item.items() if k != "model_text"} for item in items]
    json_dump(raw_dir / "raw_crawled_items.json", raw_items)
    with (raw_dir / "raw_crawled_items.jsonl").open("w", encoding="utf-8", newline="") as f:
        for item in raw_items:
            f.write(json.dumps(item, ensure_ascii=False) + "\n")
    fieldnames = list(raw_items[0].keys()) if raw_items else [
        "id", "content_hash", "source", "title", "content", "url", "raw_url",
        "published_at", "language", "country", "source_group", "created_at", "updated_at",
    ]
    with (raw_dir / "raw_crawled_items.csv").open("w", encoding="utf-8-sig", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(raw_items)
    with (raw_dir / "model_input_pretag.jsonl").open("w", encoding="utf-8", newline="") as f:
        for item in items:
            f.write(json.dumps({
                "id": item.get("id"),
                "text": item.get("model_text"),
                "title": item.get("title"),
                "source": item.get("source"),
                "source_group": item.get("source_group"),
                "language": item.get("language"),
                "published_at": item.get("published_at"),
                "url": item.get("url"),
            }, ensure_ascii=False) + "\n")


def write_csv(path: Path, rows: list[dict[str, Any]], fieldnames: list[str] | None = None) -> None:
    if fieldnames is None:
        fieldnames = list(rows[0].keys()) if rows else []
    with path.open("w", encoding="utf-8-sig", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)


def top_similarity_pairs(sim: np.ndarray, items: list[dict[str, Any]], limit: int = 80) -> list[dict[str, Any]]:
    pairs: list[dict[str, Any]] = []
    n = len(items)
    for i in range(n):
        for j in range(i + 1, n):
            pairs.append({
                "item_id_a": items[i]["id"],
                "item_id_b": items[j]["id"],
                "similarity": round(float(sim[i, j]), 6),
                "source_a": items[i].get("source"),
                "source_b": items[j].get("source"),
                "source_group_a": items[i].get("source_group"),
                "source_group_b": items[j].get("source_group"),
                "title_a": items[i].get("title"),
                "title_b": items[j].get("title"),
                "url_a": items[i].get("url"),
                "url_b": items[j].get("url"),
            })
    pairs.sort(key=lambda row: row["similarity"], reverse=True)
    return pairs[:limit]


def nearest_neighbors(sim: np.ndarray, items: list[dict[str, Any]]) -> list[dict[str, Any]]:
    rows = []
    n = len(items)
    for i in range(n):
        if n <= 1:
            continue
        scores = sim[i].copy()
        scores[i] = -2
        j = int(np.argmax(scores))
        rows.append({
            "item_id": items[i]["id"],
            "nearest_item_id": items[j]["id"],
            "similarity": round(float(scores[j]), 6),
            "source": items[i].get("source"),
            "nearest_source": items[j].get("source"),
            "source_group": items[i].get("source_group"),
            "nearest_source_group": items[j].get("source_group"),
            "title": items[i].get("title"),
            "nearest_title": items[j].get("title"),
        })
    return rows


def run_embedding_models(out_dir: Path, items: list[dict[str, Any]]) -> tuple[list[dict[str, Any]], dict[tuple[str, str], list[float]]]:
    from sentence_transformers import SentenceTransformer
    from sklearn.decomposition import TruncatedSVD
    from sklearn.feature_extraction.text import TfidfVectorizer
    from sklearn.preprocessing import normalize

    emb_dir = out_dir / "embedding_similarity"
    emb_dir.mkdir(parents=True, exist_ok=True)
    model_status: list[dict[str, Any]] = []
    metrics: list[dict[str, Any]] = []
    consensus_scores: dict[tuple[str, str], list[float]] = defaultdict(list)

    for cfg in EMBEDDING_MODELS:
        started = time.perf_counter()
        status = {**cfg, "status": "pending", "error": "", "seconds": None}
        model_dir = emb_dir / safe_name(cfg["short"])
        model_dir.mkdir(parents=True, exist_ok=True)
        try:
            texts = [f"{cfg.get('query_prefix', '')}{item['model_text']}" for item in items]
            if cfg.get("type") == "sentence_transformer":
                model = SentenceTransformer(cfg["id"], local_files_only=bool(cfg.get("local_files_only")))
                embeddings = model.encode(
                    texts,
                    batch_size=16,
                    normalize_embeddings=True,
                    show_progress_bar=False,
                    convert_to_numpy=True,
                )
                embeddings = np.asarray(embeddings, dtype=np.float32)
            elif cfg.get("type") == "tfidf_word":
                vectorizer = TfidfVectorizer(
                    analyzer="word",
                    ngram_range=(1, 2),
                    min_df=1,
                    max_df=0.92,
                    lowercase=True,
                )
                embeddings = vectorizer.fit_transform(texts).astype(np.float32).toarray()
                embeddings = normalize(embeddings, norm="l2").astype(np.float32)
                json_dump(model_dir / "vectorizer_manifest.json", {
                    "vocabulary_size": len(vectorizer.vocabulary_),
                    "analyzer": "word",
                    "ngram_range": [1, 2],
                })
            elif cfg.get("type") == "tfidf_char":
                vectorizer = TfidfVectorizer(
                    analyzer="char_wb",
                    ngram_range=(3, 5),
                    min_df=1,
                    max_df=0.95,
                    lowercase=True,
                )
                embeddings = vectorizer.fit_transform(texts).astype(np.float32).toarray()
                embeddings = normalize(embeddings, norm="l2").astype(np.float32)
                json_dump(model_dir / "vectorizer_manifest.json", {
                    "vocabulary_size": len(vectorizer.vocabulary_),
                    "analyzer": "char_wb",
                    "ngram_range": [3, 5],
                })
            elif cfg.get("type") == "lsa_tfidf":
                vectorizer = TfidfVectorizer(
                    analyzer="word",
                    ngram_range=(1, 2),
                    min_df=1,
                    max_df=0.92,
                    lowercase=True,
                )
                sparse = vectorizer.fit_transform(texts).astype(np.float32)
                n_components = max(2, min(128, sparse.shape[0] - 1, sparse.shape[1] - 1))
                svd = TruncatedSVD(n_components=n_components, random_state=42)
                embeddings = svd.fit_transform(sparse).astype(np.float32)
                embeddings = normalize(embeddings, norm="l2").astype(np.float32)
                json_dump(model_dir / "vectorizer_manifest.json", {
                    "vocabulary_size": len(vectorizer.vocabulary_),
                    "n_components": n_components,
                    "explained_variance_ratio_sum": round(float(np.sum(svd.explained_variance_ratio_)), 6),
                })
            else:
                raise ValueError(f"Unknown embedding model type: {cfg.get('type')}")
            sim = embeddings @ embeddings.T
            np.save(model_dir / "embeddings.npy", embeddings)
            top_pairs = top_similarity_pairs(sim, items)
            neighbors = nearest_neighbors(sim, items)
            write_csv(model_dir / "similarity_top_pairs.csv", top_pairs)
            write_csv(model_dir / "item_nearest_neighbors.csv", neighbors)
            json_dump(model_dir / "model_manifest.json", cfg)

            for row in top_pairs[:40]:
                key = tuple(sorted([row["item_id_a"], row["item_id_b"]]))
                consensus_scores[key].append(float(row["similarity"]))

            all_pair_scores = sim[np.triu_indices(len(items), k=1)] if len(items) > 1 else np.array([])
            nn_scores = [float(row["similarity"]) for row in neighbors]
            metric = {
                "model_short": cfg["short"],
                "model_id": cfg["id"],
                "status": "success",
                "embedding_dim": int(embeddings.shape[1]) if embeddings.ndim == 2 else 0,
                "item_count": len(items),
                "avg_pair_similarity": round(float(np.mean(all_pair_scores)), 6) if all_pair_scores.size else 0,
                "avg_nearest_neighbor_similarity": round(float(np.mean(nn_scores)), 6) if nn_scores else 0,
                "pairs_ge_0_90": int(np.sum(all_pair_scores >= 0.90)) if all_pair_scores.size else 0,
                "pairs_ge_0_85": int(np.sum(all_pair_scores >= 0.85)) if all_pair_scores.size else 0,
                "pairs_ge_0_80": int(np.sum(all_pair_scores >= 0.80)) if all_pair_scores.size else 0,
                "seconds": round(time.perf_counter() - started, 2),
            }
            metrics.append(metric)
            status.update({"status": "success", "seconds": metric["seconds"], "embedding_dim": metric["embedding_dim"]})
        except Exception as exc:
            status.update({"status": "failed", "error": str(exc), "seconds": round(time.perf_counter() - started, 2)})
            metrics.append({
                "model_short": cfg["short"],
                "model_id": cfg["id"],
                "status": "failed",
                "embedding_dim": 0,
                "item_count": len(items),
                "avg_pair_similarity": "",
                "avg_nearest_neighbor_similarity": "",
                "pairs_ge_0_90": "",
                "pairs_ge_0_85": "",
                "pairs_ge_0_80": "",
                "seconds": status["seconds"],
            })
        model_status.append(status)

    write_csv(emb_dir / "summary_embedding_metrics.csv", metrics)
    consensus_rows = []
    item_by_id = {item["id"]: item for item in items}
    for (a, b), scores in consensus_scores.items():
        if len(scores) < 2:
            continue
        ia = item_by_id.get(a, {})
        ib = item_by_id.get(b, {})
        consensus_rows.append({
            "item_id_a": a,
            "item_id_b": b,
            "model_count": len(scores),
            "avg_similarity": round(sum(scores) / len(scores), 6),
            "min_similarity": round(min(scores), 6),
            "max_similarity": round(max(scores), 6),
            "source_a": ia.get("source"),
            "source_b": ib.get("source"),
            "source_group_a": ia.get("source_group"),
            "source_group_b": ib.get("source_group"),
            "title_a": ia.get("title"),
            "title_b": ib.get("title"),
        })
    consensus_rows.sort(key=lambda row: (row["model_count"], row["avg_similarity"]), reverse=True)
    write_csv(emb_dir / "consensus_top_pairs.csv", consensus_rows[:100])
    json_dump(emb_dir / "models_manifest.json", model_status)
    return metrics, consensus_scores


def softmax(logits: Any) -> list[float]:
    arr = np.asarray(logits, dtype=np.float64)
    arr = arr - np.max(arr)
    exp = np.exp(arr)
    return (exp / exp.sum()).tolist()


def normalize_sentiment(labels: list[str], probs: list[float]) -> tuple[str, float, float, str]:
    pairs = [(str(label), float(prob)) for label, prob in zip(labels, probs)]
    lower_pairs = [(label, label.lower(), prob) for label, prob in pairs]
    raw_label, _lower, confidence = max(lower_pairs, key=lambda row: row[2])

    star_scores = []
    for label, lower, prob in lower_pairs:
        match = re.search(r"([1-5])", lower)
        if "star" in lower and match:
            star_scores.append((int(match.group(1)), prob))
    if star_scores:
        expected = sum(star * prob for star, prob in star_scores)
        score = max(-1.0, min(1.0, (expected - 3.0) / 2.0))
        if score >= 0.18:
            return "Positive", round(score, 6), round(confidence, 6), raw_label
        if score <= -0.18:
            return "Negative", round(score, 6), round(confidence, 6), raw_label
        return "Neutral", round(score, 6), round(confidence, 6), raw_label

    prob_by_name = {lower: prob for _label, lower, prob in lower_pairs}
    pos = sum(prob for lower, prob in prob_by_name.items() if "pos" in lower or "positive" in lower)
    neg = sum(prob for lower, prob in prob_by_name.items() if "neg" in lower or "negative" in lower)
    neu = sum(prob for lower, prob in prob_by_name.items() if "neu" in lower or "neutral" in lower)
    if pos or neg or neu:
        score = max(-1.0, min(1.0, pos - neg))
        if neu >= max(pos, neg) and abs(score) < 0.18:
            return "Neutral", round(score, 6), round(confidence, 6), raw_label
        if score >= 0.18:
            return "Positive", round(score, 6), round(confidence, 6), raw_label
        if score <= -0.18:
            return "Negative", round(score, 6), round(confidence, 6), raw_label
        return "Neutral", round(score, 6), round(confidence, 6), raw_label

    lower = raw_label.lower()
    if "label_2" in lower:
        return "Positive", round(confidence, 6), round(confidence, 6), raw_label
    if "label_0" in lower:
        return "Negative", round(-confidence, 6), round(confidence, 6), raw_label
    return "Neutral", 0.0, round(confidence, 6), raw_label


def language_allowed(scope: str, item: dict[str, Any]) -> bool:
    language = str(item.get("language") or "").lower()
    if scope == "multilingual":
        return True
    if scope == "en":
        return language.startswith("en")
    if scope == "ko":
        return language.startswith("ko") or any("\uac00" <= ch <= "\ud7a3" for ch in item.get("model_text", ""))
    return True


NEGATIVE_TERMS = [
    "risk", "war", "sanction", "crisis", "fall", "decline", "attack", "conflict",
    "inflation", "default", "volatility", "tariff", "restriction", "uncertainty",
    "위험", "위기", "전쟁", "제재", "규제", "하락", "급락", "감소", "부진", "침체",
    "손실", "피해", "공격", "갈등", "불안", "압박", "차단", "수출통제", "변동성",
    "무역분쟁", "인플레이션", "금리상승",
]
POSITIVE_TERMS = [
    "growth", "deal", "agreement", "investment", "cooperation", "support", "profit",
    "record", "expand", "recovery", "improve", "export", "strong demand",
    "성장", "상승", "증가", "확대", "회복", "개선", "투자", "합의", "협력", "지원",
    "성공", "수출", "이익", "최대", "신기록", "호조", "수주", "안정", "개발", "시장성",
]


def keyword_sentiment(item: dict[str, Any]) -> tuple[str, float, float, str, dict[str, int]]:
    text = str(item.get("model_text") or "").lower()
    negative = sum(text.count(term.lower()) for term in NEGATIVE_TERMS)
    positive = sum(text.count(term.lower()) for term in POSITIVE_TERMS)
    raw = positive - negative
    denom = max(positive + negative, 1)
    score = max(-1.0, min(1.0, raw / max(denom, 3)))
    if score >= 0.16:
        label = "Positive"
    elif score <= -0.16:
        label = "Negative"
    else:
        label = "Neutral"
    confidence = min(0.92, 0.42 + min(abs(raw), 6) * 0.08 + min(denom, 10) * 0.02)
    return label, round(score, 6), round(confidence, 6), f"pos={positive};neg={negative}", {
        "positive_hits": positive,
        "negative_hits": negative,
    }


def run_sentiment_models(out_dir: Path, items: list[dict[str, Any]]) -> tuple[list[dict[str, Any]], list[dict[str, Any]], list[dict[str, Any]]]:
    import torch
    from transformers import AutoModelForSequenceClassification, AutoTokenizer

    sent_dir = out_dir / "sentiment"
    sent_dir.mkdir(parents=True, exist_ok=True)
    device = "cuda" if torch.cuda.is_available() else "cpu"
    model_status: list[dict[str, Any]] = []
    prediction_rows: list[dict[str, Any]] = []

    for cfg in SENTIMENT_MODELS:
        started = time.perf_counter()
        status = {**cfg, "status": "pending", "error": "", "seconds": None, "device": device}
        try:
            if cfg.get("type") == "keyword":
                for item in items:
                    canonical, score, confidence, raw_label, raw_probs = keyword_sentiment(item)
                    prediction_rows.append({
                        "model_short": cfg["short"],
                        "model_id": cfg["id"],
                        "item_id": item["id"],
                        "language": item.get("language"),
                        "source": item.get("source"),
                        "source_group": item.get("source_group"),
                        "title": item.get("title"),
                        "canonical_label": canonical,
                        "sentiment_score": score,
                        "confidence": confidence,
                        "raw_label": raw_label,
                        "raw_probabilities_json": json.dumps(raw_probs, ensure_ascii=False),
                    })
                status.update({
                    "status": "success",
                    "seconds": round(time.perf_counter() - started, 2),
                    "covered_items": len(items),
                    "total_items": len(items),
                })
                model_status.append(status)
                continue

            tokenizer = AutoTokenizer.from_pretrained(cfg["id"], local_files_only=bool(cfg.get("local_files_only")))
            model = AutoModelForSequenceClassification.from_pretrained(
                cfg["id"],
                local_files_only=bool(cfg.get("local_files_only")),
            )
            model.to(device)
            model.eval()
            id2label = {int(k): str(v) for k, v in model.config.id2label.items()}
            scoped_items = [item for item in items if language_allowed(cfg["scope"], item)]
            for item in items:
                if item not in scoped_items:
                    prediction_rows.append({
                        "model_short": cfg["short"],
                        "model_id": cfg["id"],
                        "item_id": item["id"],
                        "language": item.get("language"),
                        "source": item.get("source"),
                        "source_group": item.get("source_group"),
                        "title": item.get("title"),
                        "canonical_label": "Skipped",
                        "sentiment_score": "",
                        "confidence": "",
                        "raw_label": "language_scope_skip",
                        "raw_probabilities_json": "{}",
                    })

            batch_size = 12
            for start in range(0, len(scoped_items), batch_size):
                batch = scoped_items[start:start + batch_size]
                encoded = tokenizer(
                    [item["model_text"] or " " for item in batch],
                    padding=True,
                    truncation=True,
                    max_length=256,
                    return_tensors="pt",
                )
                encoded = {key: value.to(device) for key, value in encoded.items()}
                with torch.no_grad():
                    logits = model(**encoded).logits.detach().cpu().tolist()
                for item, row_logits in zip(batch, logits):
                    probs = softmax(row_logits)
                    labels = [id2label.get(idx, str(idx)) for idx in range(len(probs))]
                    canonical, score, confidence, raw_label = normalize_sentiment(labels, probs)
                    raw_probs = {label: round(float(prob), 6) for label, prob in zip(labels, probs)}
                    prediction_rows.append({
                        "model_short": cfg["short"],
                        "model_id": cfg["id"],
                        "item_id": item["id"],
                        "language": item.get("language"),
                        "source": item.get("source"),
                        "source_group": item.get("source_group"),
                        "title": item.get("title"),
                        "canonical_label": canonical,
                        "sentiment_score": score,
                        "confidence": confidence,
                        "raw_label": raw_label,
                        "raw_probabilities_json": json.dumps(raw_probs, ensure_ascii=False),
                    })
            status.update({
                "status": "success",
                "seconds": round(time.perf_counter() - started, 2),
                "label_map": id2label,
                "covered_items": len(scoped_items),
                "total_items": len(items),
            })
        except Exception as exc:
            status.update({"status": "failed", "error": str(exc), "seconds": round(time.perf_counter() - started, 2)})
        model_status.append(status)

    prediction_rows.sort(key=lambda row: (row["item_id"], row["model_short"]))
    write_csv(sent_dir / "sentiment_predictions.csv", prediction_rows)
    with (sent_dir / "sentiment_predictions.jsonl").open("w", encoding="utf-8", newline="") as f:
        for row in prediction_rows:
            f.write(json.dumps(row, ensure_ascii=False) + "\n")

    distribution_rows = []
    for model_short in sorted({row["model_short"] for row in prediction_rows}):
        rows = [row for row in prediction_rows if row["model_short"] == model_short]
        counts = Counter(row["canonical_label"] for row in rows)
        total_scored = sum(counts[label] for label in ("Positive", "Neutral", "Negative"))
        distribution_rows.append({
            "model_short": model_short,
            "positive": counts.get("Positive", 0),
            "neutral": counts.get("Neutral", 0),
            "negative": counts.get("Negative", 0),
            "skipped": counts.get("Skipped", 0),
            "scored_total": total_scored,
        })
    write_csv(sent_dir / "sentiment_distribution_by_model.csv", distribution_rows)

    agreement_rows = sentiment_agreement_rows(prediction_rows)
    write_csv(sent_dir / "sentiment_agreement_matrix.csv", agreement_rows)
    disagreements = item_disagreement_rows(items, prediction_rows)
    write_csv(sent_dir / "item_disagreement_cases.csv", disagreements[:80])
    json_dump(sent_dir / "models_manifest.json", model_status)
    return model_status, distribution_rows, agreement_rows


def sentiment_agreement_rows(predictions: list[dict[str, Any]]) -> list[dict[str, Any]]:
    by_model: dict[str, dict[str, str]] = defaultdict(dict)
    for row in predictions:
        if row["canonical_label"] == "Skipped":
            continue
        by_model[row["model_short"]][row["item_id"]] = row["canonical_label"]
    models = sorted(by_model)
    rows = []
    for a in models:
        for b in models:
            common = sorted(set(by_model[a]) & set(by_model[b]))
            matches = sum(1 for item_id in common if by_model[a][item_id] == by_model[b][item_id])
            rows.append({
                "model_a": a,
                "model_b": b,
                "common_items": len(common),
                "matches": matches,
                "agreement_rate": round(matches / len(common), 6) if common else "",
            })
    return rows


def item_disagreement_rows(items: list[dict[str, Any]], predictions: list[dict[str, Any]]) -> list[dict[str, Any]]:
    item_meta = {item["id"]: item for item in items}
    by_item: dict[str, list[dict[str, Any]]] = defaultdict(list)
    for row in predictions:
        if row["canonical_label"] != "Skipped":
            by_item[row["item_id"]].append(row)
    rows = []
    for item_id, preds in by_item.items():
        labels = [row["canonical_label"] for row in preds]
        if len(set(labels)) <= 1:
            continue
        counts = Counter(labels)
        item = item_meta.get(item_id, {})
        rows.append({
            "item_id": item_id,
            "source": item.get("source"),
            "source_group": item.get("source_group"),
            "language": item.get("language"),
            "title": item.get("title"),
            "model_count": len(preds),
            "labels_json": json.dumps({row["model_short"]: row["canonical_label"] for row in preds}, ensure_ascii=False),
            "scores_json": json.dumps({row["model_short"]: row["sentiment_score"] for row in preds}, ensure_ascii=False),
            "disagreement_count": len(counts),
        })
    rows.sort(key=lambda row: (row["disagreement_count"], row["model_count"]), reverse=True)
    return rows


def svg_bar_chart(path: Path, title: str, rows: list[dict[str, Any]], labels: list[str], colors: dict[str, str]) -> None:
    width = 920
    group_h = 44
    height = 80 + max(1, len(rows)) * group_h
    max_value = max([float(row.get(label, 0) or 0) for row in rows for label in labels] + [1])
    x0, y0 = 210, 54
    bar_h = 9
    parts = [
        f'<svg xmlns="http://www.w3.org/2000/svg" width="{width}" height="{height}" viewBox="0 0 {width} {height}">',
        '<rect width="100%" height="100%" fill="#ffffff"/>',
        f'<text x="24" y="28" font-family="Inter, Arial" font-size="18" font-weight="700" fill="#111827">{title}</text>',
    ]
    for idx, row in enumerate(rows):
        y = y0 + idx * group_h
        name = str(row.get("model_short") or row.get("model") or idx)
        parts.append(f'<text x="24" y="{y + 18}" font-family="Inter, Arial" font-size="12" font-weight="600" fill="#374151">{escape_xml(name)}</text>')
        for j, label in enumerate(labels):
            value = float(row.get(label, 0) or 0)
            bw = (value / max_value) * 560
            by = y + j * (bar_h + 2)
            parts.append(f'<rect x="{x0}" y="{by}" width="{bw:.2f}" height="{bar_h}" rx="3" fill="{colors[label]}"/>')
            parts.append(f'<text x="{x0 + bw + 8:.2f}" y="{by + 8}" font-family="Inter, Arial" font-size="10" fill="#475467">{label}: {int(value)}</text>')
    parts.append("</svg>")
    path.write_text("\n".join(parts), encoding="utf-8")


def svg_heatmap(path: Path, title: str, rows: list[dict[str, Any]]) -> None:
    models = sorted({row["model_a"] for row in rows} | {row["model_b"] for row in rows})
    cell = 95
    left = 190
    top = 80
    width = left + cell * len(models) + 40
    height = top + cell * len(models) + 40
    by_pair = {(row["model_a"], row["model_b"]): row for row in rows}
    parts = [
        f'<svg xmlns="http://www.w3.org/2000/svg" width="{width}" height="{height}" viewBox="0 0 {width} {height}">',
        '<rect width="100%" height="100%" fill="#ffffff"/>',
        f'<text x="24" y="30" font-family="Inter, Arial" font-size="18" font-weight="700" fill="#111827">{title}</text>',
    ]
    for i, model in enumerate(models):
        parts.append(f'<text x="{left + i * cell + 6}" y="62" font-family="Inter, Arial" font-size="10" fill="#344054" transform="rotate(-25 {left + i * cell + 6},62)">{escape_xml(model)}</text>')
        parts.append(f'<text x="24" y="{top + i * cell + 52}" font-family="Inter, Arial" font-size="11" font-weight="600" fill="#344054">{escape_xml(model)}</text>')
    for y_idx, a in enumerate(models):
        for x_idx, b in enumerate(models):
            row = by_pair.get((a, b), {})
            value = row.get("agreement_rate")
            rate = float(value) if value != "" and value is not None else 0.0
            blue = int(245 - rate * 120)
            fill = f"rgb({blue},{blue + 8},255)"
            x = left + x_idx * cell
            y = top + y_idx * cell
            text = f"{rate:.2f}" if row else "n/a"
            parts.append(f'<rect x="{x}" y="{y}" width="{cell - 6}" height="{cell - 6}" rx="8" fill="{fill}" stroke="#d0d5dd"/>')
            parts.append(f'<text x="{x + cell / 2 - 18}" y="{y + cell / 2 + 4}" font-family="Inter, Arial" font-size="14" font-weight="700" fill="#111827">{text}</text>')
    parts.append("</svg>")
    path.write_text("\n".join(parts), encoding="utf-8")


def escape_xml(value: Any) -> str:
    return str(value or "").replace("&", "&amp;").replace("<", "&lt;").replace(">", "&gt;")


def create_graphs(out_dir: Path) -> None:
    graphs_dir = out_dir / "graphs"
    graphs_dir.mkdir(parents=True, exist_ok=True)
    sent_dist_path = out_dir / "sentiment" / "sentiment_distribution_by_model.csv"
    emb_metric_path = out_dir / "embedding_similarity" / "summary_embedding_metrics.csv"
    agreement_path = out_dir / "sentiment" / "sentiment_agreement_matrix.csv"

    if sent_dist_path.exists():
        rows = list(csv.DictReader(sent_dist_path.open("r", encoding="utf-8-sig")))
        svg_bar_chart(
            graphs_dir / "sentiment_distribution_by_model.svg",
            "Sentiment Distribution by Model",
            rows,
            ["positive", "neutral", "negative", "skipped"],
            {"positive": "#16a34a", "neutral": "#94a3b8", "negative": "#dc2626", "skipped": "#cbd5e1"},
        )
    if emb_metric_path.exists():
        rows = list(csv.DictReader(emb_metric_path.open("r", encoding="utf-8-sig")))
        svg_bar_chart(
            graphs_dir / "embedding_similarity_threshold_counts.svg",
            "High-Similarity Pair Counts by Embedding Model",
            rows,
            ["pairs_ge_0_90", "pairs_ge_0_85", "pairs_ge_0_80"],
            {"pairs_ge_0_90": "#1d4ed8", "pairs_ge_0_85": "#2563eb", "pairs_ge_0_80": "#60a5fa"},
        )
    if agreement_path.exists():
        rows = list(csv.DictReader(agreement_path.open("r", encoding="utf-8-sig")))
        svg_heatmap(graphs_dir / "sentiment_model_agreement_heatmap.svg", "Sentiment Model Agreement", rows)


def create_validation_docs(out_dir: Path, items: list[dict[str, Any]]) -> None:
    validation_dir = out_dir / "validation"
    validation_dir.mkdir(parents=True, exist_ok=True)
    schema = """# External Verification Result Schema

Return one of these files after Claude or Antigravity review.

## Preferred CSV: `external_sentiment_review.csv`

Columns:

- `reviewer`: `claude` or `antigravity`
- `item_id`
- `final_label`: one of `Positive`, `Neutral`, `Negative`, `Unclear`
- `confidence`: number from 0 to 1
- `rationale`: short Korean or English explanation
- `model_notes`: optional notes about which local model looked wrong

## Optional Similarity CSV: `external_similarity_review.csv`

Columns:

- `reviewer`: `claude` or `antigravity`
- `item_id_a`
- `item_id_b`
- `is_duplicate_or_same_event`: `yes`, `partial`, or `no`
- `confidence`: number from 0 to 1
- `rationale`

When you give these files back, Codex will merge them with the local model results and create comparison tables and graphs.
"""
    (validation_dir / "external_result_schema.md").write_text(schema, encoding="utf-8")

    base_prompt = f"""# PolitiMarket Raw Crawl Data Verification

You are reviewing raw crawled political/economic text items before dashboard use.

Data package:

- Raw input: `raw_data/model_input_pretag.jsonl`
- Full raw rows: `raw_data/raw_crawled_items.jsonl`
- Local sentiment predictions: `sentiment/sentiment_predictions.csv`
- Local similarity outputs: `embedding_similarity/*/similarity_top_pairs.csv`
- Consensus similarity pairs: `embedding_similarity/consensus_top_pairs.csv`

Current item count: {len(items)}

## Task A: Sentiment Verification

For each item, judge market-relevant sentiment toward Korean/global markets:

- `Positive`: likely supportive or risk-on for markets/sectors.
- `Neutral`: factual, balanced, or weak market direction.
- `Negative`: risk, conflict, sanctions, macro stress, geopolitical uncertainty, or market downside.
- `Unclear`: insufficient text or not market-relevant.

Use the raw title/content rather than trusting model outputs. If multiple local models disagree, explain which label you choose and why.

## Task B: Similarity / Duplicate Verification

Open `embedding_similarity/consensus_top_pairs.csv` first, then spot-check each model's `similarity_top_pairs.csv`.
For each high-similarity pair, classify whether it is:

- `yes`: duplicate or same event.
- `partial`: related theme/event but not duplicate.
- `no`: semantically different.

## Required Output

Create CSVs following `validation/external_result_schema.md`.

Keep rationales short. Do not invent facts outside the provided raw data.
"""
    (validation_dir / "claude_validation_prompt.md").write_text(
        base_prompt.replace("You are reviewing", "You are Claude reviewing"),
        encoding="utf-8",
    )
    (validation_dir / "antigravity_validation_prompt.md").write_text(
        base_prompt.replace("You are reviewing", "You are Antigravity reviewing"),
        encoding="utf-8",
    )


def create_codex_report(out_dir: Path, items: list[dict[str, Any]]) -> None:
    emb_metrics = list(csv.DictReader((out_dir / "embedding_similarity" / "summary_embedding_metrics.csv").open("r", encoding="utf-8-sig")))
    sent_dist = list(csv.DictReader((out_dir / "sentiment" / "sentiment_distribution_by_model.csv").open("r", encoding="utf-8-sig")))
    disagreements_path = out_dir / "sentiment" / "item_disagreement_cases.csv"
    disagreements = list(csv.DictReader(disagreements_path.open("r", encoding="utf-8-sig"))) if disagreements_path.exists() else []
    consensus_path = out_dir / "embedding_similarity" / "consensus_top_pairs.csv"
    consensus = list(csv.DictReader(consensus_path.open("r", encoding="utf-8-sig"))) if consensus_path.exists() else []

    lines = [
        "# Codex Comparison Report",
        "",
        f"- Export folder: `{out_dir.name}`",
        f"- Raw item count: `{len(items)}`",
        f"- Generated at: `{datetime.now().isoformat(timespec='seconds')}`",
        "",
        "## Embedding / Similarity Models",
        "",
        "| Model | Status | Dim | Avg NN Similarity | Pairs >= .90 | Pairs >= .85 | Pairs >= .80 |",
        "|---|---:|---:|---:|---:|---:|---:|",
    ]
    for row in emb_metrics:
        lines.append(
            f"| {row.get('model_short')} | {row.get('status')} | {row.get('embedding_dim')} | "
            f"{row.get('avg_nearest_neighbor_similarity')} | {row.get('pairs_ge_0_90')} | "
            f"{row.get('pairs_ge_0_85')} | {row.get('pairs_ge_0_80')} |"
        )
    lines.extend([
        "",
        "## Sentiment Model Distribution",
        "",
        "| Model | Positive | Neutral | Negative | Skipped | Scored Total |",
        "|---|---:|---:|---:|---:|---:|",
    ])
    for row in sent_dist:
        lines.append(
            f"| {row.get('model_short')} | {row.get('positive')} | {row.get('neutral')} | "
            f"{row.get('negative')} | {row.get('skipped')} | {row.get('scored_total')} |"
        )
    lines.extend([
        "",
        "## Codex Verification Notes",
        "",
        "- Financial-domain sentiment models should be weighted more heavily for market-risk interpretation than generic sentiment models.",
        "- English-only and Korean-only models are useful controls, but their skipped items must not be counted as neutral.",
        "- Multilingual embedding models are expected to produce better cross-language clustering than the English MiniLM control.",
        "- High-similarity pairs should be manually checked because boilerplate website text can inflate similarity.",
        "",
        "## Top Consensus Similarity Pairs",
        "",
        "| Avg Similarity | Models | Source A | Source B | Title A | Title B |",
        "|---:|---:|---|---|---|---|",
    ])
    for row in consensus[:12]:
        lines.append(
            f"| {row.get('avg_similarity')} | {row.get('model_count')} | {row.get('source_a')} | {row.get('source_b')} | "
            f"{short_md(row.get('title_a'))} | {short_md(row.get('title_b'))} |"
        )
    lines.extend([
        "",
        "## Highest Sentiment Disagreement Cases",
        "",
        "| Item | Source | Language | Labels | Title |",
        "|---|---|---|---|---|",
    ])
    for row in disagreements[:15]:
        lines.append(
            f"| `{row.get('item_id')}` | {row.get('source')} | {row.get('language')} | "
            f"`{row.get('labels_json')}` | {short_md(row.get('title'))} |"
        )
    lines.extend([
        "",
        "## Next Step",
        "",
        "Use `validation/claude_validation_prompt.md` and `validation/antigravity_validation_prompt.md` with the raw data. "
        "When their CSV review files are returned, Codex can merge them into final comparison tables and graphs.",
    ])
    (out_dir / "validation" / "codex_comparison_report.md").write_text("\n".join(lines), encoding="utf-8")


def short_md(value: Any, limit: int = 72) -> str:
    text = " ".join(str(value or "").replace("|", "/").split())
    return text if len(text) <= limit else text[: limit - 3].rstrip() + "..."


def write_readme(out_dir: Path, items: list[dict[str, Any]]) -> None:
    readme = f"""# PolitiMarket Model Comparison Export

Generated at: {datetime.now().isoformat(timespec='seconds')}

Raw crawled item count: {len(items)}

## Folder Map

- `raw_data/`: current raw `crawled_items` export, with no tagging or sentiment join.
- `embedding_similarity/`: model-specific embeddings, nearest-neighbor files, top similarity pairs, and consensus pairs.
- `sentiment/`: model sentiment predictions, distributions, agreement matrix, and disagreement cases.
- `graphs/`: SVG charts generated from local model outputs.
- `validation/`: Codex comparison report plus Claude and Antigravity verification prompts and result schema.

## How to Use With Claude / Antigravity

Give each tool:

1. `raw_data/model_input_pretag.jsonl`
2. `sentiment/sentiment_predictions.csv`
3. `embedding_similarity/consensus_top_pairs.csv`
4. `validation/claude_validation_prompt.md` or `validation/antigravity_validation_prompt.md`
5. `validation/external_result_schema.md`

After you receive their result CSV files, give them back to Codex. Codex will create merged comparison tables and graphs.
"""
    (out_dir / "README.md").write_text(readme, encoding="utf-8")


def write_metadata(out_dir: Path, items: list[dict[str, Any]]) -> None:
    metadata = {
        "generated_at": datetime.now().isoformat(timespec="seconds"),
        "database": str(DB_PATH),
        "row_count": len(items),
        "python": sys.version,
        "platform": platform.platform(),
        "source_group_counts": dict(sorted(Counter(item.get("source_group") or "" for item in items).items())),
        "source_counts": dict(sorted(Counter(item.get("source") or "" for item in items).items())),
        "language_counts": dict(sorted(Counter(item.get("language") or "" for item in items).items())),
        "embedding_models_requested": EMBEDDING_MODELS,
        "sentiment_models_requested": SENTIMENT_MODELS,
    }
    json_dump(out_dir / "metadata.json", metadata)


def main() -> int:
    EXPORTS_DIR.mkdir(parents=True, exist_ok=True)
    out_dir = EXPORTS_DIR / f"model_comparison_current_{now_stamp()}"
    out_dir.mkdir(parents=True, exist_ok=False)

    items = load_items()
    write_raw_data(out_dir, items)
    write_metadata(out_dir, items)

    emb_metrics, _consensus = run_embedding_models(out_dir, items)
    _statuses, _dist, _agreement = run_sentiment_models(out_dir, items)
    create_graphs(out_dir)
    create_validation_docs(out_dir, items)
    create_codex_report(out_dir, items)
    write_readme(out_dir, items)

    print(str(out_dir))
    print(json.dumps({
        "row_count": len(items),
        "embedding_models": len(EMBEDDING_MODELS),
        "sentiment_models": len(SENTIMENT_MODELS),
        "embedding_success": sum(1 for row in emb_metrics if row.get("status") == "success"),
    }, ensure_ascii=False, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
