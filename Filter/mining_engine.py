"""Shared text-mining helpers for filtering, deduplication, and entity updates."""

from __future__ import annotations

import hashlib
import itertools
import json
import math
import os
import re
import sqlite3
import time
from difflib import SequenceMatcher
from datetime import datetime, timezone
from pathlib import Path
from typing import Any
from urllib.parse import urlparse

import requests


BASE_DIR = Path(__file__).resolve().parents[1]
CRAWLING_DIR = BASE_DIR / "Crawling"
DB_PATH = CRAWLING_DIR / "crawler.db"
LLM_CONFIG_PATH = CRAWLING_DIR / "llm_config.json"
WON_CONFIG_PATH = LLM_CONFIG_PATH
BGE_MODEL_VERSION = os.environ.get("FILTER_BGE_MODEL", "BAAI/bge-m3")
DEFAULT_LLM_MODEL = os.environ.get("FILTER_LLM_MODEL", "gpt-5.5")
DEFAULT_LLM_REASONING_EFFORT = os.environ.get("FILTER_LLM_REASONING_EFFORT", "medium")
WON_DEFAULT_MODEL = DEFAULT_LLM_MODEL
NER_MODEL_VERSION = os.environ.get("FILTER_NER_MODEL", "market-entity-dictionary-gpt55-medium-v1")
CATEGORY_MODEL_VERSION = os.environ.get("FILTER_CATEGORY_MODEL", "bge-m3-category-gpt55-medium-v1")
ENABLE_QWEN_NER_REVIEW = os.environ.get("FILTER_QWEN_NER_REVIEW", "1") == "1"
ENABLE_QWEN_CATEGORY_REVIEW = os.environ.get("FILTER_QWEN_CATEGORY_REVIEW", "1") == "1"
QWEN_REVIEW_ERROR_COOLDOWN_SECONDS = int(os.environ.get("FILTER_QWEN_REVIEW_ERROR_COOLDOWN_SECONDS", "3600"))
WON_UPDATE_INTERVAL_SECONDS = int(os.environ.get("FILTER_WON_UPDATE_INTERVAL_SECONDS", str(6 * 60 * 60)))
ENABLE_WON_LOCAL_FALLBACK = os.environ.get("FILTER_WON_LOCAL_FALLBACK", "0") == "1"
LLM_USAGE_STATE_KEY = "llm_usage_summary"
ALLOWED_MARKET_TAGS = {"IT", "Energy", "Finance", "Healthcare", "Commodities", "Defense", "Chemicals", "Shipbuilding"}
BGE_DEDUP_LIMIT = int(os.environ.get("FILTER_BGE_DEDUP_LIMIT", "1000"))
BGE_DEDUP_CANDIDATE_THRESHOLD = float(os.environ.get("FILTER_BGE_DEDUP_CANDIDATE_THRESHOLD", "0.74"))
BGE_DEDUP_GROUP_THRESHOLD = float(os.environ.get("FILTER_BGE_DEDUP_GROUP_THRESHOLD", "0.80"))
BGE_DEDUP_WINDOW_HOURS = float(os.environ.get("FILTER_BGE_DEDUP_WINDOW_HOURS", "72"))
BGE_DEDUP_MAX_CANDIDATES_PER_ITEM = int(os.environ.get("FILTER_BGE_DEDUP_MAX_CANDIDATES_PER_ITEM", "12"))
DEDUP_PIPELINE_MODEL_VERSION = os.environ.get("FILTER_DEDUP_PIPELINE_MODEL", "lsa-tfidf-svd-128+bge-m3+gpt55")
DEDUP_LSA_CANDIDATE_THRESHOLD = float(os.environ.get("FILTER_DEDUP_LSA_CANDIDATE_THRESHOLD", "0.58"))
DEDUP_TITLE_CANDIDATE_THRESHOLD = float(os.environ.get("FILTER_DEDUP_TITLE_CANDIDATE_THRESHOLD", "0.72"))
DEDUP_COMPOSITE_THRESHOLD = float(os.environ.get("FILTER_DEDUP_COMPOSITE_THRESHOLD", "0.55"))
DEDUP_GPT_CONFIDENCE_THRESHOLD = float(os.environ.get("FILTER_DEDUP_GPT_CONFIDENCE_THRESHOLD", "0.68"))
DEDUP_MAX_CANDIDATE_PAIRS = int(os.environ.get("FILTER_DEDUP_MAX_CANDIDATE_PAIRS", "180"))
DEDUP_GPT_BATCH_SIZE = int(os.environ.get("FILTER_DEDUP_GPT_BATCH_SIZE", "4"))
ENABLE_GPT_DEDUP_AUDIT = os.environ.get("FILTER_ENABLE_GPT_DEDUP_AUDIT", "1") == "1"


KR_ENTITIES = [
    "한국",
    "대한민국",
    "Korea",
    "ROK",
    "서울",
    "KOSPI",
    "KOSDAQ",
    "원화",
    "한국은행",
    "금통위",
    "삼성전자",
    "SK하이닉스",
    "현대차",
    "LG에너지솔루션",
]
US_ENTITIES = [
    "미국",
    "United States",
    "U.S.",
    "US ",
    "Fed",
    "Federal Reserve",
    "Treasury",
    "White House",
    "Trump",
    "Biden",
    "S&P 500",
    "Nasdaq",
    "VIX",
    "dollar",
]
GEOPOLITICAL_ENTITIES = [
    "중국",
    "China",
    "러시아",
    "Russia",
    "대만",
    "Taiwan",
    "우크라이나",
    "Ukraine",
    "EU",
    "European Union",
    "중동",
    "Middle East",
]
FINANCIAL_TERMS = [
    "금융",
    "시장",
    "증시",
    "주식",
    "채권",
    "국채",
    "금리",
    "환율",
    "달러",
    "물가",
    "인플레이션",
    "수출",
    "공급망",
    "관세",
    "제재",
    "반도체",
    "배터리",
    "에너지",
    "원유",
    "가스",
    "희토류",
    "bank",
    "bond",
    "market",
    "stock",
    "rate",
    "inflation",
    "tariff",
    "sanction",
    "semiconductor",
    "energy",
    "oil",
    "supply chain",
]
POLICY_RISK_TERMS = [
    "관세",
    "제재",
    "수출통제",
    "규제",
    "금리",
    "전쟁",
    "안보",
    "군사",
    "tariff",
    "sanction",
    "export control",
    "restriction",
    "rate",
    "war",
    "security",
    "military",
]

KR_ENTITIES.extend([
    "한국", "대한민국", "서울", "원화", "한국은행", "금융위원회", "금융감독원",
    "코스피", "코스닥", "삼성전자", "SK하이닉스", "현대차", "LG에너지솔루션",
    "한미", "한국 기업", "국내 증시",
])
US_ENTITIES.extend([
    "미국", "미 정부", "워싱턴", "연준", "연방준비제도", "재무부", "백악관",
    "트럼프", "바이든", "나스닥", "달러", "미 국채", "미중",
])
GEOPOLITICAL_ENTITIES.extend([
    "중국", "러시아", "대만", "우크라이나", "유럽연합", "EU", "중동",
    "북한", "남중국해", "미중 갈등",
])
FINANCIAL_TERMS.extend([
    "금융", "시장", "증시", "주식", "채권", "국채", "금리", "환율", "달러",
    "물가", "인플레이션", "수출", "공급망", "관세", "제재", "반도체",
    "배터리", "에너지", "원유", "유가", "가스", "희토류", "은행", "신용",
    "무역", "투자", "코스피", "나스닥", "S&P",
])
POLICY_RISK_TERMS.extend([
    "관세", "제재", "수출통제", "규제", "금리", "전쟁", "안보", "군사",
    "무역분쟁", "공급망 차질", "제한", "금수", "긴장", "분쟁",
])

CATEGORY_REVIEW_GUIDE = {
    "IT": "AI, LLM, 반도체, GPU, 데이터센터, 클라우드, 사이버 보안, 첨단 칩 수출통제, 플랫폼/로봇.",
    "Energy": "원유, 유가, 천연가스, LNG, 전력망, 원전, 재생에너지, 정유, 에너지 제재와 안보.",
    "Finance": "금리, 환율, 달러, 물가, 채권, 증시, 은행, 관세, 무역, GDP, 재정, 세금, 유동성.",
    "Healthcare": "제약, 바이오, 백신, 신약 승인, 임상, 병원, 의료기기, 건강보험, 감염병.",
    "Commodities": "금, 은, 구리, 철광석, 철강, 희토류, 리튬 원료, 알루미늄, 곡물, 광물 공급망.",
    "Defense": "전쟁, 제재, 미사일, 드론, 핵무기, 군사 동맹, 방산, 해군, 북한, 대만, 우크라이나.",
    "Chemicals": "석유화학, 배터리 소재, 양극재, 음극재, 비료, 리튬 가공, 플라스틱, 화학 공장.",
    "Shipbuilding": "조선, 선박 수주, LNG선, 탱커, 컨테이너선, 해양플랜트, 항만, 함정/잠수함 건조.",
}

SOURCE_WEIGHTS = {
    "Axios": 0.90,
    "News_Newsis": 0.80,
    "ThinkTank_CSIS": 0.70,
    "ThinkTank_PIIE": 0.70,
    "ThinkTank_Brookings": 0.70,
    "TruthSocial": 0.27,
    "News_PeopleCN_KO": 0.12,
    "Gov_Kremlin": 0.12,
}
GROUP_WEIGHTS = {
    "thinktank": 0.70,
    "news": 0.80,
    "truth": 0.27,
    "china": 0.12,
    "russia": 0.12,
    "gov": 0.55,
    "x": 0.50,
}


def utc_now() -> str:
    return datetime.now(timezone.utc).isoformat(timespec="seconds")


def json_loads(value: Any, default: Any) -> Any:
    try:
        return json.loads(value) if value else default
    except Exception:
        return default


def normalize_space(value: Any) -> str:
    return " ".join(str(value or "").split())


def text_hash(value: str) -> str:
    return hashlib.sha256(value.encode("utf-8", errors="ignore")).hexdigest()


def compact_item_text(item: dict[str, Any], content_limit: int = 1200) -> str:
    title = normalize_space(item.get("title"))
    content = normalize_space(item.get("content"))
    if title and title not in content[:200]:
        return f"{title}. {content[:content_limit]}".strip()
    return (content or title)[:content_limit]


def contains_any(text: str, terms: list[str]) -> list[str]:
    lower = text.lower()
    matches: list[str] = []
    seen: set[str] = set()
    for term in terms:
        normalized = normalize_space(term).strip()
        if not normalized:
            continue
        needle = normalized.lower()
        matched = False
        if re.fullmatch(r"[a-z0-9 .&+/()-]+", needle):
            pattern = r"(?<![a-z0-9])" + re.escape(needle).replace(r"\ ", r"\s+") + r"(?![a-z0-9])"
            matched = bool(re.search(pattern, lower))
        else:
            matched = needle in lower
        if matched and needle not in seen:
            seen.add(needle)
            matches.append(normalized)
    return matches


def load_dynamic_entities(conn: sqlite3.Connection) -> list[dict[str, Any]]:
    try:
        rows = conn.execute(
            """
            SELECT entity_name, category, synonyms_json, association_rules_json,
                   rationale, estimated_market_impact, source
            FROM entity_dictionary
            """
        ).fetchall()
    except sqlite3.Error:
        return []
    entities = []
    for row in rows:
        entity = dict(row)
        entity["synonyms"] = json_loads(entity.pop("synonyms_json", "[]"), [])
        entity["association_rules"] = json_loads(entity.pop("association_rules_json", "[]"), [])
        entities.append(entity)
    return entities


def seed_builtin_entities(conn: sqlite3.Connection) -> None:
    rows = [
        ("KOSPI", "GPE", ["코스피", "Korea Composite Stock Price Index"], ["한국 시장", "증시"], "Korean benchmark equity index", "HIGH"),
        ("Federal Reserve", "ORG", ["Fed", "FOMC", "연준"], ["금리", "달러", "채권"], "US monetary policy driver", "HIGH"),
        ("US-China tariff", "EVENT", ["관세", "tariff", "trade war"], ["미국", "중국", "수출"], "Tariff conflict affects Korean exporters", "HIGH"),
        ("Semiconductor export control", "EVENT", ["수출통제", "export control", "chip restriction"], ["반도체", "중국", "미국"], "Chip restrictions affect Korean tech names", "HIGH"),
    ]
    now = utc_now()
    for entity_name, category, synonyms, rules, rationale, impact in rows:
        conn.execute(
            """
            INSERT INTO entity_dictionary (
                entity_name, category, synonyms_json, association_rules_json,
                rationale, estimated_market_impact, source, updated_at
            )
            VALUES (?, ?, ?, ?, ?, ?, 'builtin', ?)
            ON CONFLICT(entity_name) DO NOTHING
            """,
            (entity_name, category, json.dumps(synonyms, ensure_ascii=False), json.dumps(rules, ensure_ascii=False), rationale, impact, now),
        )


def evaluate_ner_relevance(item: dict[str, Any], conn: sqlite3.Connection | None = None) -> dict[str, Any]:
    text = compact_item_text(item, 3000)
    source = str(item.get("source") or "")
    source_group = str(item.get("source_group") or "").lower()

    kr_matches = contains_any(text, KR_ENTITIES)
    us_matches = contains_any(text, US_ENTITIES)
    geopolitical_matches = contains_any(text, GEOPOLITICAL_ENTITIES)
    financial_matches = contains_any(text, FINANCIAL_TERMS)
    policy_matches = contains_any(text, POLICY_RISK_TERMS)

    dynamic_matches: list[dict[str, Any]] = []
    if conn is not None:
        for entity in load_dynamic_entities(conn):
            names = [entity.get("entity_name", ""), *entity.get("synonyms", [])]
            hits = contains_any(text, [str(name) for name in names])
            if hits:
                dynamic_matches.append(
                    {
                        "entity_name": entity.get("entity_name"),
                        "category": entity.get("category"),
                        "impact": entity.get("estimated_market_impact"),
                        "hits": hits,
                    }
                )

    direct = bool(kr_matches and us_matches and financial_matches)
    china_russia = source_group in {"china", "russia"}
    indirect = bool(china_russia and (us_matches or geopolitical_matches) and (policy_matches or financial_matches))
    thinktank = bool(source_group == "thinktank" and (financial_matches or policy_matches) and (us_matches or kr_matches or geopolitical_matches))
    dynamic = bool(dynamic_matches and (financial_matches or policy_matches))
    truth_policy = bool(source_group == "truth" and (financial_matches or policy_matches))
    evidence_score = (
        len(kr_matches) * 1.2
        + len(us_matches) * 1.3
        + len(geopolitical_matches) * 0.8
        + len(financial_matches) * 1.1
        + len(policy_matches) * 1.2
        + len(dynamic_matches) * 1.5
    )
    ambiguous = bool(
        china_russia
        and not indirect
        and (
            evidence_score >= 2.0
            or bool(financial_matches and geopolitical_matches)
            or bool(policy_matches and (us_matches or geopolitical_matches))
        )
    )

    passed = direct or indirect or thinktank or dynamic or truth_policy
    if passed:
        reason = "direct_kr_us_finance" if direct else "indirect_policy_market"
        if thinktank:
            reason = "thinktank_policy_market"
        if dynamic:
            reason = "dynamic_entity_dictionary"
        if truth_policy:
            reason = "truth_policy_market"
    else:
        reason = "no_ner_market_link"

    return {
        "passed": passed,
        "ambiguous": ambiguous,
        "model_version": NER_MODEL_VERSION,
        "evidence_score": round(evidence_score, 4),
        "reason": reason,
        "matched_entities": {
            "kr": kr_matches,
            "us": us_matches,
            "geopolitical": geopolitical_matches,
            "dynamic": dynamic_matches,
        },
        "matched_terms": {
            "financial": financial_matches,
            "policy": policy_matches,
        },
    }


def record_ner_event(conn: sqlite3.Connection, item: dict[str, Any], evaluation: dict[str, Any]) -> None:
    conn.execute(
        """
        INSERT INTO ner_filter_events (
            item_id, content_hash, decision, reason,
            matched_entities_json, matched_terms_json, created_at
        )
        VALUES (?, ?, ?, ?, ?, ?, ?)
        """,
        (
            item.get("item_id") or item.get("id"),
            item.get("content_hash"),
            "pass" if evaluation.get("passed") else "reject",
            evaluation.get("reason", ""),
            json.dumps(evaluation.get("matched_entities") or {}, ensure_ascii=False),
            json.dumps(evaluation.get("matched_terms") or {}, ensure_ascii=False),
            utc_now(),
        ),
    )


def recent_qwen_error(conn: sqlite3.Connection, key: str) -> bool:
    row = conn.execute(
        "SELECT value, updated_at FROM engine_state WHERE key = ?",
        (key,),
    ).fetchone()
    if not row:
        return False
    value = str(row["value"] or "")
    if "HTTP 402" not in value and "depleted" not in value.lower():
        return False
    try:
        updated = datetime.fromisoformat(str(row["updated_at"]).replace("Z", "+00:00"))
        return (datetime.now(timezone.utc) - updated).total_seconds() < QWEN_REVIEW_ERROR_COOLDOWN_SECONDS
    except Exception:
        return True


def feedback_exists(conn: sqlite3.Connection, item_id: str, source: str) -> bool:
    try:
        row = conn.execute(
            "SELECT 1 FROM label_feedback WHERE item_id = ? AND source = ? LIMIT 1",
            (item_id, source),
        ).fetchone()
        return bool(row)
    except sqlite3.Error:
        return False


def insert_qwen_feedback(
    conn: sqlite3.Connection,
    item: dict[str, Any],
    result: dict[str, Any],
    review: dict[str, Any],
    source: str,
    approved: bool,
) -> None:
    item_id = str(item.get("item_id") or item.get("id") or "")
    content_hash = str(item.get("content_hash") or "")
    if not item_id or not content_hash or feedback_exists(conn, item_id, source):
        return
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
            content_hash,
            result.get("primary_tag") or "",
            result.get("sentiment_label") or "",
            review.get("corrected_tag") or review.get("primary_tag") or result.get("primary_tag") or "",
            review.get("corrected_sentiment") or result.get("sentiment_label") or "",
            1 if approved else 0,
            source,
            json.dumps(review, ensure_ascii=False)[:1800],
            source,
            utc_now(),
        ),
    )


def build_qwen_ner_review_prompt(item: dict[str, Any], result: dict[str, Any], evaluation: dict[str, Any]) -> str:
    sample = {
        "item_id": item.get("item_id") or item.get("id"),
        "source": item.get("source"),
        "source_group": item.get("source_group"),
        "country": item.get("country"),
        "language": item.get("language"),
        "title": item.get("title"),
        "content": compact_item_text(item, 1600),
        "current_tag": result.get("primary_tag"),
        "current_sentiment": result.get("sentiment_label"),
        "ner_evaluation": evaluation,
    }
    return (
        "You are checking whether a China/Russia source article should be included in a Korean/US market dataset. "
        "Use only the provided title/content. Include the article only if it plausibly affects Korean or US financial markets, "
        "KOSPI, S&P 500, FX, rates, bonds, VIX, exports, semiconductors, energy, commodities, defense, healthcare, chemicals, "
        "shipbuilding, or supply chains. Return strict JSON only with schema: "
        '{"decision":"include|exclude","corrected_tag":"IT|Energy|Finance|Healthcare|Commodities|Defense|Chemicals|Shipbuilding",'
        '"corrected_sentiment":"Positive|Neutral|Warning|Panic","confidence":0.0,"rationale":"short reason",'
        '"detected_entities":[{"entity_name":"string","synonyms":["string"],"category":"GPE|PERSON|ORG|EVENT",'
        '"financial_relevance_rationale":"string","estimated_market_impact":"HIGH|MEDIUM|LOW","suggested_rules":["string"]}]}. '
        "Do not include chain-of-thought. Sample:\n"
        + json.dumps(sample, ensure_ascii=False, indent=2)
    )


def normalize_qwen_review(payload: dict[str, Any], fallback_tag: str, fallback_sentiment: str) -> dict[str, Any]:
    decision = normalize_space(payload.get("decision")).lower()
    if decision not in {"include", "exclude", "restore", "keep_excluded"}:
        decision = "include" if str(payload.get("market_relevant", "")).lower() in {"true", "yes", "1"} else "exclude"
    if decision == "restore":
        decision = "include"
    if decision == "keep_excluded":
        decision = "exclude"
    tag = normalize_space(payload.get("corrected_tag") or payload.get("primary_tag") or fallback_tag)
    if tag not in ALLOWED_MARKET_TAGS:
        tag = fallback_tag if fallback_tag in ALLOWED_MARKET_TAGS else "Finance"
    sentiment = normalize_space(payload.get("corrected_sentiment") or fallback_sentiment)
    if sentiment not in {"Positive", "Neutral", "Warning", "Panic"}:
        sentiment = fallback_sentiment if fallback_sentiment in {"Positive", "Neutral", "Warning", "Panic"} else "Neutral"
    try:
        confidence = float(payload.get("confidence") or 0)
    except Exception:
        confidence = 0.0
    return {
        **payload,
        "decision": decision,
        "corrected_tag": tag,
        "corrected_sentiment": sentiment,
        "confidence": max(0.0, min(1.0, confidence)),
        "rationale": normalize_space(payload.get("rationale")),
    }


def apply_qwen_ner_review(conn: sqlite3.Connection, item: dict[str, Any], result: dict[str, Any], evaluation: dict[str, Any]) -> dict[str, Any]:
    if not ENABLE_QWEN_NER_REVIEW or not evaluation.get("ambiguous"):
        return result
    config = read_won_config()
    if not config.get("enabled") or recent_qwen_error(conn, "last_qwen_ner_review_error"):
        return result
    try:
        generated = call_won_reasoning_api(build_qwen_ner_review_prompt(item, result, evaluation), config)
        payload = extract_json_object(generated)
        review = normalize_qwen_review(payload, str(result.get("primary_tag") or "Finance"), str(result.get("sentiment_label") or "Neutral"))
        review["review_type"] = "qwen_ner_market_filter"
        review["model_id"] = config.get("model_id") or WON_DEFAULT_MODEL
        evaluation["qwen_review"] = review
        upsert_won_entities(conn, payload)
        if review["decision"] == "include" and review["confidence"] >= 0.55:
            result["is_excluded"] = False
            result["reason"] = "gpt55_ner_market_review"
            result["primary_tag"] = review["corrected_tag"]
            result["sentiment_label"] = review["corrected_sentiment"]
            result["relevance_score"] = max(float(result.get("relevance_score") or 0), 0.55)
            result["confidence"] = max(float(result.get("confidence") or 0), review["confidence"])
        elif review["decision"] == "exclude" and review["confidence"] >= 0.65:
            result["is_excluded"] = True
            result["reason"] = "gpt55_ner_market_excluded"
        insert_qwen_feedback(conn, item, result, review, "gpt55_ner_market_filter", approved=review["confidence"] >= 0.70)
        set_engine_state(conn, "last_qwen_ner_review_result", json.dumps({"status": "ok", "item_id": item.get("item_id") or item.get("id"), **review}, ensure_ascii=False))
    except Exception as exc:
        set_engine_state(conn, "last_qwen_ner_review_error", str(exc)[:1000])
    return result


def build_qwen_category_review_prompt(item: dict[str, Any], result: dict[str, Any]) -> str:
    sample = {
        "item_id": item.get("item_id") or item.get("id"),
        "source": item.get("source"),
        "source_group": item.get("source_group"),
        "country": item.get("country"),
        "language": item.get("language"),
        "title": item.get("title"),
        "content": compact_item_text(item, 1600),
        "bge_primary_tag": result.get("primary_tag"),
        "bge_top_score": result.get("bge_category_top_score"),
        "bge_second_score": result.get("bge_category_second_score"),
        "bge_margin": result.get("bge_category_margin"),
        "category_scores": result.get("category_scores", [])[:4],
        "category_guide": CATEGORY_REVIEW_GUIDE,
        "current_sentiment": result.get("sentiment_label"),
    }
    return (
        "You are choosing the best market category for a political/economic news article. "
        "Choose exactly one category from IT, Energy, Finance, Healthcare, Commodities, Defense, Chemicals, Shipbuilding. "
        "Use the category guide, article context, and BGE candidates; do not rely on a single keyword. "
        "If the article is administrative or general politics with no plausible market/sector impact, keep the closest category "
        "but use low confidence. Return strict JSON only with schema: "
        '{"decision":"include","corrected_tag":"IT|Energy|Finance|Healthcare|Commodities|Defense|Chemicals|Shipbuilding",'
        '"corrected_sentiment":"Positive|Neutral|Warning|Panic","confidence":0.0,"rationale":"short reason"}. '
        "Do not include chain-of-thought. Sample:\n"
        + json.dumps(sample, ensure_ascii=False, indent=2)
    )


def apply_qwen_category_review(conn: sqlite3.Connection, item: dict[str, Any], result: dict[str, Any]) -> dict[str, Any]:
    if not ENABLE_QWEN_CATEGORY_REVIEW or not result.get("category_ambiguous"):
        return result
    relevance = float(result.get("relevance_score") or 0)
    ner_filter = result.get("ner_filter") if isinstance(result.get("ner_filter"), dict) else {}
    if result.get("is_excluded") and relevance < 0.45 and not ner_filter.get("passed"):
        return result
    config = read_won_config()
    if not config.get("enabled") or recent_qwen_error(conn, "last_qwen_category_review_error"):
        return result
    try:
        generated = call_won_reasoning_api(build_qwen_category_review_prompt(item, result), config)
        payload = extract_json_object(generated)
        review = normalize_qwen_review(payload, str(result.get("primary_tag") or "Finance"), str(result.get("sentiment_label") or "Neutral"))
        review["review_type"] = "qwen_category_review"
        review["model_id"] = config.get("model_id") or WON_DEFAULT_MODEL
        result["qwen_category_review"] = review
        if review["confidence"] >= 0.55 and review["corrected_tag"] in ALLOWED_MARKET_TAGS:
            result["primary_tag"] = review["corrected_tag"]
            result["sentiment_label"] = review["corrected_sentiment"]
            result["confidence"] = max(float(result.get("confidence") or 0), review["confidence"])
            result["category_source"] = "bge_m3_gpt55_review"
            result["reason"] = None if not result.get("is_excluded") else result.get("reason")
            result["tags"] = [
                {"tag": review["corrected_tag"], "score": round(review["confidence"], 4), "hits": 0, "source": "gpt55_category_review"},
                *[tag for tag in result.get("tags", []) if tag.get("tag") != review["corrected_tag"]][:3],
            ]
            result["matching_keywords"] = result["tags"]
        insert_qwen_feedback(conn, item, result, review, "gpt55_category_review", approved=review["confidence"] >= 0.70)
        set_engine_state(conn, "last_qwen_category_review_result", json.dumps({"status": "ok", "item_id": item.get("item_id") or item.get("id"), **review}, ensure_ascii=False))
    except Exception as exc:
        set_engine_state(conn, "last_qwen_category_review_error", str(exc)[:1000])
    return result


def apply_ner_gate(conn: sqlite3.Connection, item: dict[str, Any], result: dict[str, Any]) -> dict[str, Any]:
    evaluation = evaluate_ner_relevance(item, conn)
    result["ner_filter"] = evaluation
    if evaluation["passed"]:
        result["relevance_score"] = max(float(result.get("relevance_score") or 0), 0.45)
        result["matching_keywords"] = list(result.get("matching_keywords") or []) + [
            {"tag": "NER", "score": 0.6, "hits": 1, "reason": evaluation["reason"]}
        ]
    elif float(result.get("relevance_score") or 0) < 0.45:
        result["is_excluded"] = True
        result["reason"] = "ner_relevance_filter"
    result = apply_qwen_ner_review(conn, item, result, evaluation)
    record_ner_event(conn, item, evaluation)
    return result


def media_weight_for(source: Any, source_group: Any = "") -> float:
    source_text = str(source or "")
    group_text = str(source_group or "").lower()
    for prefix, weight in SOURCE_WEIGHTS.items():
        if source_text.startswith(prefix):
            return weight
    return GROUP_WEIGHTS.get(group_text, 0.45)


def read_won_config() -> dict[str, Any]:
    config = {
        "model_id": WON_DEFAULT_MODEL,
        "openai_api_key": "",
        "reasoning_effort": DEFAULT_LLM_REASONING_EFFORT,
        "api_url": "https://api.openai.com/v1/responses",
        "enabled": False,
    }
    if LLM_CONFIG_PATH.exists():
        try:
            saved = json.loads(LLM_CONFIG_PATH.read_text(encoding="utf-8"))
            if isinstance(saved, dict):
                for key in config:
                    if key in saved:
                        config[key] = saved[key]
                if not config.get("openai_api_key") and saved.get("api_key"):
                    config["openai_api_key"] = saved.get("api_key", "")
        except Exception:
            pass
    env_key = os.environ.get("OPENAI_API_KEY", "").strip()
    if env_key:
        config["openai_api_key"] = env_key
    config["model_id"] = str(config.get("model_id") or WON_DEFAULT_MODEL)
    if config["model_id"] != WON_DEFAULT_MODEL:
        config["model_id"] = WON_DEFAULT_MODEL
    effort = str(config.get("reasoning_effort") or DEFAULT_LLM_REASONING_EFFORT).lower()
    if effort not in {"none", "low", "medium", "high", "xhigh"}:
        effort = DEFAULT_LLM_REASONING_EFFORT
    config["reasoning_effort"] = effort
    config["api_url"] = os.environ.get("OPENAI_RESPONSES_URL", str(config.get("api_url") or "https://api.openai.com/v1/responses"))
    key_error = ""
    token = str(config.get("openai_api_key") or "").strip()
    if token:
        try:
            validate_openai_api_key(token)
        except ValueError as exc:
            key_error = str(exc)
            config["openai_api_key"] = ""
    if key_error:
        config["key_error"] = key_error
    if env_key and not LLM_CONFIG_PATH.exists():
        config["enabled"] = True
    config["enabled"] = bool(config.get("enabled") and str(config.get("openai_api_key") or "").strip())
    return config


def mask_secret(value: str) -> str:
    token = str(value or "")
    if not token:
        return ""
    if len(token) <= 10:
        return token[:2] + "*" * max(0, len(token) - 4) + token[-2:]
    return token[:6] + "*" * (len(token) - 10) + token[-4:]


def validate_openai_api_key(api_key: str) -> str:
    token = str(api_key or "").strip()
    if not token:
        raise ValueError("OPENAI_API_KEY or llm_config.json openai_api_key is required.")
    if any(ch.isspace() for ch in token):
        raise ValueError("OpenAI API key contains whitespace/newlines.")
    if not token.startswith("sk-"):
        raise ValueError(
            f"OpenAI API key format is invalid. Expected an sk- or sk-proj- key, got {mask_secret(token)}."
        )
    return token


def estimate_token_count(value: Any) -> int:
    text = normalize_space(value)
    if not text:
        return 0
    return max(1, math.ceil(len(text) / 4))


def record_llm_usage(
    model: str,
    api_url: str,
    prompt: str,
    output: str,
    mode: str = "api",
    provider_usage: dict[str, Any] | None = None,
) -> None:
    """Persist a lightweight local usage ledger for LLM calls.

    Hosted APIs do not always return billing usage. When they do, use it;
    otherwise keep an approximate token count so the dashboard can still show
    whether the app is making LLM calls and roughly how much text is involved.
    """
    usage = provider_usage if isinstance(provider_usage, dict) else {}
    used_provider_usage = any(key in usage for key in ("prompt_tokens", "completion_tokens", "total_tokens"))

    prompt_tokens = usage.get("prompt_tokens", usage.get("input_tokens"))
    completion_tokens = usage.get("completion_tokens", usage.get("output_tokens"))
    total_tokens = usage.get("total_tokens")
    try:
        prompt_tokens = int(prompt_tokens)
    except Exception:
        prompt_tokens = estimate_token_count(prompt)
    try:
        completion_tokens = int(completion_tokens)
    except Exception:
        completion_tokens = estimate_token_count(output)
    try:
        total_tokens = int(total_tokens)
    except Exception:
        total_tokens = int(prompt_tokens) + int(completion_tokens)

    now = utc_now()
    model_key = model or WON_DEFAULT_MODEL
    try:
        with sqlite3.connect(DB_PATH) as conn:
            conn.row_factory = sqlite3.Row
            conn.execute(
                """
                CREATE TABLE IF NOT EXISTS engine_state (
                    key TEXT PRIMARY KEY,
                    value TEXT NOT NULL,
                    updated_at TEXT NOT NULL
                )
                """
            )
            row = conn.execute("SELECT value FROM engine_state WHERE key = ?", (LLM_USAGE_STATE_KEY,)).fetchone()
            summary = json_loads(row["value"] if row else "", {})
            if not isinstance(summary, dict):
                summary = {}

            summary.setdefault("total_calls", 0)
            summary.setdefault("estimated_calls", 0)
            summary.setdefault("provider_reported_calls", 0)
            summary.setdefault("total_prompt_tokens", 0)
            summary.setdefault("total_completion_tokens", 0)
            summary.setdefault("total_tokens", 0)
            summary.setdefault("by_model", {})

            summary["total_calls"] += 1
            summary["total_prompt_tokens"] += prompt_tokens
            summary["total_completion_tokens"] += completion_tokens
            summary["total_tokens"] += total_tokens
            if used_provider_usage:
                summary["provider_reported_calls"] += 1
            else:
                summary["estimated_calls"] += 1

            by_model = summary["by_model"]
            model_summary = by_model.setdefault(
                model_key,
                {
                    "calls": 0,
                    "prompt_tokens": 0,
                    "completion_tokens": 0,
                    "total_tokens": 0,
                    "last_call_at": "",
                },
            )
            model_summary["calls"] += 1
            model_summary["prompt_tokens"] += prompt_tokens
            model_summary["completion_tokens"] += completion_tokens
            model_summary["total_tokens"] += total_tokens
            model_summary["last_call_at"] = now

            summary["last_call"] = {
                "at": now,
                "mode": mode,
                "model": model_key,
                "api_url": api_url or "local",
                "prompt_tokens": prompt_tokens,
                "completion_tokens": completion_tokens,
                "total_tokens": total_tokens,
                "provider_reported": used_provider_usage,
            }
            conn.execute(
                """
                INSERT INTO engine_state (key, value, updated_at)
                VALUES (?, ?, ?)
                ON CONFLICT(key) DO UPDATE SET value = excluded.value, updated_at = excluded.updated_at
                """,
                (LLM_USAGE_STATE_KEY, json.dumps(summary, ensure_ascii=False), now),
            )
    except Exception:
        pass


def extract_json_object(text: str) -> dict[str, Any]:
    cleaned = text.strip()
    cleaned = re.sub(r"^```(?:json)?\s*", "", cleaned)
    cleaned = re.sub(r"\s*```$", "", cleaned)
    start = cleaned.find("{")
    end = cleaned.rfind("}")
    if start >= 0 and end > start:
        cleaned = cleaned[start : end + 1]
    decoder = json.JSONDecoder()
    try:
        value, _idx = decoder.raw_decode(cleaned)
        if isinstance(value, dict):
            return value
    except json.JSONDecodeError:
        pass
    for idx, char in enumerate(cleaned):
        if char != "{":
            continue
        try:
            value, _end = decoder.raw_decode(cleaned[idx:])
        except json.JSONDecodeError:
            continue
        if isinstance(value, dict):
            return value
    return json.loads(cleaned)


def openai_response_text(data: dict[str, Any]) -> str:
    if data.get("output_text"):
        return str(data.get("output_text") or "")
    parts: list[str] = []
    for output in data.get("output") or []:
        if not isinstance(output, dict):
            continue
        for content in output.get("content") or []:
            if isinstance(content, dict):
                text = content.get("text") or content.get("output_text")
                if text:
                    parts.append(str(text))
            elif isinstance(content, str):
                parts.append(content)
    return "\n".join(parts).strip()


def call_won_reasoning_api(prompt: str, config: dict[str, Any] | None = None) -> str:
    config = config or read_won_config()
    model = config.get("model_id") or WON_DEFAULT_MODEL
    api_url = config.get("api_url") or "https://api.openai.com/v1/responses"
    token = config.get("openai_api_key") or os.environ.get("OPENAI_API_KEY", "")
    token = validate_openai_api_key(token)
    headers = {
        "Authorization": f"Bearer {token}",
        "Content-Type": "application/json",
    }
    payload = {
        "model": model,
        "input": [
            {"role": "developer", "content": "Return strict JSON only. Do not include chain-of-thought."},
            {"role": "user", "content": prompt},
        ],
        "reasoning": {"effort": str(config.get("reasoning_effort") or DEFAULT_LLM_REASONING_EFFORT)},
        "max_output_tokens": int(config.get("max_output_tokens") or os.environ.get("OPENAI_MAX_OUTPUT_TOKENS", "1800")),
    }
    response = requests.post(api_url, headers=headers, json=payload, timeout=120)
    if response.status_code >= 400:
        raise RuntimeError(f"OpenAI API HTTP {response.status_code}: {response.text[:500]}")
    data = response.json()
    provider_usage = data.get("usage") if isinstance(data, dict) else None
    if not isinstance(data, dict):
        text = json.dumps(data, ensure_ascii=False)
    else:
        if data.get("error"):
            raise RuntimeError(str(data["error"]))
        text = openai_response_text(data) or json.dumps(data, ensure_ascii=False)
    record_llm_usage(model, api_url, prompt, text, "openai_responses", provider_usage)
    return text


def call_won_reasoning_local(prompt: str, config: dict[str, Any] | None = None) -> str:
    config = config or read_won_config()
    model_name = config.get("model_id") or WON_DEFAULT_MODEL
    token = config.get("hf_token") or None
    import torch  # type: ignore
    from transformers import AutoModelForCausalLM, AutoTokenizer  # type: ignore

    tokenizer = AutoTokenizer.from_pretrained(model_name, token=token)
    model = AutoModelForCausalLM.from_pretrained(
        model_name,
        token=token,
        torch_dtype=torch.float16 if torch.cuda.is_available() else torch.float32,
        low_cpu_mem_usage=True,
    )
    device = "cuda" if torch.cuda.is_available() else "cpu"
    model.to(device)
    model.eval()
    messages = [
        {"role": "system", "content": "Return strict JSON only."},
        {"role": "user", "content": prompt},
    ]
    if hasattr(tokenizer, "apply_chat_template"):
        input_text = tokenizer.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)
    else:
        input_text = f"System: Return strict JSON only.\nUser: {prompt}\nAssistant:"
    encoded = tokenizer(input_text, return_tensors="pt", truncation=True, max_length=4096).to(device)
    with torch.no_grad():
        output = model.generate(
            **encoded,
            max_new_tokens=900,
            do_sample=False,
            pad_token_id=tokenizer.eos_token_id,
        )
    generated = output[0][encoded["input_ids"].shape[-1]:]
    result = tokenizer.decode(generated, skip_special_tokens=True)
    record_llm_usage(model_name, "local", prompt, result, "local")
    return result


def build_won_entity_prompt(samples: list[dict[str, Any]]) -> str:
    sample_payload = [
        {
            "source": item.get("source"),
            "source_group": item.get("source_group"),
            "title": item.get("title"),
            "content": compact_item_text(item, 900),
            "reject_reason": item.get("exclude_reason") or item.get("reason"),
        }
        for item in samples
    ]
    return (
        "You are a financial-market NER dictionary updater for Korean and US political/geopolitical news. "
        "Review rejected crawler samples and extract only entities that are likely to affect KOSPI, S&P 500, FX, rates, bonds, VIX, exports, semiconductors, energy, or defense. "
        "Return strict JSON with this schema: "
        '{"detected_entities":[{"entity_name":"string","synonyms":["string"],"category":"GPE|PERSON|ORG|EVENT","financial_relevance_rationale":"string","estimated_market_impact":"HIGH|MEDIUM|LOW","suggested_rules":["string"]}]}. '
        "Ignore generic noise. Samples:\n"
        + json.dumps(sample_payload, ensure_ascii=False, indent=2)
    )


def ensure_review_tables(conn: sqlite3.Connection) -> None:
    conn.executescript(
        """
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
        CREATE INDEX IF NOT EXISTS idx_llm_excluded_decision ON llm_excluded_reviews(decision);
        """
    )


def build_excluded_review_prompt(items: list[dict[str, Any]]) -> str:
    samples = []
    for item in items:
        samples.append(
            {
                "item_id": item.get("item_id") or item.get("id"),
                "source": item.get("source"),
                "source_group": item.get("source_group"),
                "country": item.get("country"),
                "language": item.get("language"),
                "title": item.get("title"),
                "content": compact_item_text(item, 1200),
                "current_tag": item.get("primary_tag"),
                "current_sentiment": item.get("sentiment_label"),
                "current_exclude_reason": item.get("exclude_reason") or item.get("reason"),
                "relevance_score": item.get("relevance_score"),
                "confidence": item.get("confidence"),
            }
        )
    return (
        "You are reviewing crawler items that were excluded by a market-news filter. "
        "Decide whether each item should stay excluded or be restored into the financial analysis dataset. "
        "Restore only when the text has a plausible connection to Korean/US financial markets, KOSPI, S&P 500, FX, rates, bonds, VIX, exports, semiconductors, energy, commodities, defense, healthcare, chemicals, or shipbuilding. "
        "If restored, choose corrected_tag from exactly one of: IT, Energy, Finance, Healthcare, Commodities, Defense, Chemicals, Shipbuilding. "
        "Choose corrected_sentiment from exactly one of: Positive, Neutral, Warning, Panic. "
        "Also extract high/medium impact entities useful for future filtering. "
        "Return strict JSON only with schema: "
        '{"reviews":[{"item_id":"string","decision":"restore|keep_excluded","corrected_tag":"string","corrected_sentiment":"string","confidence":0.0,"rationale":"string"}],'
        '"detected_entities":[{"entity_name":"string","synonyms":["string"],"category":"GPE|PERSON|ORG|EVENT","financial_relevance_rationale":"string","estimated_market_impact":"HIGH|MEDIUM|LOW","suggested_rules":["string"]}]}. '
        "Do not include chain-of-thought; use concise rationales. Items:\n"
        + json.dumps(samples, ensure_ascii=False, indent=2)
    )


def normalize_review_payload(payload: dict[str, Any]) -> list[dict[str, Any]]:
    allowed_tags = {"IT", "Energy", "Finance", "Healthcare", "Commodities", "Defense", "Chemicals", "Shipbuilding"}
    allowed_sentiments = {"Positive", "Neutral", "Warning", "Panic"}
    reviews = []
    for review in payload.get("reviews", []):
        if not isinstance(review, dict):
            continue
        item_id = normalize_space(review.get("item_id"))
        decision = normalize_space(review.get("decision")).lower()
        tag = normalize_space(review.get("corrected_tag"))
        sentiment = normalize_space(review.get("corrected_sentiment"))
        if not item_id or decision not in {"restore", "keep_excluded"}:
            continue
        if tag not in allowed_tags:
            tag = "Finance"
        if sentiment not in allowed_sentiments:
            sentiment = "Neutral"
        try:
            confidence = float(review.get("confidence") or 0)
        except Exception:
            confidence = 0.0
        reviews.append(
            {
                "item_id": item_id,
                "decision": decision,
                "corrected_tag": tag,
                "corrected_sentiment": sentiment,
                "confidence": max(0.0, min(1.0, confidence)),
                "rationale": normalize_space(review.get("rationale")),
            }
        )
    return reviews


def apply_excluded_reviews(conn: sqlite3.Connection, payload: dict[str, Any], items_by_id: dict[str, dict[str, Any]], model_id: str) -> dict[str, Any]:
    ensure_review_tables(conn)
    reviews = normalize_review_payload(payload)
    now = utc_now()
    restored = 0
    kept = 0
    feedback = 0
    for review in reviews:
        item = items_by_id.get(review["item_id"])
        if not item:
            continue
        conn.execute(
            """
            INSERT INTO llm_excluded_reviews (
                item_id, content_hash, model_id, decision, corrected_tag,
                corrected_sentiment, confidence, rationale, payload_json, created_at
            )
            VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
            ON CONFLICT(item_id) DO UPDATE SET
                content_hash = excluded.content_hash,
                model_id = excluded.model_id,
                decision = excluded.decision,
                corrected_tag = excluded.corrected_tag,
                corrected_sentiment = excluded.corrected_sentiment,
                confidence = excluded.confidence,
                rationale = excluded.rationale,
                payload_json = excluded.payload_json,
                created_at = excluded.created_at
            """,
            (
                review["item_id"],
                item["content_hash"],
                model_id,
                review["decision"],
                review["corrected_tag"],
                review["corrected_sentiment"],
                review["confidence"],
                review["rationale"],
                json.dumps(review, ensure_ascii=False),
                now,
            ),
        )
        if review["decision"] == "restore" and review["confidence"] >= 0.55:
            conn.execute(
                """
                UPDATE tag_results
                   SET excluded = 0,
                       exclude_reason = 'gpt55_excluded_retag_review',
                       primary_tag = ?,
                       sentiment_label = ?,
                       confidence = MAX(confidence, ?)
                 WHERE item_id = ?
                """,
                (review["corrected_tag"], review["corrected_sentiment"], review["confidence"], review["item_id"]),
            )
            conn.execute(
                """
                UPDATE tagging_queue
                   SET status = 'tagged',
                       finished_at = ?,
                       last_error = ''
                 WHERE item_id = ?
                """,
                (now, review["item_id"]),
            )
            conn.execute(
                """
                INSERT INTO label_feedback (
                    item_id, content_hash, original_tag, original_sentiment,
                    corrected_tag, corrected_sentiment, approved, reviewer,
                    notes, source, created_at
                ) VALUES (?, ?, ?, ?, ?, ?, 1, 'gpt55_excluded_review', ?, 'gpt55_excluded_review', ?)
                """,
                (
                    review["item_id"],
                    item["content_hash"],
                    item.get("primary_tag") or "",
                    item.get("sentiment_label") or "",
                    review["corrected_tag"],
                    review["corrected_sentiment"],
                    f"GPT-5.5 restored excluded item. confidence={review['confidence']:.2f}; rationale={review['rationale']}",
                    now,
                ),
            )
            restored += 1
            feedback += 1
        else:
            kept += 1
    entity_updates = upsert_won_entities(conn, payload)
    return {"reviewed": len(reviews), "restored": restored, "kept": kept, "feedback": feedback, "entity_updates": entity_updates}


def run_qwen_excluded_review(conn: sqlite3.Connection, limit: int = 8) -> dict[str, Any]:
    config = read_won_config()
    if not config.get("enabled"):
        return {"status": "disabled", "reviewed": 0, "restored": 0}
    ensure_review_tables(conn)
    rows = conn.execute(
        """
        SELECT c.id AS item_id, c.*, r.primary_tag, r.sentiment_label, r.relevance_score,
               r.confidence, r.exclude_reason
        FROM tag_results r
        JOIN crawled_items c ON c.id = r.item_id
        LEFT JOIN llm_excluded_reviews v ON v.item_id = r.item_id
        WHERE r.excluded = 1
          AND COALESCE(c.source_group, '') <> 'test'
          AND v.item_id IS NULL
        ORDER BY COALESCE(r.tagged_at, c.updated_at) DESC
        LIMIT ?
        """,
        (limit,),
    ).fetchall()
    items = [dict(row) for row in rows]
    if not items:
        return {"status": "no_pending_excluded", "reviewed": 0, "restored": 0}
    prompt = build_excluded_review_prompt(items)
    try:
        generated = call_won_reasoning_api(prompt, config)
        set_engine_state(conn, "last_excluded_review_call_mode", "api")
    except Exception as exc:
        set_engine_state(conn, "last_excluded_review_api_error", str(exc)[:1000])
        if not ENABLE_WON_LOCAL_FALLBACK:
            raise
        generated = call_won_reasoning_local(prompt, config)
        set_engine_state(conn, "last_excluded_review_call_mode", "local")
    payload = extract_json_object(generated)
    result = apply_excluded_reviews(
        conn,
        payload,
        {str(item["item_id"]): item for item in items},
        str(config.get("model_id") or WON_DEFAULT_MODEL),
    )
    set_engine_state(conn, "last_excluded_review_epoch", str(time.time()))
    set_engine_state(conn, "last_excluded_review_result", json.dumps({"status": "ok", **result}, ensure_ascii=False))
    return {"status": "ok", **result}


def upsert_won_entities(conn: sqlite3.Connection, payload: dict[str, Any]) -> int:
    count = 0
    now = utc_now()
    for item in payload.get("detected_entities", []):
        if not isinstance(item, dict):
            continue
        impact = str(item.get("estimated_market_impact") or "LOW").upper()
        if impact not in {"HIGH", "MEDIUM"}:
            continue
        name = normalize_space(item.get("entity_name"))
        category = normalize_space(item.get("category"))
        rationale = normalize_space(item.get("financial_relevance_rationale"))
        if not name or not category or not rationale:
            continue
        conn.execute(
            """
            INSERT INTO entity_dictionary (
                entity_name, category, synonyms_json, association_rules_json,
                rationale, estimated_market_impact, source, updated_at
            )
            VALUES (?, ?, ?, ?, ?, ?, 'gpt55_api', ?)
            ON CONFLICT(entity_name) DO UPDATE SET
                category = excluded.category,
                synonyms_json = excluded.synonyms_json,
                association_rules_json = excluded.association_rules_json,
                rationale = excluded.rationale,
                estimated_market_impact = excluded.estimated_market_impact,
                source = excluded.source,
                updated_at = excluded.updated_at
            """,
            (
                name,
                category,
                json.dumps(item.get("synonyms") or [], ensure_ascii=False),
                json.dumps(item.get("suggested_rules") or [], ensure_ascii=False),
                rationale,
                impact,
                now,
            ),
        )
        count += 1
    return count


def get_engine_state(conn: sqlite3.Connection, key: str) -> str:
    row = conn.execute("SELECT value FROM engine_state WHERE key = ?", (key,)).fetchone()
    return str(row["value"]) if row else ""


def set_engine_state(conn: sqlite3.Connection, key: str, value: str) -> None:
    conn.execute(
        """
        INSERT INTO engine_state (key, value, updated_at)
        VALUES (?, ?, ?)
        ON CONFLICT(key) DO UPDATE SET value = excluded.value, updated_at = excluded.updated_at
        """,
        (key, value, utc_now()),
    )


def run_won_entity_update(conn: sqlite3.Connection, limit: int = 8) -> dict[str, Any]:
    config = read_won_config()
    if not config.get("enabled"):
        return {"status": "disabled", "updated": 0}
    last_value = get_engine_state(conn, "last_won_entity_update_epoch")
    if last_value and time.time() - float(last_value) < WON_UPDATE_INTERVAL_SECONDS:
        return {"status": "skipped_interval", "updated": 0}

    rows = conn.execute(
        """
        SELECT c.*, r.exclude_reason, r.tagged_at
        FROM tag_results r
        JOIN crawled_items c ON c.id = r.item_id
        WHERE r.excluded = 1
          AND COALESCE(c.source_group, '') <> 'test'
        ORDER BY COALESCE(r.tagged_at, c.updated_at) DESC
        LIMIT ?
        """,
        (limit,),
    ).fetchall()
    samples = [dict(row) for row in rows]
    if not samples:
        set_engine_state(conn, "last_won_entity_update_epoch", str(time.time()))
        return {"status": "no_samples", "updated": 0}

    prompt = build_won_entity_prompt(samples)
    try:
        generated = call_won_reasoning_api(prompt, config)
        set_engine_state(conn, "last_won_call_mode", "api")
    except Exception as exc:
        set_engine_state(conn, "last_won_api_error", str(exc)[:1000])
        if not ENABLE_WON_LOCAL_FALLBACK:
            raise
        generated = call_won_reasoning_local(prompt, config)
        set_engine_state(conn, "last_won_call_mode", "local")
    payload = extract_json_object(generated)
    updated = upsert_won_entities(conn, payload)
    set_engine_state(conn, "last_won_entity_update_epoch", str(time.time()))
    set_engine_state(conn, "last_won_entity_update_status", f"updated={updated}")
    return {"status": "ok", "updated": updated}


def parse_ts(value: Any) -> float:
    raw = str(value or "").strip()
    if not raw:
        return 0.0
    try:
        return datetime.fromisoformat(raw.replace("Z", "+00:00")).timestamp()
    except Exception:
        return 0.0


def ensure_dedup_candidate_table(conn: sqlite3.Connection) -> None:
    conn.execute(
        """
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
        )
        """
    )
    conn.execute(
        """
        CREATE INDEX IF NOT EXISTS idx_dedup_candidates_model
        ON dedup_candidate_pairs(model_version, similarity)
        """
    )
    existing = {
        row["name"]
        for row in conn.execute("PRAGMA table_info(dedup_candidate_pairs)").fetchall()
    }
    for name, definition in {
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
    }.items():
        if name not in existing:
            conn.execute(f"ALTER TABLE dedup_candidate_pairs ADD COLUMN {name} {definition}")


def run_bge_dedup(
    conn: sqlite3.Connection,
    limit: int | None = None,
    threshold: float | None = None,
    window_hours: float | None = None,
    candidate_threshold: float | None = None,
) -> dict[str, Any]:
    """Build duplicate candidates with LSA/TF-IDF, score with BGE/features, and promote GPT-audited duplicates."""
    try:
        from sentence_transformers import SentenceTransformer  # type: ignore
    except Exception as exc:
        return {"status": "missing_sentence_transformers", "error": str(exc), "groups": 0}
    try:
        import numpy as np  # type: ignore
        from sklearn.decomposition import TruncatedSVD  # type: ignore
        from sklearn.feature_extraction.text import TfidfVectorizer  # type: ignore
        from sklearn.preprocessing import normalize  # type: ignore
    except Exception as exc:
        return {"status": "missing_sklearn_runtime", "error": str(exc), "groups": 0}

    ensure_dedup_candidate_table(conn)
    limit = int(limit or BGE_DEDUP_LIMIT)
    threshold = float(threshold or BGE_DEDUP_GROUP_THRESHOLD)
    candidate_threshold = float(candidate_threshold or BGE_DEDUP_CANDIDATE_THRESHOLD)
    window_hours = float(window_hours or BGE_DEDUP_WINDOW_HOURS)
    max_candidates_per_item = max(1, BGE_DEDUP_MAX_CANDIDATES_PER_ITEM)

    rows = conn.execute(
        """
        SELECT c.*
        FROM crawled_items c
        JOIN tag_results r ON r.item_id = c.id
        WHERE r.excluded = 0
          AND COALESCE(c.source_group, '') <> 'test'
        ORDER BY COALESCE(c.published_at, c.created_at) DESC
        LIMIT ?
        """,
        (limit,),
    ).fetchall()
    items = [dict(row) for row in rows]
    if len(items) < 2:
        return {"status": "not_enough_items", "groups": 0}

    item_by_id = {item["id"]: item for item in items}
    item_ids = [item["id"] for item in items]
    texts = [compact_item_text(item, 1800) for item in items]

    def normalized_url(value: Any) -> str:
        parsed = urlparse(str(value or "").strip())
        path = re.sub(r"/+$", "", parsed.path or "")
        return f"{parsed.netloc.lower()}{path.lower()}"

    def title_similarity(left: dict[str, Any], right: dict[str, Any]) -> float:
        left_title = normalize_space(left.get("title")).lower()
        right_title = normalize_space(right.get("title")).lower()
        if not left_title or not right_title:
            return 0.0
        return float(SequenceMatcher(None, left_title, right_title).ratio())

    def url_similarity(left: dict[str, Any], right: dict[str, Any]) -> float:
        left_url = normalized_url(left.get("url") or left.get("raw_url"))
        right_url = normalized_url(right.get("url") or right.get("raw_url"))
        if not left_url or not right_url:
            return 0.0
        if left_url == right_url:
            return 1.0
        return float(SequenceMatcher(None, left_url, right_url).ratio())

    def time_delta_hours(left: dict[str, Any], right: dict[str, Any]) -> float:
        left_ts = parse_ts(left.get("published_at") or left.get("created_at"))
        right_ts = parse_ts(right.get("published_at") or right.get("created_at"))
        if not left_ts or not right_ts:
            return 999999.0
        return abs(left_ts - right_ts) / 3600.0

    def composite_score(
        lsa_score: float,
        bge_score: float,
        title_score: float,
        url_score: float,
        source_match: bool,
        delta_hours: float,
    ) -> float:
        time_bonus = 0.08 if delta_hours <= 12 else 0.04 if delta_hours <= 72 else 0.0
        source_bonus = 0.04 if source_match else 0.0
        score = (
            lsa_score * 0.42
            + bge_score * 0.22
            + title_score * 0.22
            + url_score * 0.08
            + time_bonus
            + source_bonus
        )
        return round(max(0.0, min(1.0, score)), 6)

    vectorizer = TfidfVectorizer(
        analyzer="word",
        ngram_range=(1, 2),
        max_features=90000,
        min_df=1,
        sublinear_tf=True,
    )
    tfidf = vectorizer.fit_transform(texts)
    n_components = max(2, min(128, tfidf.shape[0] - 1, tfidf.shape[1] - 1))
    if n_components < 2:
        return {"status": "not_enough_lsa_features", "groups": 0}
    lsa = TruncatedSVD(n_components=n_components, random_state=42)
    lsa_dense = normalize(lsa.fit_transform(tfidf))
    lsa_sim = lsa_dense @ lsa_dense.T

    model = SentenceTransformer(BGE_MODEL_VERSION)
    vectors: dict[str, list[float]] = {}
    missing_items = []
    for item in items:
        emb_text = compact_item_text(item, 1800)
        digest = text_hash(emb_text)
        cached = conn.execute(
            """
            SELECT vector_json
            FROM item_embeddings
            WHERE item_id = ? AND model_version = ? AND text_hash = ?
            """,
            (item["id"], BGE_MODEL_VERSION, digest),
        ).fetchone()
        if cached:
            vectors[item["id"]] = json_loads(cached["vector_json"], [])
        else:
            missing_items.append((item, emb_text, digest))

    if missing_items:
        encoded = model.encode([text for _, text, _ in missing_items], normalize_embeddings=True, show_progress_bar=False)
        now = utc_now()
        for (item, _text, digest), vector in zip(missing_items, encoded):
            values = [float(v) for v in vector.tolist()]
            vectors[item["id"]] = values
            conn.execute(
                """
                INSERT OR REPLACE INTO item_embeddings (
                    item_id, content_hash, model_version, text_hash, vector_json, created_at
                )
                VALUES (?, ?, ?, ?, ?, ?)
                """,
                (item["id"], item["content_hash"], BGE_MODEL_VERSION, digest, json.dumps(values), now),
            )

    bge_matrix = np.array([vectors.get(item_id) or [] for item_id in item_ids], dtype=float)
    if bge_matrix.ndim != 2 or bge_matrix.shape[0] != len(item_ids):
        return {"status": "invalid_bge_vectors", "groups": 0}
    bge_sim = bge_matrix @ bge_matrix.T

    parent = {item["id"]: item["id"] for item in items}

    def find(x: str) -> str:
        while parent[x] != x:
            parent[x] = parent[parent[x]]
            x = parent[x]
        return x

    def union(a: str, b: str) -> None:
        ra, rb = find(a), find(b)
        if ra != rb:
            parent[rb] = ra

    candidate_map: dict[tuple[str, str], dict[str, Any]] = {}
    per_item_candidates: dict[str, list[tuple[str, float]]] = {}
    window_seconds = window_hours * 60 * 60
    for i, left in enumerate(items):
        left_id = left["id"]
        left_ts = parse_ts(left.get("published_at") or left.get("created_at"))
        for j, right in enumerate(items[i + 1 :], start=i + 1):
            right_id = right["id"]
            right_ts = parse_ts(right.get("published_at") or right.get("created_at"))
            if left_ts and right_ts and abs(left_ts - right_ts) > window_seconds:
                continue
            lsa_score = float(lsa_sim[i, j])
            bge_score = float(bge_sim[i, j])
            title_score = title_similarity(left, right)
            url_score = url_similarity(left, right)
            source_same = str(left.get("source") or "") == str(right.get("source") or "")
            delta_hours = abs(left_ts - right_ts) / 3600.0 if left_ts and right_ts else 999999.0
            score = composite_score(lsa_score, bge_score, title_score, url_score, source_same, delta_hours)
            is_candidate = (
                lsa_score >= DEDUP_LSA_CANDIDATE_THRESHOLD
                or title_score >= DEDUP_TITLE_CANDIDATE_THRESHOLD
                or (bge_score >= candidate_threshold and lsa_score >= 0.38)
                or score >= DEDUP_COMPOSITE_THRESHOLD
            )
            if not is_candidate:
                continue
            normalized_pair = tuple(sorted((left_id, right_id)))
            candidate_map[normalized_pair] = {
                "left_item_id": normalized_pair[0],
                "right_item_id": normalized_pair[1],
                "similarity": score,
                "bge_similarity": round(bge_score, 6),
                "lsa_similarity": round(lsa_score, 6),
                "title_similarity": round(title_score, 6),
                "url_similarity": round(url_score, 6),
                "source_match": 1 if source_same else 0,
                "time_delta_hours": round(delta_hours, 4),
                "composite_score": score,
                "audit_status": "pending_gpt",
                "audit_model": "",
                "gpt_is_duplicate": None,
                "gpt_confidence": None,
                "gpt_rationale": "",
                "audited_at": "",
                "within_group": 0,
            }
            per_item_candidates.setdefault(left_id, []).append((right_id, score))
            per_item_candidates.setdefault(right_id, []).append((left_id, score))

    allowed_candidate_pairs: set[tuple[str, str]] = set()
    for item_id, neighbors in per_item_candidates.items():
        neighbors.sort(key=lambda pair: pair[1], reverse=True)
        for neighbor_id, _score in neighbors[:max_candidates_per_item]:
            allowed_candidate_pairs.add(tuple(sorted((item_id, neighbor_id))))
    ranked_candidates = sorted(
        [
            row
            for pair_key, row in candidate_map.items()
            if pair_key in allowed_candidate_pairs
        ],
        key=lambda row: row["composite_score"],
        reverse=True,
    )[:DEDUP_MAX_CANDIDATE_PAIRS]

    audit_error = ""
    audited_count = 0
    duplicate_pairs: set[tuple[str, str]] = set()

    def build_duplicate_audit_prompt(batch: list[dict[str, Any]]) -> str:
        pairs = []
        for index, row in enumerate(batch, start=1):
            left = item_by_id[row["left_item_id"]]
            right = item_by_id[row["right_item_id"]]
            pairs.append({
                "pair_id": f"p{index}",
                "left_item_id": row["left_item_id"],
                "right_item_id": row["right_item_id"],
                "features": {
                    "lsa_similarity": row["lsa_similarity"],
                    "bge_similarity": row["bge_similarity"],
                    "title_similarity": row["title_similarity"],
                    "url_similarity": row["url_similarity"],
                    "source_match": bool(row["source_match"]),
                    "time_delta_hours": row["time_delta_hours"],
                    "composite_score": row["composite_score"],
                },
                "left": {
                    "source": left.get("source"),
                    "source_group": left.get("source_group"),
                    "published_at": left.get("published_at"),
                    "title": left.get("title"),
                    "url": left.get("url"),
                    "content": compact_item_text(left, 850),
                },
                "right": {
                    "source": right.get("source"),
                    "source_group": right.get("source_group"),
                    "published_at": right.get("published_at"),
                    "title": right.get("title"),
                    "url": right.get("url"),
                    "content": compact_item_text(right, 850),
                },
            })
        return (
            "You are a professional news deduplication auditor. "
            "Decide whether each pair is a duplicate or near-duplicate: same event/story, republication, translation, live update, or same article with minor edits. "
            "Return false for same broad topic but different event, different angle, or different factual development. "
            "Use the text, title, source, time, URL, and similarity features. "
            "Return strict JSON only with schema: "
            '{"duplicate_evaluations":[{"pair_id":"p1","is_duplicate":true,"confidence":0.0,"rationale":"short reason"}]}. '
            "Pairs:\n"
            + json.dumps(pairs, ensure_ascii=False, indent=2)
        )

    if ENABLE_GPT_DEDUP_AUDIT and ranked_candidates:
        config = read_won_config()
        if not config.get("enabled"):
            audit_error = "GPT dedup audit disabled: OPENAI_API_KEY or llm_config.json openai_api_key is required."
            for row in ranked_candidates:
                row["audit_status"] = "blocked_missing_gpt"
                row["gpt_rationale"] = audit_error
        else:
            for start in range(0, len(ranked_candidates), max(1, DEDUP_GPT_BATCH_SIZE)):
                batch = ranked_candidates[start:start + max(1, DEDUP_GPT_BATCH_SIZE)]
                try:
                    payload = extract_json_object(call_won_reasoning_api(build_duplicate_audit_prompt(batch), config))
                    evaluations = payload.get("duplicate_evaluations") or payload.get("evaluations") or []
                    by_pair_id = {
                        str(row.get("pair_id") or ""): row
                        for row in evaluations
                        if isinstance(row, dict)
                    }
                    for index, row in enumerate(batch, start=1):
                        pair_id = f"p{index}"
                        result = by_pair_id.get(pair_id)
                        if not result:
                            row["audit_status"] = "audit_failed"
                            row["gpt_is_duplicate"] = None
                            row["gpt_confidence"] = None
                            row["gpt_rationale"] = f"GPT response missing evaluation for {pair_id}"
                            continue
                        is_duplicate = bool(result.get("is_duplicate"))
                        try:
                            confidence = float(result.get("confidence") or 0.0)
                        except Exception:
                            confidence = 0.0
                        row["audit_status"] = "audited"
                        row["audit_model"] = str(config.get("model_id") or WON_DEFAULT_MODEL)
                        row["gpt_is_duplicate"] = 1 if is_duplicate else 0
                        row["gpt_confidence"] = max(0.0, min(1.0, confidence))
                        row["gpt_rationale"] = normalize_space(result.get("rationale"))
                        row["audited_at"] = utc_now()
                        audited_count += 1
                        if is_duplicate and confidence >= DEDUP_GPT_CONFIDENCE_THRESHOLD:
                            duplicate_pairs.add(tuple(sorted((row["left_item_id"], row["right_item_id"]))))
                except Exception as exc:
                    audit_error = str(exc)[:1000]
                    for row in batch:
                        row["audit_status"] = "audit_failed"
                        row["gpt_rationale"] = audit_error
                    break
    else:
        for row in ranked_candidates:
            row["audit_status"] = "pending_manual"

    conn.execute("DELETE FROM dedup_group_members")
    conn.execute("DELETE FROM dedup_groups WHERE model_version IN (?, ?)", (BGE_MODEL_VERSION, DEDUP_PIPELINE_MODEL_VERSION))
    conn.execute("DELETE FROM dedup_candidate_pairs WHERE model_version IN (?, ?)", (BGE_MODEL_VERSION, DEDUP_PIPELINE_MODEL_VERSION))
    now = utc_now()
    group_count = 0
    stored_candidate_count = 0

    def similarity_between(left_id: str, right_id: str) -> float:
        if left_id == right_id:
            return 1.0
        row = candidate_map.get(tuple(sorted((left_id, right_id))))
        return float(row.get("composite_score") or row.get("lsa_similarity") or 0.0) if row else 0.0

    for left_id, right_id in duplicate_pairs:
        union(left_id, right_id)

    clusters: dict[str, list[dict[str, Any]]] = {}
    for item in items:
        clusters.setdefault(find(item["id"]), []).append(item)

    grouped_pairs: set[tuple[str, str]] = set()
    for members in clusters.values():
        if len(members) < 2:
            continue
        member_ids = [member["id"] for member in members]
        members.sort(
            key=lambda row: (
                sum(similarity_between(row["id"], other_id) for other_id in member_ids if other_id != row["id"]),
                row.get("published_at") or row.get("created_at") or "",
            ),
            reverse=True,
        )
        representative = members[0]
        for left_id, right_id in itertools.combinations(sorted(member_ids), 2):
            if tuple(sorted((left_id, right_id))) in duplicate_pairs:
                grouped_pairs.add(tuple(sorted((left_id, right_id))))
        group_id = hashlib.sha1((DEDUP_PIPELINE_MODEL_VERSION + "|" + "|".join(sorted(member_ids))).encode("utf-8")).hexdigest()[:16]
        conn.execute(
            """
            INSERT INTO dedup_groups (
                group_id, representative_item_id, model_version, item_count, created_at, updated_at
            )
            VALUES (?, ?, ?, ?, ?, ?)
            """,
            (group_id, representative["id"], DEDUP_PIPELINE_MODEL_VERSION, len(members), now, now),
        )
        for member in members:
            if member["id"] == representative["id"]:
                sim = 1.0
            else:
                sim = similarity_between(representative["id"], member["id"])
                if sim <= 0:
                    sim = max(
                        [similarity_between(member["id"], other["id"]) for other in members if other["id"] != member["id"]],
                        default=threshold,
                    )
            conn.execute(
                """
                INSERT OR REPLACE INTO dedup_group_members (
                    item_id, group_id, representative_item_id, similarity, is_representative, created_at
                )
                VALUES (?, ?, ?, ?, ?, ?)
                """,
                (member["id"], group_id, representative["id"], float(sim), 1 if member["id"] == representative["id"] else 0, now),
            )
        group_count += 1

    for row in ranked_candidates:
        normalized_pair = tuple(sorted((row["left_item_id"], row["right_item_id"])))
        row["within_group"] = 1 if normalized_pair in grouped_pairs else 0
        conn.execute(
            """
            INSERT OR REPLACE INTO dedup_candidate_pairs (
                left_item_id, right_item_id, model_version, similarity,
                bge_similarity, lsa_similarity, title_similarity, url_similarity,
                source_match, time_delta_hours, composite_score,
                audit_status, audit_model, gpt_is_duplicate, gpt_confidence,
                gpt_rationale, audited_at, within_group, created_at
            )
            VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
            """,
            (
                normalized_pair[0],
                normalized_pair[1],
                DEDUP_PIPELINE_MODEL_VERSION,
                float(row["composite_score"]),
                float(row["bge_similarity"]),
                float(row["lsa_similarity"]),
                float(row["title_similarity"]),
                float(row["url_similarity"]),
                int(row["source_match"]),
                float(row["time_delta_hours"]),
                float(row["composite_score"]),
                str(row["audit_status"]),
                str(row.get("audit_model") or ""),
                row.get("gpt_is_duplicate"),
                row.get("gpt_confidence"),
                str(row.get("gpt_rationale") or ""),
                str(row.get("audited_at") or ""),
                int(row["within_group"]),
                now,
            ),
        )
        stored_candidate_count += 1

    conn.commit()
    return {
        "status": "ok",
        "groups": group_count,
        "candidate_pairs": stored_candidate_count,
        "duplicate_pairs": len(duplicate_pairs),
        "audited_pairs": audited_count,
        "audit_error": audit_error,
        "model": DEDUP_PIPELINE_MODEL_VERSION,
        "embedding_model": BGE_MODEL_VERSION,
        "limit": limit,
        "lsa_candidate_threshold": DEDUP_LSA_CANDIDATE_THRESHOLD,
        "bge_candidate_threshold": candidate_threshold,
        "gpt_confidence_threshold": DEDUP_GPT_CONFIDENCE_THRESHOLD,
        "window_hours": window_hours,
        "max_candidates_per_item": max_candidates_per_item,
    }


def dedup_observation_weight(cluster_size: int) -> float:
    if cluster_size <= 1:
        return 1.0
    return 1.0 + min(1.25, math.log(cluster_size) / 2.0)
