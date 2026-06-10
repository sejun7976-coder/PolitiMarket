"""Build dashboard analysis artifacts from crawler.db tag_results."""

from __future__ import annotations

import json
import os
import sqlite3
import sys
from collections import defaultdict
from datetime import datetime, timezone
from pathlib import Path
from typing import Any


BASE_DIR = Path(__file__).resolve().parents[1]
CRAWLING_DIR = BASE_DIR / "Crawling"
WEB_DIR = BASE_DIR / "Web"
DB_PATH = CRAWLING_DIR / "crawler.db"
TEST_MODE_PATH = CRAWLING_DIR / "test_mode.json"
OUTPUT_DIR = Path(__file__).resolve().parent / "output"
OUTPUT_JSON = OUTPUT_DIR / "category_analysis.json"
ANALYSIS_JS = WEB_DIR / "analysis.js"
SUMMARY_CACHE_PATH = OUTPUT_DIR / "summary_cache.json"
ENABLE_LLM_CATEGORY_SUMMARY = os.environ.get("FILTER_LLM_CATEGORY_SUMMARY", "1") == "1"

if str(Path(__file__).resolve().parent) not in sys.path:
    sys.path.insert(0, str(Path(__file__).resolve().parent))

try:
    import mining_engine
except Exception:
    mining_engine = None

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

SCORE_SCALE = 5.0
PANIC_THRESHOLD = -3.2
WARNING_THRESHOLD = -1.2
POSITIVE_THRESHOLD = 1.2
OVERHEATED_THRESHOLD = 4.0

LABEL_KO = {
    "Panic": "공포",
    "Warning": "주의",
    "Neutral": "중립",
    "Positive": "긍정",
    "Overheated": "과열",
}


def utc_now() -> str:
    return datetime.now(timezone.utc).isoformat(timespec="seconds")


def connect() -> sqlite3.Connection:
    conn = sqlite3.connect(DB_PATH)
    conn.row_factory = sqlite3.Row
    return conn


def row_to_dict(row: sqlite3.Row) -> dict[str, Any]:
    return {key: row[key] for key in row.keys()}


def load_test_mode() -> dict[str, Any]:
    if not TEST_MODE_PATH.exists():
        return {"active": False}
    try:
        return json.loads(TEST_MODE_PATH.read_text(encoding="utf-8"))
    except Exception:
        return {"active": False}


def normalize_item(row: dict[str, Any]) -> dict[str, Any]:
    tags = json.loads(row["tags_json"] or "[]")
    content = row.get("content") or ""
    title = row.get("title") or ""
    text = title if title and title not in content[:120] else content
    return {
        "id": row["item_id"],
        "content_hash": row["content_hash"],
        "source": row.get("source"),
        "source_group": row.get("source_group"),
        "country": row.get("country"),
        "language": row.get("language"),
        "url": row.get("url"),
        "raw_url": row.get("raw_url"),
        "author": row.get("author") or row.get("source"),
        "title": title,
        "content": content,
        "text": text,
        "published_at": row.get("published_at"),
        "crawled_at": row.get("created_at"),
        "tagged_at": row.get("tagged_at"),
        "primary_tag": row.get("primary_tag"),
        "tags": tags,
        "sentiment_label": row.get("sentiment_label"),
        "sentiment_score": float(row.get("sentiment_score") or 0),
        "confidence": float(row.get("confidence") or 0),
        "impact_type": row.get("impact_type"),
        "is_excluded": bool(row.get("is_excluded")),
        "reason": row.get("reason"),
        "model_version": row.get("model_version"),
        "dedup_group_id": row.get("dedup_group_id"),
        "dedup_similarity": float(row.get("dedup_similarity") or 1),
        "dedup_is_representative": bool(row.get("dedup_is_representative") or False),
        "dedup_cluster_size": int(row.get("dedup_cluster_size") or 1),
        "media_weight": (
            mining_engine.media_weight_for(row.get("source"), row.get("source_group"))
            if mining_engine is not None else 0.45
        ),
    }


def fetch_items(conn: sqlite3.Connection) -> list[dict[str, Any]]:
    test_mode = load_test_mode()
    test_only = bool(test_mode.get("active"))
    rows = conn.execute(
        """
        SELECT c.id AS item_id, c.*, r.model_version, r.primary_tag, r.tags_json, r.sentiment_label,
               r.sentiment_score, r.confidence, r.impact_type, r.excluded AS is_excluded,
               r.exclude_reason AS reason, r.tagged_at,
               dgm.group_id AS dedup_group_id,
               dgm.similarity AS dedup_similarity,
               dgm.is_representative AS dedup_is_representative,
               dg.item_count AS dedup_cluster_size
        FROM tag_results r
        JOIN crawled_items c ON c.id = r.item_id
        LEFT JOIN dedup_group_members dgm ON dgm.item_id = c.id
        LEFT JOIN dedup_groups dg ON dg.group_id = dgm.group_id
        WHERE ((? = 1 AND c.source_group = 'test')
            OR (? = 0 AND COALESCE(c.source_group, '') <> 'test'))
          AND NOT (
              COALESCE(c.source_group, '') NOT IN ('truth', 'russia', 'x', 'market', 'test')
              AND (
                  c.crawl_status = 'no_body_title_only'
                  OR LENGTH(TRIM(COALESCE(c.content, ''))) <= LENGTH(TRIM(COALESCE(c.title, ''))) + 20
                  OR (
                      c.source LIKE 'Gov_KoreaNet%'
                      AND c.url NOT LIKE '%articleId=%'
                      AND LENGTH(TRIM(COALESCE(c.content, ''))) < 300
                  )
              )
          )
        ORDER BY COALESCE(c.published_at, c.created_at) DESC
        """,
        (1 if test_only else 0, 1 if test_only else 0),
    ).fetchall()
    return [normalize_item(row_to_dict(row)) for row in rows]


def sentiment_bucket(item: dict[str, Any]) -> str:
    label = str(item.get("sentiment_label") or "").lower()
    score = float(item.get("sentiment_score") or 0)
    if label in {"panic", "warning"} or score < -0.15:
        return "negative"
    if label in {"positive", "optimistic"} or score > 0.15:
        return "positive"
    return "neutral"


def clamp_score(value: float, minimum: float = -5.0, maximum: float = 5.0) -> float:
    return max(minimum, min(maximum, value))


def theme_score(raw_sentiment: float) -> float:
    return round(clamp_score(raw_sentiment * SCORE_SCALE), 2)


def weighted_observations(items: list[dict[str, Any]]) -> list[tuple[float, float]]:
    grouped: dict[str, list[dict[str, Any]]] = defaultdict(list)
    for item in items:
        group_id = str(item.get("dedup_group_id") or item.get("id") or "")
        grouped[group_id].append(item)

    observations: list[tuple[float, float]] = []
    for members in grouped.values():
        weighted_sum = 0.0
        weight_total = 0.0
        cluster_size = max([int(item.get("dedup_cluster_size") or 1) for item in members], default=len(members))
        for item in members:
            confidence = max(0.35, min(1.0, float(item.get("confidence") or 0.35)))
            media_weight = max(0.05, float(item.get("media_weight") or 0.45))
            item_weight = confidence * media_weight
            weighted_sum += float(item.get("sentiment_score") or 0) * item_weight
            weight_total += item_weight
        if not weight_total:
            continue
        avg_sentiment = weighted_sum / weight_total
        if mining_engine is not None:
            dedup_weight = mining_engine.dedup_observation_weight(cluster_size)
        else:
            dedup_weight = 1.0
        observations.append((avg_sentiment, (weight_total / max(len(members), 1)) * dedup_weight))
    return observations


def confidence_weighted_sentiment(items: list[dict[str, Any]]) -> float:
    observations = weighted_observations(items)
    if not observations:
        return 0.0
    weighted_sum = sum(sentiment * weight for sentiment, weight in observations)
    weight_total = sum(weight for _sentiment, weight in observations)
    return weighted_sum / weight_total if weight_total else 0.0


def category_theme_score(items: list[dict[str, Any]]) -> float:
    if not items:
        return 0.0
    effective_count = sum(weight for _sentiment, weight in weighted_observations(items))
    sample_factor = min(1.0, 0.65 + effective_count * 0.07)
    return theme_score(confidence_weighted_sentiment(items) * sample_factor)


def label_from_score(score: float) -> str:
    if score <= PANIC_THRESHOLD:
        return "Panic"
    if score < WARNING_THRESHOLD:
        return "Warning"
    if score < POSITIVE_THRESHOLD:
        return "Neutral"
    if score < OVERHEATED_THRESHOLD:
        return "Positive"
    return "Overheated"


def average(values: list[float]) -> float:
    return sum(values) / len(values) if values else 0.0


def compact_text(value: Any, limit: int = 58) -> str:
    text = " ".join(str(value or "").split())
    if len(text) <= limit:
        return text
    return text[: max(0, limit - 3)].rstrip() + "..."


def friendly_source(source: Any) -> str:
    value = str(source or "수집 데이터")
    if value == "Gov_Kremlin":
        return "러시아 정부"
    if value == "News_PeopleCN_KO":
        return "중국 인민망"
    if value.startswith("TruthSocial_"):
        name = value.replace("TruthSocial_", "")
        if name == "DonaldTrump":
            name = "Donald Trump"
        return f"Truth Social({name})"
    if value.startswith("X_"):
        name = value.replace("X_", "")
        x_names = {
            "ElonMusk": "Elon Musk",
            "ROK_President_Lee": "이재명 대통령",
            "ROK_PresidentialOffice": "대통령실",
            "ROK_OPM": "국무총리실",
            "ROK_MND": "국방부",
            "ROK_MOFA": "외교부",
            "LeeJaemyung": "이재명",
        }
        return f"X({x_names.get(name, name)})"
    if value.startswith("Gov_"):
        name = value.replace("Gov_", "")
        gov_names = {
            "President": "대통령실",
            "Mofa": "외교부",
            "Mnd": "국방부",
            "Mohw": "보건복지부",
            "PolicyBriefing": "정책브리핑",
        }
        return gov_names.get(name, "정부 발표")
    if value.startswith("News_"):
        name = value.replace("News_", "")
        if name.startswith("Newsis"):
            return "뉴시스"
        return "뉴스"
    return value


def item_hint(item: dict[str, Any]) -> str:
    title = compact_text(item.get("title") or item.get("content") or item.get("source"), 42)
    source = friendly_source(item.get("source"))
    return f"{source}의 '{title}'"


def category_summary(category: str, items: list[dict[str, Any]]) -> str:
    if not items:
        return f"{category}는 아직 연결된 새 소식이 없어 중립으로 보고 있습니다. 수집과 태깅이 쌓이면 다시 요약됩니다."

    positive_items = sorted(
        [item for item in items if sentiment_bucket(item) == "positive"],
        key=lambda item: float(item.get("sentiment_score") or 0),
        reverse=True,
    )
    negative_items = sorted(
        [item for item in items if sentiment_bucket(item) == "negative"],
        key=lambda item: float(item.get("sentiment_score") or 0),
    )
    neutral_count = len([item for item in items if sentiment_bucket(item) == "neutral"])
    avg_score = category_theme_score(items)
    label = label_from_score(avg_score)
    label_ko = LABEL_KO.get(label, label)

    sentences = [
        f"{category}는 현재 {label_ko} 구간입니다. 최근 관련 소식 {len(items)}개 중 호재 {len(positive_items)}개, 악재 {len(negative_items)}개, 중립 {neutral_count}개가 반영됐습니다.",
    ]

    if positive_items:
        sentences.append(
            f"좋게 본 이유는 {item_hint(positive_items[0])}처럼 수요, 투자, 정책 기대를 키우는 신호가 있었기 때문입니다."
        )
    else:
        sentences.append("뚜렷한 호재는 아직 적어서 상승 판단은 보수적으로 두었습니다.")

    if negative_items:
        sentences.append(
            f"주의할 점은 {item_hint(negative_items[0])}처럼 규제, 갈등, 비용 부담으로 이어질 수 있는 신호입니다."
        )
    else:
        sentences.append("눈에 띄는 악재는 적어 당장 큰 위험 신호는 약합니다.")

    if label == "Overheated":
        sentences.append("다만 과열은 좋은 소식이 많이 몰렸다는 뜻이라, 단기 변동성도 함께 조심해서 봐야 합니다.")
    elif label == "Positive":
        sentences.append("초보자 관점에서는 분위기가 우호적이지만, 새 악재가 나오면 점수가 빠르게 내려갈 수 있습니다.")
    elif label == "Neutral":
        sentences.append("초보자 관점에서는 방향이 뚜렷하지 않으므로 추가 뉴스가 나올 때까지 관망에 가깝습니다.")
    else:
        sentences.append("초보자 관점에서는 방어적으로 보고, 관련 악재가 줄어드는지 먼저 확인하는 편이 좋습니다.")

    return " ".join(sentences)


def item_signature(items: list[dict[str, Any]]) -> str:
    hashes = [str(item.get("content_hash") or item.get("id") or "") for item in items]
    return "|".join(sorted(value for value in hashes if value))


def load_summary_cache() -> dict[str, Any]:
    if not SUMMARY_CACHE_PATH.exists():
        return {"version": 1, "categories": {}}
    try:
        data = json.loads(SUMMARY_CACHE_PATH.read_text(encoding="utf-8"))
        if isinstance(data, dict):
            data.setdefault("version", 1)
            data.setdefault("categories", {})
            return data
    except Exception:
        pass
    return {"version": 1, "categories": {}}


def save_summary_cache(cache: dict[str, Any]) -> None:
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    SUMMARY_CACHE_PATH.write_text(json.dumps(cache, ensure_ascii=False, indent=2), encoding="utf-8")


def summary_sort_key(item: dict[str, Any]) -> tuple[float, str]:
    media_weight = float(item.get("media_weight") or 0.45)
    confidence = float(item.get("confidence") or 0.35)
    sentiment_strength = abs(float(item.get("sentiment_score") or 0))
    cluster = max(1, int(item.get("dedup_cluster_size") or 1))
    priority = media_weight * 1.4 + confidence + sentiment_strength + min(cluster, 6) * 0.08
    return (priority, str(item.get("published_at") or item.get("crawled_at") or ""))


def representative_items_for_summary(items: list[dict[str, Any]], limit: int = 10) -> list[dict[str, Any]]:
    grouped: dict[str, list[dict[str, Any]]] = defaultdict(list)
    for item in items:
        group_id = str(item.get("dedup_group_id") or item.get("id") or "")
        grouped[group_id].append(item)

    representatives = []
    for members in grouped.values():
        members.sort(key=summary_sort_key, reverse=True)
        representatives.append(members[0])
    representatives.sort(key=summary_sort_key, reverse=True)
    return representatives[:limit]


def build_category_summary_prompt(category: str, payload: dict[str, Any], items: list[dict[str, Any]]) -> str:
    selected = representative_items_for_summary(items)
    article_payload = []
    for item in selected:
        article_payload.append(
            {
                "source": item.get("source"),
                "published_at": item.get("published_at"),
                "sentiment_label": item.get("sentiment_label"),
                "sentiment_score": item.get("sentiment_score"),
                "confidence": item.get("confidence"),
                "media_weight": item.get("media_weight"),
                "dedup_cluster_size": item.get("dedup_cluster_size"),
                "title": compact_text(item.get("title") or item.get("content"), 180),
                "content": compact_text(item.get("content"), 650),
            }
        )

    context = {
        "category": category,
        "score": payload.get("score"),
        "label": payload.get("label"),
        "item_count": payload.get("item_count"),
        "weighted_item_count": payload.get("weighted_item_count"),
        "positive_count": payload.get("positive_count"),
        "neutral_count": payload.get("neutral_count"),
        "negative_count": payload.get("negative_count"),
        "dedup_group_count": payload.get("dedup_group_count"),
        "articles": article_payload,
    }
    return (
        "너는 한국 개인투자자에게 정치/지정학 뉴스가 금융시장에 주는 영향을 설명하는 애널리스트다. "
        "아래 카테고리 데이터만 근거로 삼고, 모르는 내용은 추정하지 마라. "
        "중복 보도는 dedup_cluster_size를 참고해 한 이슈로 묶어라. "
        "반드시 JSON만 출력해라. 스키마는 "
        '{"summary":"3~4문장 한국어 요약","positive_factors":["호재 요인"],'
        '"negative_factors":["악재 요인"],"watch_points":["앞으로 볼 변수"],'
        '"confidence_note":"데이터 신뢰도/한계 1문장"} 이다.\n'
        + json.dumps(context, ensure_ascii=False, indent=2)
    )


def llm_category_summary(category: str, payload: dict[str, Any], items: list[dict[str, Any]]) -> dict[str, Any] | None:
    if not ENABLE_LLM_CATEGORY_SUMMARY or mining_engine is None or not items:
        return None
    config = mining_engine.read_won_config()
    if not config.get("enabled"):
        return None
    prompt = build_category_summary_prompt(category, payload, items)
    raw = mining_engine.call_won_reasoning_api(prompt, config)
    data = mining_engine.extract_json_object(raw)
    summary = str(data.get("summary") or "").strip()
    if not summary:
        return None
    return {
        "summary": summary,
        "positive_factors": [str(v) for v in data.get("positive_factors") or []][:4],
        "negative_factors": [str(v) for v in data.get("negative_factors") or []][:4],
        "watch_points": [str(v) for v in data.get("watch_points") or []][:4],
        "confidence_note": str(data.get("confidence_note") or "").strip(),
        "model": config.get("model_id"),
        "generated_at": utc_now(),
    }


def apply_llm_category_summaries(categories: dict[str, dict[str, Any]]) -> None:
    if not ENABLE_LLM_CATEGORY_SUMMARY or mining_engine is None:
        return
    cache = load_summary_cache()
    cache_categories = cache.setdefault("categories", {})
    changed = False
    for category, payload in categories.items():
        items = payload.get("all_items") or []
        if not items:
            payload["summary_source"] = "template_empty"
            continue
        config = mining_engine.read_won_config()
        model = config.get("model_id")
        signature = payload.get("summary_signature")
        cache_key = f"{category}:{model}"
        cached = cache_categories.get(cache_key)
        if cached and cached.get("signature") == signature:
            llm_payload = cached.get("payload") or {}
            if llm_payload.get("summary"):
                payload["summary"] = llm_payload["summary"]
                payload["llm_summary"] = llm_payload
                payload["summary_source"] = "llm_cache"
                payload["summary_model"] = model
                continue
        try:
            llm_payload = llm_category_summary(category, payload, items)
        except Exception as exc:
            payload["summary_source"] = "template_llm_error"
            payload["summary_error"] = str(exc)[:500]
            continue
        if not llm_payload:
            payload["summary_source"] = "template"
            continue
        payload["summary"] = llm_payload["summary"]
        payload["llm_summary"] = llm_payload
        payload["summary_source"] = "llm"
        payload["summary_model"] = model
        cache_categories[cache_key] = {
            "signature": signature,
            "payload": llm_payload,
            "updated_at": utc_now(),
        }
        changed = True
    if changed:
        save_summary_cache(cache)


def build_category_analysis(items: list[dict[str, Any]]) -> dict[str, Any]:
    grouped: dict[str, list[dict[str, Any]]] = defaultdict(list)
    excluded = []
    for item in items:
        if item["is_excluded"]:
            excluded.append(item)
            continue
        grouped[item.get("primary_tag") or "Finance"].append(item)

    categories: dict[str, dict[str, Any]] = {}
    category_list = []
    for category in CATEGORY_ORDER:
        bucket_items = grouped.get(category, [])
        buckets = {"positive": 0, "neutral": 0, "negative": 0}
        for item in bucket_items:
            buckets[sentiment_bucket(item)] += 1

        sentiment_scores = [float(item.get("sentiment_score") or 0) for item in bucket_items]
        positive_scores = [
            float(item.get("sentiment_score") or 0)
            for item in bucket_items
            if sentiment_bucket(item) == "positive"
        ]
        negative_scores = [
            float(item.get("sentiment_score") or 0)
            for item in bucket_items
            if sentiment_bucket(item) == "negative"
        ]
        avg_score = average(sentiment_scores)
        avg_confidence = average([float(item.get("confidence") or 0) for item in bucket_items])
        weighted_items = round(sum(weight for _sentiment, weight in weighted_observations(bucket_items)), 4)
        dedup_group_count = len({item.get("dedup_group_id") for item in bucket_items if item.get("dedup_group_id")})
        score = category_theme_score(bucket_items)
        payload = {
            "name": category,
            "category": category,
            "item_count": len(bucket_items),
            "score": score,
            "label": label_from_score(score),
            "average_sentiment": round(avg_score, 4),
            "average_confidence": round(avg_confidence, 4),
            "weighted_item_count": weighted_items,
            "dedup_group_count": dedup_group_count,
            "positive_count": buckets["positive"],
            "neutral_count": buckets["neutral"],
            "negative_count": buckets["negative"],
            "positive_score": theme_score(average(positive_scores)),
            "negative_score": theme_score(average(negative_scores)),
            "summary": category_summary(category, bucket_items),
            "summary_signature": item_signature(bucket_items),
            "latest_tagged_at": max([str(item.get("tagged_at") or "") for item in bucket_items], default=""),
            "items": bucket_items[:6],
            "all_items": bucket_items,
        }
        categories[category] = payload
        category_list.append(payload)

    apply_llm_category_summaries(categories)

    tagged_total = len([item for item in items if not item["is_excluded"]])
    excluded_total = len(excluded)
    return {
        "generated_at": utc_now(),
        "source": "crawler.db/tag_results",
        "realtime_pipeline": {
            "flow": "crawl -> crawled_items -> tagging_queue -> tag_results -> analysis.js",
            "training_source": "label_feedback approved/corrected labels only",
            "analysis_js_role": "dashboard output artifact",
        },
        "category_order": CATEGORY_ORDER,
        "totals": {
            "items": len(items),
            "tagged": tagged_total,
            "excluded": excluded_total,
            "categories": len(category_list),
        },
        "categories": categories,
        "category_list": category_list,
        "excluded": excluded[:100],
        "model_selection": {
            "mode": "realtime_tagging_periodic_validated_training",
            "primary_model": "ProsusAI/finbert with GPT-assisted keyword/entity tuning artifacts when promoted",
            "ko_model": "snunlp/KR-FinBert-SC with GPT-assisted keyword/entity tuning artifacts when promoted",
            "fallback_model": "realtime-keyword-v3",
            "fine_tuning_note": "Current pipeline exports BGE/NER/FinBERT keyword-tuning artifacts; it does not change BGE or FinBERT weights unless a validated promotion harness is added.",
        },
    }


def build_market_sentiment(analysis: dict[str, Any]) -> dict[str, Any]:
    raw_categories = analysis["categories"]
    categories = list(raw_categories.values()) if isinstance(raw_categories, dict) else raw_categories
    item_count = sum(category["item_count"] for category in categories)
    if item_count:
        weighted = sum(category["score"] * category["item_count"] for category in categories) / item_count
    else:
        weighted = 0.0

    risk_categories = [
        category["name"]
        for category in categories
        if category["item_count"] and category["score"] < WARNING_THRESHOLD
    ]
    opportunity_categories = [
        category["name"]
        for category in categories
        if category["item_count"] and POSITIVE_THRESHOLD <= category["score"] < OVERHEATED_THRESHOLD
    ]
    overheated_categories = [
        category["name"]
        for category in categories
        if category["item_count"] and category["score"] >= OVERHEATED_THRESHOLD
    ]
    overall_score = round(clamp_score(weighted), 2)
    overall_label = label_from_score(overall_score)
    if item_count:
        opportunities = ", ".join(opportunity_categories[:3]) or "없음"
        overheated = ", ".join(overheated_categories[:3]) or "없음"
        risks = ", ".join(risk_categories[:3]) or "없음"
        summary = (
            f"전체 시장 심리는 {LABEL_KO.get(overall_label, overall_label)} 구간입니다. "
            f"총 {item_count}개 태깅 데이터를 가중 평균해 점수 {overall_score:+.1f}로 계산했습니다. "
            f"기회 섹터는 {opportunities}, 과열 섹터는 {overheated}, 리스크 섹터는 {risks}입니다."
        )
    else:
        summary = "아직 전체 시장 심리를 계산할 만큼 태깅된 실시간 데이터가 충분하지 않습니다."

    return {
        "generated_at": analysis["generated_at"],
        "overall_score": overall_score,
        "overall_label": overall_label,
        "summary": summary,
        "risk_categories": risk_categories,
        "opportunity_categories": opportunity_categories,
        "overheated_categories": overheated_categories,
        "tagged_items": item_count,
        "excluded_items": analysis["totals"]["excluded"],
        "thresholds": {
            "panic": PANIC_THRESHOLD,
            "warning": WARNING_THRESHOLD,
            "positive": POSITIVE_THRESHOLD,
            "overheated": OVERHEATED_THRESHOLD,
        },
    }


def write_outputs(analysis: dict[str, Any], market_sentiment: dict[str, Any]) -> None:
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    WEB_DIR.mkdir(parents=True, exist_ok=True)
    OUTPUT_JSON.write_text(json.dumps(analysis, ensure_ascii=False, indent=2), encoding="utf-8")
    js = (
        "// Generated from crawler.db real-time tag_results. Do not edit manually.\n"
        f"window.categoryAnalysisData = {json.dumps(analysis, ensure_ascii=False, indent=2)};\n\n"
        f"window.marketSentimentAnalysis = {json.dumps(market_sentiment, ensure_ascii=False, indent=2)};\n"
    )
    ANALYSIS_JS.write_text(js, encoding="utf-8")


def main() -> int:
    if not DB_PATH.exists():
        empty = build_category_analysis([])
        write_outputs(empty, build_market_sentiment(empty))
        return 0

    conn = connect()
    try:
        items = fetch_items(conn)
    finally:
        conn.close()

    analysis = build_category_analysis(items)
    write_outputs(analysis, build_market_sentiment(analysis))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
