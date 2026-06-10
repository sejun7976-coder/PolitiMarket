"""Real-time queue tagger for PolitiMarket crawler.db.

Category tagging uses the lightweight sector keyword matcher, while sentiment
is scored with FinBERT-family sequence classifiers.
"""

from __future__ import annotations

import argparse
import json
import os
import re
import sqlite3
import sys
import time
from datetime import datetime, timezone
from pathlib import Path
from typing import Any


BASE_DIR = Path(__file__).resolve().parents[1]
CRAWLING_DIR = BASE_DIR / "Crawling"
WEB_DIR = BASE_DIR / "Web"
DB_PATH = CRAWLING_DIR / "crawler.db"
OUTPUT_DIR = Path(__file__).resolve().parent / "output"
PROGRESS_PATH = OUTPUT_DIR / "tagging_progress.json"

MODEL_VERSION = os.environ.get("FILTER_MODEL_VERSION", "bge-finbert-ner-keyword-v1")
EXCLUDED_TAG = "Excluded"
BATCH_SIZE = int(os.environ.get("FILTER_BATCH_SIZE", "32"))
MIN_RELEVANCE = float(os.environ.get("FILTER_MIN_RELEVANCE", "0.32"))
AUTO_REVIEW_EXCLUDED = os.environ.get("FILTER_AUTO_REVIEW_EXCLUDED", "1") == "1"
EN_FINBERT_MODEL = os.environ.get("FILTER_EN_FINBERT_MODEL", "ProsusAI/finbert")
KO_FINBERT_MODEL = os.environ.get("FILTER_KO_FINBERT_MODEL", "snunlp/KR-FinBert-SC")
ALLOW_KEYWORD_SENTIMENT_FALLBACK = os.environ.get("FILTER_ALLOW_KEYWORD_SENTIMENT_FALLBACK", "0") == "1"
ENABLE_BGE_CATEGORY = os.environ.get("FILTER_ENABLE_BGE_CATEGORY", "1") == "1"
BGE_CATEGORY_MODEL = os.environ.get("FILTER_CATEGORY_BGE_MODEL", "BAAI/bge-m3")
REQUIRE_BGE_CATEGORY = os.environ.get("FILTER_REQUIRE_BGE_CATEGORY", "1") == "1"
REQUIRE_FINBERT = os.environ.get("FILTER_REQUIRE_FINBERT", "1") == "1"
REQUIRE_NER = os.environ.get("FILTER_REQUIRE_NER", "1") == "1"
FILTER_TORCH_DEVICE = os.environ.get("FILTER_TORCH_DEVICE", "cuda").strip().lower()
REQUIRE_GPU = os.environ.get("FILTER_REQUIRE_GPU", "1") == "1"
BGE_CATEGORY_MIN_SCORE = float(os.environ.get("FILTER_BGE_CATEGORY_MIN_SCORE", "0.34"))
BGE_CATEGORY_AMBIGUOUS_MARGIN = float(os.environ.get("FILTER_BGE_CATEGORY_AMBIGUOUS_MARGIN", "0.06"))
BGE_CATEGORY_QWEN_MARGIN = float(os.environ.get("FILTER_BGE_CATEGORY_QWEN_MARGIN", "0.08"))
ENABLE_WON_ENTITY_UPDATE = os.environ.get("FILTER_WON_ENTITY_UPDATE", "0") == "1"
ENABLE_QWEN_EXCLUDED_REVIEW = os.environ.get("FILTER_QWEN_EXCLUDED_REVIEW", "0") == "1"
ENABLE_BGE_DEDUP = os.environ.get("FILTER_BGE_DEDUP", "1") == "1"
BGE_DEDUP_INTERVAL_SECONDS = int(os.environ.get("FILTER_BGE_DEDUP_INTERVAL_SECONDS", "1800"))

if str(CRAWLING_DIR) not in sys.path:
    sys.path.insert(0, str(CRAWLING_DIR))

try:
    import db_utils
except Exception:
    db_utils = None

try:
    import mining_engine
except Exception:
    mining_engine = None


def append_system_log(message: str) -> None:
    log_path = CRAWLING_DIR / "crawler_loop.log"
    try:
        with log_path.open("a", encoding="utf-8", errors="replace") as handle:
            handle.write(message.rstrip() + "\n")
    except Exception:
        pass


CATEGORY_KEYWORDS: dict[str, list[str]] = {
    "IT": [
        "ai",
        "openai",
        "anthropic",
        "chatgpt",
        "codex",
        "llm",
        "machine learning",
        "artificial intelligence",
        "semiconductor",
        "chip",
        "technology",
        "cyber",
        "cloud",
        "data center",
        "반도체",
        "인공지능",
        "엔비디아",
        "기술",
        "사이버",
    ],
    "Energy": [
        "oil",
        "gas",
        "lng",
        "energy",
        "power",
        "electricity",
        "crude",
        "석유",
        "원유",
        "가스",
        "에너지",
        "전력",
        "태양광",
        "풍력",
    ],
    "Finance": [
        "market",
        "economy",
        "inflation",
        "rate",
        "bank",
        "treasury",
        "dollar",
        "finance",
        "stock",
        "경제",
        "금융",
        "금리",
        "환율",
        "증시",
        "투자",
        "물가",
        "은행",
    ],
    "Healthcare": [
        "health",
        "pharma",
        "vaccine",
        "hospital",
        "bio",
        "의료",
        "보건",
        "제약",
        "바이오",
        "백신",
        "병원",
    ],
    "Commodities": [
        "gold",
        "silver",
        "copper",
        "wheat",
        "steel",
        "commodity",
        "rare earth",
        "원자재",
        "금값",
        "금 가격",
        "금 시세",
        "은 가격",
        "은 시세",
        "구리",
        "철강",
        "희토류",
        "곡물",
    ],
    "Defense": [
        "war",
        "weapon",
        "missile",
        "nuclear",
        "security",
        "military",
        "defense",
        "drone",
        "전쟁",
        "무기",
        "미사일",
        "핵",
        "안보",
        "군사",
        "방산",
        "드론",
    ],
    "Chemicals": [
        "chemical",
        "battery",
        "petrochemical",
        "fertilizer",
        "lithium",
        "화학",
        "배터리",
        "석유화학",
        "비료",
        "리튬",
    ],
    "Shipbuilding": [
        "ship",
        "vessel",
        "shipping",
        "shipbuilding",
        "naval",
        "조선",
        "선박",
        "해운",
        "lng선",
        "함정",
    ],
}

NEGATIVE_KEYWORDS = [
    "risk",
    "war",
    "sanction",
    "crisis",
    "fall",
    "decline",
    "attack",
    "conflict",
    "inflation",
    "default",
    "volatility",
    "위험",
    "전쟁",
    "제재",
    "위기",
    "하락",
    "공격",
    "갈등",
    "물가",
    "디폴트",
    "변동성",
]
POSITIVE_KEYWORDS = [
    "growth",
    "deal",
    "agreement",
    "investment",
    "cooperation",
    "support",
    "profit",
    "record",
    "expand",
    "성장",
    "합의",
    "투자",
    "협력",
    "지원",
    "수익",
    "최대",
    "확대",
]

KOREAN_CATEGORY_KEYWORDS: dict[str, list[str]] = {
    "IT": ["AI", "인공지능", "반도체", "칩", "기술", "사이버", "클라우드", "데이터센터", "디지털", "소프트웨어"],
    "Energy": ["원유", "석유", "가스", "LNG", "에너지", "전력", "전기", "발전", "태양광", "풍력", "원전"],
    "Finance": ["경제", "금융", "금리", "환율", "은행", "증시", "주식", "투자", "물가", "국부펀드", "채권", "시장"],
    "Healthcare": ["의료", "보건", "제약", "바이오", "백신", "병원", "신약", "헬스케어"],
    "Commodities": ["금값", "금 가격", "은 가격", "은 시세", "실버", "구리", "철강", "원자재", "희토류", "곡물", "밀", "원자재"],
    "Defense": ["전쟁", "무기", "미사일", "핵", "안보", "군사", "국방", "방산", "드론", "동맹", "사령관"],
    "Chemicals": ["화학", "배터리", "석유화학", "비료", "리튬", "소재", "정유"],
    "Shipbuilding": ["조선", "선박", "해운", "LNG선", "함정", "수주", "조선업"],
}

for _category, _keywords in KOREAN_CATEGORY_KEYWORDS.items():
    CATEGORY_KEYWORDS.setdefault(_category, []).extend(_keywords)

CLEAN_KOREAN_CATEGORY_KEYWORDS: dict[str, list[str]] = {
    "IT": [
        "AI", "인공지능", "반도체", "칩", "GPU", "데이터센터", "클라우드", "소프트웨어",
        "사이버", "보안", "수출통제", "엔비디아", "삼성전자", "SK하이닉스", "로봇",
    ],
    "Energy": [
        "유가", "원유", "석유", "가스", "LNG", "전력", "전기", "원전", "원자력",
        "재생에너지", "태양광", "풍력", "정유", "에너지 안보",
    ],
    "Finance": [
        "증시", "주식", "주가", "상하이지수", "코스피", "코스닥", "나스닥", "S&P",
        "경제", "금리", "환율", "달러", "위안", "원화", "채권", "국채", "은행",
        "인플레이션", "물가", "관세", "무역", "수출", "수입", "GDP", "재정",
    ],
    "Healthcare": [
        "의료", "보건", "병원", "제약", "바이오", "백신", "신약", "임상", "의약품",
        "건강보험", "감염병", "의료기기",
    ],
    "Commodities": [
        "원자재", "금값", "금 가격", "은 가격", "구리", "철광석", "철강", "희토류",
        "알루미늄", "석탄", "리튬 원료", "곡물", "밀", "광물",
    ],
    "Defense": [
        "전쟁", "제재", "무기", "미사일", "핵", "안보", "군사", "방산", "동맹",
        "드론", "잠수함", "핵잠수함", "북한", "대만", "우크라이나", "정보기관",
    ],
    "Chemicals": [
        "화학", "석유화학", "배터리 소재", "양극재", "음극재", "비료", "리튬 가공",
        "플라스틱", "정유 마진", "화학 공장",
    ],
    "Shipbuilding": [
        "조선", "선박", "선박 수주", "LNG선", "컨테이너선", "탱커", "해운", "항만",
        "해양플랜트", "함정", "잠수함 건조", "조선소", "한화오션", "HD현대중공업", "삼성중공업",
    ],
}

for _category, _keywords in CLEAN_KOREAN_CATEGORY_KEYWORDS.items():
    CATEGORY_KEYWORDS.setdefault(_category, []).extend(_keywords)

CLEAN_KOREAN_CATEGORY_KEYWORDS = {
    "IT": [
        "AI", "인공지능", "생성형 AI", "LLM", "반도체", "칩", "GPU", "데이터센터",
        "클라우드", "소프트웨어", "사이버", "보안", "수출통제", "엔비디아",
        "삼성전자", "SK하이닉스", "로봇", "플랫폼",
    ],
    "Energy": [
        "원유", "유가", "석유", "천연가스", "가스", "LNG", "전력", "전기",
        "발전", "원전", "재생에너지", "태양광", "풍력", "정유", "에너지 안보",
        "OPEC", "송유관",
    ],
    "Finance": [
        "증시", "주식", "주가", "코스피", "코스닥", "S&P", "나스닥", "경제",
        "금융", "금리", "환율", "달러", "원화", "채권", "국채", "대출",
        "인플레이션", "물가", "관세", "무역", "수출", "수입", "GDP", "재정",
        "은행", "유동성", "신용", "세금", "예산",
    ],
    "Healthcare": [
        "의료", "보건", "병원", "제약", "바이오", "백신", "신약", "임상",
        "의약품", "건강보험", "감염병", "의료기기", "치료제",
    ],
    "Commodities": [
        "원자재", "금값", "금 가격", "은 가격", "구리", "철광석", "철강",
        "희토류", "알루미늄", "석탄", "리튬 원료", "곡물", "밀", "광물",
    ],
    "Defense": [
        "전쟁", "제재", "무기", "미사일", "핵무기", "안보", "군사", "방산",
        "드론", "동맹", "해군", "잠수함", "북한", "대만", "우크라이나",
        "정보기관", "국방",
    ],
    "Chemicals": [
        "화학", "석유화학", "배터리 소재", "양극재", "음극재", "비료",
        "리튬 가공", "플라스틱", "정유 마진", "화학 공장",
    ],
    "Shipbuilding": [
        "조선", "선박", "선박 수주", "LNG선", "컨테이너선", "탱커", "해운",
        "항만", "해양플랜트", "함정 건조", "조선소", "HD현대중공업", "삼성중공업",
    ],
}

for _category, _keywords in CLEAN_KOREAN_CATEGORY_KEYWORDS.items():
    CATEGORY_KEYWORDS.setdefault(_category, []).extend(_keywords)

SCORING_CATEGORY_KEYWORDS: dict[str, list[str]] = {
    "IT": [
        "ai", "openai", "anthropic", "chatgpt", "llm", "machine learning", "artificial intelligence",
        "semiconductor", "chip", "gpu", "technology", "cyber", "cloud", "data center", "software",
        "nvidia", "robot", *CLEAN_KOREAN_CATEGORY_KEYWORDS["IT"],
    ],
    "Energy": [
        "oil", "crude", "gas", "lng", "energy", "power", "electricity", "nuclear", "solar",
        "wind power", "renewable", "opec", "pipeline", "refinery", *CLEAN_KOREAN_CATEGORY_KEYWORDS["Energy"],
    ],
    "Finance": [
        "market", "economy", "inflation", "rate", "bank", "treasury", "dollar", "finance",
        "stock", "bond", "tariff", "trade", "gdp", "tax", "export", "import", "fiscal",
        "liquidity", "fx", *CLEAN_KOREAN_CATEGORY_KEYWORDS["Finance"],
    ],
    "Healthcare": [
        "health", "pharma", "vaccine", "hospital", "bio", "drug", "medical", "clinical",
        "insurance", "epidemic", *CLEAN_KOREAN_CATEGORY_KEYWORDS["Healthcare"],
    ],
    "Commodities": [
        "gold", "silver", "copper", "wheat", "steel", "commodity", "rare earth", "aluminum",
        "coal", "iron ore", "lithium", "grain", "mining", *CLEAN_KOREAN_CATEGORY_KEYWORDS["Commodities"],
    ],
    "Defense": [
        "war", "weapon", "missile", "nuclear weapon", "security", "military", "defense",
        "drone", "sanction", "conflict", "alliance", "submarine", "intelligence", "naval",
        *CLEAN_KOREAN_CATEGORY_KEYWORDS["Defense"],
    ],
    "Chemicals": [
        "chemical", "battery material", "petrochemical", "fertilizer", "lithium processing",
        "cathode", "anode", "plastic", *CLEAN_KOREAN_CATEGORY_KEYWORDS["Chemicals"],
    ],
    "Shipbuilding": [
        "ship", "vessel", "shipping", "shipbuilding", "naval vessel", "port", "shipyard",
        "lng carrier", "tanker", "container ship", "offshore plant", *CLEAN_KOREAN_CATEGORY_KEYWORDS["Shipbuilding"],
    ],
}

SCORING_CATEGORY_KEYWORDS["Defense"] = [
    keyword for keyword in SCORING_CATEGORY_KEYWORDS["Defense"]
    if keyword not in {"제재"}
]

NEGATIVE_KEYWORDS.extend([
    "위험", "전쟁", "제재", "위기", "하락", "공격", "갈등", "물가", "인플레이션",
    "부도", "변동성", "침체", "탄핵", "파산", "논란", "피해", "재해", "침수",
    "차단", "감축", "압박", "리스크", "불법",
])
POSITIVE_KEYWORDS.extend([
    "성장", "합의", "협력", "지원", "투자", "수익", "최대", "확대", "증가", "개선",
    "구축", "착수", "강화", "수출", "전략", "제휴", "호황", "발전", "기여", "확보",
])

CATEGORY_PROFILES: dict[str, str] = {
    "IT": (
        "Information technology and digital infrastructure. Includes artificial intelligence, "
        "large language models, cloud software, cybersecurity, semiconductors, memory chips, GPUs, "
        "data centers, export controls on advanced chips, robotics, platform regulation, and corporate "
        "AI spending. Korean examples: AI 투자, 반도체 수출통제, 데이터센터 전력수요, 클라우드 비용, "
        "사이버 보안, 삼성전자, SK하이닉스, 엔비디아 공급망."
    ),
    "Energy": (
        "Energy markets and power supply. Includes crude oil, refined products, natural gas, LNG, "
        "electricity prices, grids, nuclear power, renewable energy, energy sanctions, OPEC, pipelines, "
        "shipping of fuel, and energy security. Korean examples: 유가, LNG선 수요, 전력망, 원전, "
        "재생에너지, 에너지 안보, 가스 공급, 정유."
    ),
    "Finance": (
        "Macro economy and financial markets. Includes central banks, interest rates, inflation, FX, "
        "dollar, won, treasury bonds, fiscal policy, tariffs, trade balances, GDP, banking, credit risk, "
        "stock markets, bond markets, taxes, government budgets, and market liquidity. Korean examples: "
        "금리, 환율, 물가, 관세, 무역, 코스피, 채권, 달러, 은행, 경기 둔화, 재정정책."
    ),
    "Healthcare": (
        "Healthcare, medicine, biotech, hospitals, public health and pharmaceuticals. Includes vaccines, "
        "drug approvals, clinical trials, medical devices, health insurance, epidemics, biotechnology, "
        "and policy that affects pharma or hospital operators. Korean examples: 제약, 바이오, 백신, 병원, "
        "신약, 의료기기, 건강보험, 감염병."
    ),
    "Commodities": (
        "Raw materials and commodity supply chains. Includes gold, silver, copper, iron ore, steel, wheat, "
        "rare earths, aluminum, coal, lithium as a raw material, fertilizer feedstocks, food commodities, "
        "and mining restrictions. Korean examples: 금, 은, 구리, 철광석, 철강, 희토류, 리튬 원료, 곡물, 원자재 가격."
    ),
    "Defense": (
        "Defense, security and geopolitical conflict. Includes war, sanctions tied to security, missiles, "
        "drones, nuclear weapons, military alliances, naval power, intelligence, arms procurement, export "
        "controls for security, North Korea, Taiwan, Ukraine, China-US tension, and defense industry demand. "
        "Korean examples: 방산, 미사일, 핵잠수함, 한미 안보, 제재, 군사동맹, 드론, 무기 수출."
    ),
    "Chemicals": (
        "Chemicals and industrial materials. Includes petrochemicals, battery materials, fertilizers, "
        "chemical plants, lithium processing, cathode and anode materials, plastics, refining-linked "
        "chemical margins, and industrial safety issues. Korean examples: 석유화학, 배터리 소재, 양극재, "
        "음극재, 비료, 리튬 가공, 화학 공장."
    ),
    "Shipbuilding": (
        "Shipbuilding, shipping and naval construction. Includes commercial vessels, LNG carriers, tankers, "
        "container ships, offshore plants, ports, maritime logistics, naval vessels, submarine construction, "
        "and shipyard order books. Korean examples: 조선, LNG선, 선박 수주, 해운, 항만, 잠수함 건조, "
        "HD현대중공업, 한화오션, 삼성중공업."
    ),
}

CATEGORY_PROFILES = {
    "IT": (
        "Information technology, AI and digital infrastructure. Includes artificial intelligence, "
        "large language models, cloud software, cybersecurity, semiconductors, memory chips, GPUs, "
        "data centers, advanced chip export controls, robotics, platform regulation and enterprise "
        "AI spending. 한국어 예시: AI 투자, 반도체 수출통제, 데이터센터 전력 수요, 클라우드 비용, "
        "사이버 보안, 삼성전자, SK하이닉스, 엔비디아 공급망."
    ),
    "Energy": (
        "Energy markets and power supply. Includes crude oil, refined products, natural gas, LNG, "
        "electricity prices, power grids, nuclear power, renewable energy, energy sanctions, OPEC, "
        "pipelines, fuel shipping and energy security. 한국어 예시: 유가 급등, LNG 수요, 전력망 투자, "
        "원전 정책, 재생에너지, 에너지 안보, 가스 공급, 정유 마진."
    ),
    "Finance": (
        "Macro economy and financial markets. Includes central banks, interest rates, inflation, FX, "
        "dollar and won moves, treasury bonds, fiscal policy, tariffs, trade balances, GDP, banking, "
        "credit risk, stock markets, bond markets, taxes, government budgets and liquidity. 한국어 예시: "
        "금리 인하, 환율 급등, 물가 상승, 관세와 무역, 코스피, 채권, 달러 강세, 재정 정책."
    ),
    "Healthcare": (
        "Healthcare, medicine, biotechnology, hospitals, public health and pharmaceuticals. Includes "
        "vaccines, drug approvals, clinical trials, medical devices, health insurance, epidemics, "
        "biotech investment and policy affecting pharma or hospitals. 한국어 예시: 제약, 바이오, 백신, "
        "병원, 신약 승인, 의료기기, 건강보험, 감염병."
    ),
    "Commodities": (
        "Raw materials and commodity supply chains. Includes gold, silver, copper, iron ore, steel, "
        "rare earths, aluminum, coal, lithium as a raw material, grains, mining restrictions and "
        "fertilizer feedstocks. 한국어 예시: 금값, 구리 가격, 철광석, 철강, 희토류, 리튬 원료, 곡물, "
        "원자재 가격."
    ),
    "Defense": (
        "Defense, security and geopolitical conflict. Includes war, sanctions tied to security, "
        "missiles, drones, nuclear weapons, military alliances, naval power, intelligence, arms "
        "procurement, security export controls, North Korea, Taiwan, Ukraine, China-US tension and "
        "defense industry demand. 한국어 예시: 방산, 미사일, 잠수함, 안보, 제재, 군사 동맹, 드론, "
        "무기 수출."
    ),
    "Chemicals": (
        "Chemicals and industrial materials. Includes petrochemicals, battery materials, fertilizers, "
        "chemical plants, lithium processing, cathode and anode materials, plastics, refining-linked "
        "chemical margins and industrial safety. 한국어 예시: 석유화학, 배터리 소재, 양극재, 음극재, "
        "비료, 리튬 가공, 화학 공장."
    ),
    "Shipbuilding": (
        "Shipbuilding, shipping and naval construction. Includes commercial vessels, LNG carriers, "
        "tankers, container ships, offshore plants, ports, maritime logistics, naval vessels, "
        "submarine construction and shipyard order books. 한국어 예시: 조선, LNG선 수주, 선박 발주, "
        "해운, 항만, 함정 건조, HD현대중공업, 삼성중공업."
    ),
}

NEGATIVE_KEYWORDS.extend([
    "위험", "전쟁", "제재", "위기", "하락", "공격", "갈등", "물가", "인플레이션",
    "부도", "변동성", "침체", "위협", "파산", "손실", "피해", "차단", "감축",
    "규제", "불법", "긴장", "관세 인상",
])
POSITIVE_KEYWORDS.extend([
    "성장", "합의", "협력", "지원", "투자", "수익", "최대", "흑자", "증가",
    "개선", "구축", "강화", "수출", "계약", "제휴", "호황", "발전", "기여",
    "호재", "관세 인하",
])


def utc_now() -> str:
    return datetime.now(timezone.utc).isoformat(timespec="seconds")


def connect() -> sqlite3.Connection:
    if db_utils is not None:
        return db_utils.connect(str(DB_PATH))
    conn = sqlite3.connect(DB_PATH)
    conn.row_factory = sqlite3.Row
    conn.execute("PRAGMA journal_mode=WAL")
    conn.execute("PRAGMA busy_timeout=30000")
    return conn


def init_db(conn: sqlite3.Connection) -> None:
    if db_utils is not None:
        db_utils.init_db(str(DB_PATH))
        if mining_engine is not None:
            mining_engine.seed_builtin_entities(conn)
            conn.commit()
        return
    conn.executescript(
        """
        CREATE TABLE IF NOT EXISTS crawled_items (
            id TEXT PRIMARY KEY,
            content_hash TEXT NOT NULL UNIQUE,
            source TEXT NOT NULL,
            source_group TEXT,
            country TEXT,
            language TEXT,
            raw_url TEXT,
            title TEXT,
            content TEXT NOT NULL,
            published_at TEXT,
            created_at TEXT NOT NULL,
            updated_at TEXT NOT NULL,
            metadata_json TEXT DEFAULT '{}'
        );

        CREATE TABLE IF NOT EXISTS tagging_queue (
            item_id TEXT PRIMARY KEY,
            status TEXT NOT NULL DEFAULT 'pending',
            attempts INTEGER NOT NULL DEFAULT 0,
            priority INTEGER NOT NULL DEFAULT 0,
            queued_at TEXT NOT NULL,
            started_at TEXT,
            finished_at TEXT,
            last_error TEXT
        );

        CREATE TABLE IF NOT EXISTS tag_results (
            item_id TEXT PRIMARY KEY,
            content_hash TEXT NOT NULL,
            model_version TEXT NOT NULL,
            language TEXT DEFAULT '',
            primary_tag TEXT,
            tags_json TEXT NOT NULL,
            sentiment_label TEXT NOT NULL,
            sentiment_score REAL NOT NULL,
            relevance_score REAL NOT NULL DEFAULT 0,
            confidence REAL NOT NULL,
            impact_type TEXT NOT NULL,
            matching_keywords_json TEXT NOT NULL DEFAULT '[]',
            excluded INTEGER NOT NULL DEFAULT 0,
            exclude_reason TEXT,
            inference_ms INTEGER NOT NULL DEFAULT 0,
            cache_hit INTEGER NOT NULL DEFAULT 0,
            tagged_at TEXT NOT NULL
        );

        CREATE TABLE IF NOT EXISTS tag_cache (
            content_hash TEXT NOT NULL,
            model_version TEXT NOT NULL,
            payload_json TEXT NOT NULL,
            created_at TEXT NOT NULL,
            PRIMARY KEY (content_hash, model_version)
        );
        """
    )
    conn.executescript(
        """
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
        """
    )
    if mining_engine is not None:
        mining_engine.seed_builtin_entities(conn)
    conn.commit()


def row_to_dict(row: sqlite3.Row) -> dict[str, Any]:
    return {key: row[key] for key in row.keys()}


def get_progress(
    conn: sqlite3.Connection,
    cache_hits: int = 0,
    inference_ms: int = 0,
    model_ctx: Any | None = None,
) -> dict[str, Any]:
    queue_counts = {
        row["status"]: row["count"]
        for row in conn.execute(
            "SELECT status, COUNT(*) AS count FROM tagging_queue GROUP BY status"
        )
    }
    result_counts = conn.execute(
        """
        SELECT
            COUNT(*) AS tagged_total,
            SUM(CASE WHEN excluded = 1 THEN 1 ELSE 0 END) AS excluded_total
        FROM tag_results
        WHERE model_version = ?
        """,
        (MODEL_VERSION,),
    ).fetchone()
    category_counts = {
        row["primary_tag"]: int(row["count"] or 0)
        for row in conn.execute(
            """
            SELECT primary_tag, COUNT(*) AS count
            FROM tag_results
            WHERE excluded = 0
              AND model_version = ?
            GROUP BY primary_tag
            """,
            (MODEL_VERSION,),
        ).fetchall()
        if row["primary_tag"]
    }
    cache_total = conn.execute(
        "SELECT COUNT(*) AS count FROM tag_cache WHERE model_version = ?",
        (MODEL_VERSION,),
    ).fetchone()["count"]
    processed_total = int(result_counts["tagged_total"] or 0)
    excluded_total = int(result_counts["excluded_total"] or 0)
    completed_total = max(processed_total - excluded_total, 0)
    pending_total = int(queue_counts.get("pending", 0)) + int(queue_counts.get("tagging", 0))
    status = "running" if pending_total else "complete"
    return {
        "status": status,
        "generated_at": utc_now(),
        "total_count": processed_total + pending_total,
        "in_progress_count": pending_total,
        "completed_count": completed_total,
        "excluded_count": excluded_total,
        "category_counts": category_counts,
        "queue": {
            "pending": int(queue_counts.get("pending", 0)),
            "tagging": int(queue_counts.get("tagging", 0)),
            "tagged": int(queue_counts.get("tagged", 0)),
            "excluded": int(queue_counts.get("excluded", 0)),
            "failed": int(queue_counts.get("failed", 0)),
        },
        "processed_total": processed_total,
        "excluded_total": excluded_total,
        "model_version": MODEL_VERSION,
        "batch_size": BATCH_SIZE,
        "cache_hit_count": cache_hits,
        "cache_total": int(cache_total or 0),
        "cache_hit_rate": round(cache_hits / max(processed_total + cache_hits, 1), 4),
        "last_inference_ms": inference_ms,
        "runtime": torch_runtime_info(
            torch_module=model_ctx.get("torch") if isinstance(model_ctx, dict) else None,
            model_ctx=model_ctx,
        ),
    }


def write_progress(
    conn: sqlite3.Connection,
    cache_hits: int = 0,
    inference_ms: int = 0,
    model_ctx: Any | None = None,
) -> None:
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    PROGRESS_PATH.write_text(
        json.dumps(get_progress(conn, cache_hits, inference_ms, model_ctx), ensure_ascii=False, indent=2),
        encoding="utf-8",
    )


def write_blocked_progress(conn: sqlite3.Connection, errors: list[dict[str, str]]) -> None:
    progress = get_progress(conn)
    progress["status"] = "blocked"
    progress["blocked_reason"] = "required_model_unavailable"
    progress["model_errors"] = errors
    progress["required_models"] = {
        "bge": BGE_CATEGORY_MODEL,
        "finbert_en": EN_FINBERT_MODEL,
        "finbert_ko": KO_FINBERT_MODEL,
        "ner": "mining_engine entity dictionary",
    }
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    PROGRESS_PATH.write_text(json.dumps(progress, ensure_ascii=False, indent=2), encoding="utf-8")
    for error in errors:
        append_system_log(
            f"[{utc_now()}] [NLP][MODEL_ERROR] {error.get('component')}: {error.get('message')}"
        )


def text_score(text: str, keywords: list[str]) -> int:
    lower = text.lower()
    total = 0
    for keyword in keywords:
        token = keyword.lower().strip()
        if not token:
            continue
        if re.fullmatch(r"[a-z0-9][a-z0-9+.#-]*", token):
            pattern = rf"(?<![a-z0-9]){re.escape(token)}(?![a-z0-9])"
            total += len(re.findall(pattern, lower))
        elif any("\uac00" <= char <= "\ud7a3" for char in token):
            pattern = re.escape(token).replace(r"\ ", r"\s+")
            pattern = rf"(?<![\uac00-\ud7a3a-z0-9]){pattern}(?![\uac00-\ud7a3a-z0-9])"
            total += len(re.findall(pattern, lower))
        else:
            total += lower.count(token)
    return total


def infer_keyword(item: dict[str, Any]) -> dict[str, Any]:
    text = " ".join(
        str(item.get(key) or "")
        for key in ["source", "country", "language", "title", "content"]
    )
    category_scores = {
        category: text_score(text, keywords)
        for category, keywords in SCORING_CATEGORY_KEYWORDS.items()
    }
    top_category, top_hits = max(category_scores.items(), key=lambda pair: pair[1])
    total_hits = sum(category_scores.values())

    relevance = min(1.0, (top_hits / 3.0) + (total_hits / 18.0))
    confidence = max(0.35, min(0.96, 0.45 + relevance * 0.45 + min(top_hits, 4) * 0.03))

    negative = text_score(text, NEGATIVE_KEYWORDS)
    positive = text_score(text, POSITIVE_KEYWORDS)
    sentiment_raw = positive - negative
    if sentiment_raw <= -2:
        sentiment_label = "Panic"
        sentiment_score = -0.82
        impact_type = "negative"
    elif sentiment_raw < 0:
        sentiment_label = "Warning"
        sentiment_score = -0.42
        impact_type = "negative"
    elif sentiment_raw >= 5:
        sentiment_label = "Positive"
        sentiment_score = 0.94
        impact_type = "positive"
    elif sentiment_raw >= 3:
        sentiment_label = "Positive"
        sentiment_score = 0.82
        impact_type = "positive"
    elif sentiment_raw >= 2:
        sentiment_label = "Positive"
        sentiment_score = 0.68
        impact_type = "positive"
    elif sentiment_raw == 1:
        sentiment_label = "Positive"
        sentiment_score = 0.24
        impact_type = "positive"
    else:
        sentiment_label = "Neutral"
        sentiment_score = 0.0
        impact_type = "neutral"

    tags = [
        {
            "tag": category,
            "score": round(min(1.0, 0.3 + hits / max(top_hits, 1) * 0.6), 4),
            "hits": hits,
        }
        for category, hits in sorted(category_scores.items(), key=lambda pair: pair[1], reverse=True)
        if hits > 0
    ]

    if not tags:
        tags = [{"tag": "Finance", "score": 0.2, "hits": 0}]
        top_category = "Finance"

    is_excluded = relevance < MIN_RELEVANCE
    return {
        "model_version": MODEL_VERSION,
        "primary_tag": top_category,
        "tags": tags,
        "relevance_score": round(relevance, 4),
        "matching_keywords": tags,
        "sentiment_label": sentiment_label,
        "sentiment_score": sentiment_score,
        "confidence": round(confidence, 4),
        "impact_type": impact_type,
        "is_excluded": is_excluded,
        "reason": None if not is_excluded else "low_relevance_realtime_filter",
    }


def model_language_key(item: dict[str, Any]) -> str:
    language = str(item.get("language") or "").lower()
    if language.startswith("ko") or language.startswith("kr"):
        return "ko"
    text = " ".join(str(item.get(key) or "") for key in ["title", "content"])
    if any("\uac00" <= char <= "\ud7a3" for char in text):
        return "ko"
    return "en"


def model_input_text(item: dict[str, Any]) -> str:
    title = " ".join(str(item.get("title") or "").split())
    content = " ".join(str(item.get("content") or "").split())
    if title and title not in content[:180]:
        text = f"{title}. {content}"
    else:
        text = content or title
    return text[:4000]


def resolve_torch_device(torch_module: Any) -> str:
    requested = FILTER_TORCH_DEVICE
    cuda_available = bool(torch_module.cuda.is_available())
    if requested in {"cuda", "gpu"}:
        if not cuda_available:
            raise RuntimeError(
                "GPU/CUDA is required, but torch.cuda.is_available() is false. "
                "Install the CUDA PyTorch wheel and check the NVIDIA driver."
            )
        return "cuda"
    if requested == "cpu":
        if REQUIRE_GPU:
            raise RuntimeError("GPU is required by FILTER_REQUIRE_GPU=1, but FILTER_TORCH_DEVICE=cpu was requested.")
        return "cpu"
    if cuda_available:
        return "cuda"
    if REQUIRE_GPU:
        raise RuntimeError(
            "GPU/CUDA is required by FILTER_REQUIRE_GPU=1, but torch.cuda.is_available() is false."
        )
    return "cpu"


def torch_runtime_info(torch_module: Any | None = None, model_ctx: Any | None = None) -> dict[str, Any]:
    info: dict[str, Any] = {
        "requested_device": FILTER_TORCH_DEVICE,
        "require_gpu": REQUIRE_GPU,
        "device": model_ctx.get("device") if isinstance(model_ctx, dict) else None,
    }
    try:
        torch_ref = torch_module
        if torch_ref is None:
            import torch as torch_ref  # type: ignore
        info.update({
            "torch": getattr(torch_ref, "__version__", ""),
            "torch_cuda_version": getattr(getattr(torch_ref, "version", None), "cuda", None),
            "cuda_available": bool(torch_ref.cuda.is_available()),
            "cuda_device_count": int(torch_ref.cuda.device_count()),
        })
        if torch_ref.cuda.is_available() and torch_ref.cuda.device_count():
            index = torch_ref.cuda.current_device()
            info["gpu_name"] = torch_ref.cuda.get_device_name(index)
            info["cuda_device_index"] = int(index)
        if not info.get("device"):
            info["device"] = "cuda" if info.get("cuda_available") else "cpu"
    except Exception as exc:
        info["error"] = str(exc)
    return info


def try_load_bge_category() -> Any | None:
    if not ENABLE_BGE_CATEGORY:
        if REQUIRE_BGE_CATEGORY:
            raise RuntimeError("BGE category model is disabled by FILTER_ENABLE_BGE_CATEGORY=0.")
        return None
    try:
        import torch  # type: ignore
        from sentence_transformers import SentenceTransformer  # type: ignore
    except Exception as exc:
        if REQUIRE_BGE_CATEGORY:
            raise RuntimeError(
                f"BGE category runtime is required but torch/sentence-transformers is unavailable: {exc}"
            ) from exc
        return None
    try:
        device = resolve_torch_device(torch)
        model = SentenceTransformer(BGE_CATEGORY_MODEL, device=device)
        labels = list(CATEGORY_PROFILES)
        profile_texts = [f"{label}. {CATEGORY_PROFILES[label]}" for label in labels]
        profile_embeddings = model.encode(
            profile_texts,
            normalize_embeddings=True,
            show_progress_bar=False,
        )
        return {
            "name": BGE_CATEGORY_MODEL,
            "model": model,
            "device": device,
            "labels": labels,
            "profile_embeddings": profile_embeddings,
        }
    except Exception as exc:
        if REQUIRE_BGE_CATEGORY:
            raise RuntimeError(f"BGE category model load failed ({BGE_CATEGORY_MODEL}): {exc}") from exc
        return None


def vector_dot(left: Any, right: Any) -> float:
    try:
        return float(left @ right)
    except Exception:
        return float(sum(float(a) * float(b) for a, b in zip(left, right)))


def keyword_top_hits(result: dict[str, Any], category: str) -> int:
    for tag in result.get("tags") or []:
        if tag.get("tag") == category:
            try:
                return int(tag.get("hits") or 0)
            except Exception:
                return 0
    return 0


def infer_bge_categories(model_ctx: Any | None, batch: list[dict[str, Any]], results: list[dict[str, Any]]) -> list[dict[str, Any]]:
    category_ctx = (model_ctx or {}).get("category") if isinstance(model_ctx, dict) else None
    if not category_ctx or not batch:
        return results

    model = category_ctx["model"]
    labels = category_ctx["labels"]
    profile_embeddings = category_ctx["profile_embeddings"]
    texts = [model_input_text(item) or " " for item in batch]
    item_embeddings = model.encode(texts, normalize_embeddings=True, show_progress_bar=False)

    updated: list[dict[str, Any]] = []
    for item, base_result, item_embedding in zip(batch, results, item_embeddings):
        scores = [
            {
                "tag": label,
                "score": round(vector_dot(item_embedding, profile_embedding), 6),
                "source": "bge_m3_category_profile",
                "hits": keyword_top_hits(base_result, label),
            }
            for label, profile_embedding in zip(labels, profile_embeddings)
        ]
        for row in scores:
            row["bge_score"] = row["score"]
            row["keyword_boost"] = round(min(0.18, float(row["hits"]) * 0.035), 6)
            row["score"] = round(float(row["bge_score"]) + float(row["keyword_boost"]), 6)
        scores.sort(key=lambda row: row["score"], reverse=True)
        top = scores[0]
        second = scores[1] if len(scores) > 1 else {"score": 0.0, "tag": ""}
        top_score = float(top["score"])
        second_score = float(second["score"])
        margin = top_score - second_score
        top_hits = int(top.get("hits") or 0)
        keyword_category = str(base_result.get("primary_tag") or "")
        keyword_hits = keyword_top_hits(base_result, keyword_category)
        keyword_disagrees = bool(keyword_category and keyword_category != top["tag"] and keyword_hits >= 2)
        keyword_supported_top = top_hits > 0
        no_keyword_evidence = keyword_hits == 0 and not any(int(row.get("hits") or 0) for row in scores)

        bge_relevance = max(0.0, min(1.0, top_score * 1.45))
        confidence = max(
            float(base_result.get("confidence") or 0.35),
            min(0.96, 0.34 + top_score * 0.72 + max(0.0, margin) * 1.25),
        )
        ambiguous = (
            top_score < BGE_CATEGORY_MIN_SCORE
            or margin < BGE_CATEGORY_AMBIGUOUS_MARGIN
            or keyword_disagrees
        )

        result = dict(base_result)
        bge_has_clear_margin = top_score >= BGE_CATEGORY_MIN_SCORE and margin >= BGE_CATEGORY_AMBIGUOUS_MARGIN
        bge_has_strong_score = top_score >= (BGE_CATEGORY_MIN_SCORE + 0.12) and margin >= 0.035
        should_use_bge = (
            (keyword_supported_top and top_score >= BGE_CATEGORY_MIN_SCORE)
            or bge_has_clear_margin
            or bge_has_strong_score
            or (no_keyword_evidence and bge_has_clear_margin)
        )
        if should_use_bge:
            result["primary_tag"] = top["tag"]
            result["tags"] = [
                {
                    "tag": row["tag"],
                    "score": round(max(0.0, min(1.0, row["score"])), 4),
                    "hits": row["hits"],
                    "source": row["source"],
                    "bge_score": round(float(row.get("bge_score") or 0), 4),
                    "keyword_boost": round(float(row.get("keyword_boost") or 0), 4),
                }
                for row in scores[:4]
            ]
            result["matching_keywords"] = result["tags"]
            result["relevance_score"] = round(max(float(result.get("relevance_score") or 0), bge_relevance), 4)
            result["confidence"] = round(confidence, 4)
            result["is_excluded"] = result["relevance_score"] < MIN_RELEVANCE
            result["reason"] = None if not result["is_excluded"] else "low_bge_market_relevance"
        else:
            result["category_source"] = "keyword_fallback_bge_ambiguous"
            result["category_ambiguous"] = True
            result["bge_category_model"] = BGE_CATEGORY_MODEL
            result["bge_category_top_score"] = round(top_score, 6)
            result["bge_category_second_score"] = round(second_score, 6)
            result["bge_category_margin"] = round(margin, 6)
            result["category_scores"] = scores
            updated.append(result)
            continue

        result["category_source"] = "bge_m3_category_profile"
        result["bge_category_model"] = BGE_CATEGORY_MODEL
        result["bge_category_top_score"] = round(top_score, 6)
        result["bge_category_second_score"] = round(second_score, 6)
        result["bge_category_margin"] = round(margin, 6)
        result["category_ambiguous"] = bool(ambiguous or margin < BGE_CATEGORY_QWEN_MARGIN)
        result["category_scores"] = scores
        updated.append(result)
    return updated


def finbert_prediction_to_result(label: str, confidence: float, probabilities: dict[str, float]) -> dict[str, Any]:
    label = label.lower()
    positive_prob = probabilities.get("positive", 0.0)
    negative_prob = probabilities.get("negative", 0.0)
    neutral_prob = probabilities.get("neutral", 0.0)

    if label == "positive":
        sentiment_label = "Positive"
        sentiment_score = min(0.94, 0.24 + positive_prob * 0.74)
        impact_type = "positive"
    elif label == "negative":
        margin = negative_prob - max(positive_prob, neutral_prob)
        if negative_prob >= 0.8 and margin >= 0.25:
            sentiment_label = "Panic"
            sentiment_score = -0.82
        else:
            sentiment_label = "Warning"
            sentiment_score = -min(0.72, 0.24 + negative_prob * 0.55)
        impact_type = "negative"
    else:
        tilt = positive_prob - negative_prob
        if abs(tilt) >= 0.08 and neutral_prob < 0.95:
            sentiment_score = max(-0.36, min(0.36, tilt * 0.65))
        else:
            sentiment_score = 0.0
        if sentiment_score >= 0.18:
            sentiment_label = "Positive"
            impact_type = "positive"
        elif sentiment_score <= -0.18:
            sentiment_label = "Warning"
            impact_type = "negative"
        else:
            sentiment_label = "Neutral"
            impact_type = "positive" if sentiment_score > 0 else "negative" if sentiment_score < 0 else "neutral"

    return {
        "sentiment_label": sentiment_label,
        "sentiment_score": round(float(sentiment_score), 4),
        "confidence": round(max(float(confidence), 0.35), 4),
        "impact_type": impact_type,
        "finbert_probabilities": probabilities,
    }


def merge_neutral_sentiment(keyword_result: dict[str, Any], model_result: dict[str, Any]) -> dict[str, Any]:
    if model_result.get("sentiment_label") != "Neutral":
        return model_result
    if abs(float(model_result.get("sentiment_score") or 0.0)) > 0:
        return model_result

    keyword_score = float(keyword_result.get("sentiment_score") or 0.0)
    if abs(keyword_score) < 0.24:
        return model_result

    merged = dict(model_result)
    sentiment_score = max(-0.48, min(0.48, keyword_score * 0.75))
    merged["sentiment_score"] = round(sentiment_score, 4)
    if sentiment_score > 0:
        merged["sentiment_label"] = "Positive"
        merged["impact_type"] = "positive"
    else:
        merged["sentiment_label"] = "Warning"
        merged["impact_type"] = "negative"
    merged["sentiment_source"] = "keyword_fallback_after_neutral_model"
    return merged


def infer_finbert_sentiment(model_ctx: Any, lang_key: str, items: list[dict[str, Any]]) -> list[dict[str, Any]]:
    ctx = model_ctx["models"][lang_key]
    tokenizer = ctx["tokenizer"]
    model = ctx["model"]
    torch = model_ctx["torch"]
    device = model_ctx["device"]
    texts = [model_input_text(item) or " " for item in items]

    encoded = tokenizer(
        texts,
        padding=True,
        truncation=True,
        max_length=256,
        return_tensors="pt",
    )
    encoded = {key: value.to(device) for key, value in encoded.items()}

    with torch.no_grad():
        logits = model(**encoded).logits
        probs = torch.nn.functional.softmax(logits, dim=-1).detach().cpu().tolist()

    predictions = []
    id2label = ctx["id2label"]
    for row in probs:
        best_index = max(range(len(row)), key=lambda idx: row[idx])
        label = id2label.get(best_index, str(best_index)).lower()
        probabilities = {
            id2label.get(idx, str(idx)).lower(): round(float(probability), 6)
            for idx, probability in enumerate(row)
        }
        predictions.append(finbert_prediction_to_result(label, float(row[best_index]), probabilities))
    return predictions


def try_load_finbert(category_context: Any | None = None, load_category: bool = True) -> Any | None:
    try:
        import torch  # type: ignore
        from transformers import AutoModelForSequenceClassification, AutoTokenizer  # type: ignore
    except Exception as exc:
        if ALLOW_KEYWORD_SENTIMENT_FALLBACK and not REQUIRE_FINBERT:
            return None
        raise RuntimeError(
            f"FinBERT runtime is required but torch/transformers is unavailable: {exc}. "
            "Install requirements-nlp.txt in the active virtualenv."
        ) from exc

    device = resolve_torch_device(torch)
    models = {}
    for key, model_name in (("en", EN_FINBERT_MODEL), ("ko", KO_FINBERT_MODEL)):
        try:
            tokenizer = AutoTokenizer.from_pretrained(model_name)
            model = AutoModelForSequenceClassification.from_pretrained(model_name)
        except Exception as exc:
            raise RuntimeError(f"FinBERT model load failed ({model_name}): {exc}") from exc
        if device == "cuda":
            model = model.half()
        model.to(device)
        model.eval()
        models[key] = {
            "name": model_name,
            "tokenizer": tokenizer,
            "model": model,
            "id2label": {int(k): str(v).lower() for k, v in model.config.id2label.items()},
        }
    category = try_load_bge_category() if load_category else category_context
    return {"torch": torch, "device": device, "models": models, "category": category}


def load_required_model_context(conn: sqlite3.Connection) -> Any | None:
    errors: list[dict[str, str]] = []
    if REQUIRE_NER and mining_engine is None:
        errors.append({
            "component": "NER",
            "message": "Filter/mining_engine.py import failed, so NER/entity filtering cannot run.",
        })
    if db_utils is None:
        errors.append({
            "component": "DB",
            "message": "Crawling/db_utils.py import failed, so tagging cannot safely update crawler.db.",
        })
    category_ctx = None
    try:
        category_ctx = try_load_bge_category()
    except Exception as exc:
        errors.append({"component": "BGE", "message": str(exc)})
    model_ctx = None
    try:
        model_ctx = try_load_finbert(category_context=category_ctx, load_category=False)
    except Exception as exc:
        errors.append({"component": "FinBERT", "message": str(exc)})
    if errors:
        write_blocked_progress(conn, errors)
        raise RuntimeError("; ".join(f"{row['component']}: {row['message']}" for row in errors))
    return model_ctx


def infer_batch(model_ctx: Any | None, batch: list[dict[str, Any]]) -> list[dict[str, Any]]:
    results = [infer_keyword(item) for item in batch]
    results = infer_bge_categories(model_ctx, batch, results)
    if model_ctx is None:
        if ALLOW_KEYWORD_SENTIMENT_FALLBACK and not REQUIRE_FINBERT:
            return results
        raise RuntimeError("FinBERT models are not loaded.")

    for lang_key in ("en", "ko"):
        indexed_items = [
            (index, item)
            for index, item in enumerate(batch)
            if model_language_key(item) == lang_key
        ]
        if not indexed_items:
            continue
        predictions = infer_finbert_sentiment(model_ctx, lang_key, [item for _, item in indexed_items])
        for (index, _item), prediction in zip(indexed_items, predictions):
            prediction = merge_neutral_sentiment(results[index], prediction)
            results[index].update(prediction)
            results[index]["model_version"] = MODEL_VERSION
    return results


def fetch_batch(conn: sqlite3.Connection, batch_size: int) -> list[dict[str, Any]]:
    rows = conn.execute(
        """
        SELECT q.item_id AS item_id, c.*
        FROM tagging_queue q
        JOIN crawled_items c ON c.id = q.item_id
        WHERE q.status = 'pending'
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
        ORDER BY q.queued_at ASC
        LIMIT ?
        """,
        (batch_size,),
    ).fetchall()
    items = [row_to_dict(row) for row in rows]
    item_ids = [item["item_id"] for item in items]
    if item_ids:
        placeholders = ",".join("?" for _ in item_ids)
        conn.execute(
            f"""
            UPDATE tagging_queue
            SET status = 'tagging', attempts = attempts + 1, started_at = ?, last_error = ''
            WHERE item_id IN ({placeholders})
            """,
            [utc_now(), *item_ids],
        )
        conn.commit()
    return items


def load_cached(conn: sqlite3.Connection, content_hash: str) -> dict[str, Any] | None:
    row = conn.execute(
        """
        SELECT payload_json
        FROM tag_cache
        WHERE content_hash = ? AND model_version = ?
        """,
        (content_hash, MODEL_VERSION),
    ).fetchone()
    if not row:
        return None
    return json.loads(row["payload_json"])


def normalize_result_for_storage(result: dict[str, Any]) -> dict[str, Any]:
    if not result.get("is_excluded"):
        return result
    normalized = dict(result)
    previous_tag = str(normalized.get("primary_tag") or "").strip()
    if previous_tag and previous_tag != EXCLUDED_TAG:
        normalized["pre_exclusion_tag"] = previous_tag
    reason = str(normalized.get("reason") or normalized.get("exclude_reason") or "excluded")
    excluded_tag = {
        "tag": EXCLUDED_TAG,
        "score": 0.0,
        "hits": 0,
        "reason": reason,
    }
    if previous_tag and previous_tag != EXCLUDED_TAG:
        excluded_tag["previous_tag"] = previous_tag
    normalized["primary_tag"] = EXCLUDED_TAG
    normalized["tags"] = [excluded_tag]
    normalized["matching_keywords"] = [excluded_tag]
    return normalized


def save_result(conn: sqlite3.Connection, item: dict[str, Any], result: dict[str, Any]) -> None:
    now = utc_now()
    result = normalize_result_for_storage(result)
    result_json = json.dumps(result, ensure_ascii=False)
    conn.execute(
        """
        INSERT OR REPLACE INTO tag_cache (content_hash, model_version, payload_json, created_at)
        VALUES (?, ?, ?, ?)
        """,
        (item["content_hash"], MODEL_VERSION, result_json, now),
    )
    conn.execute(
        """
        INSERT OR REPLACE INTO tag_results (
            item_id, content_hash, model_version, language, primary_tag, tags_json,
            sentiment_label, sentiment_score, relevance_score, confidence, impact_type,
            matching_keywords_json, excluded, exclude_reason, inference_ms, cache_hit, tagged_at
        )
        VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
        """,
        (
            item["item_id"],
            item["content_hash"],
            MODEL_VERSION,
            item.get("language") or "",
            result["primary_tag"],
            json.dumps(result["tags"], ensure_ascii=False),
            result["sentiment_label"],
            float(result["sentiment_score"]),
            float(result.get("relevance_score") or 0),
            float(result["confidence"]),
            result["impact_type"],
            json.dumps(result.get("matching_keywords") or [], ensure_ascii=False),
            1 if result["is_excluded"] else 0,
            result.get("reason"),
            int(result.get("inference_ms") or 0),
            int(result.get("cache_hit") or 0),
            now,
        ),
    )
    conn.execute(
        """
        UPDATE tagging_queue
        SET status = ?, finished_at = ?, last_error = ''
        WHERE item_id = ?
        """,
        ("excluded" if result["is_excluded"] else "tagged", now, item["item_id"]),
    )


def process_once(conn: sqlite3.Connection, model_ctx: Any | None) -> tuple[int, int, int]:
    items = fetch_batch(conn, BATCH_SIZE)
    if not items:
        write_progress(conn, model_ctx=model_ctx)
        return 0, 0, 0

    cache_hits = 0
    to_infer: list[dict[str, Any]] = []
    for item in items:
        cached = load_cached(conn, item["content_hash"])
        if cached:
            cache_hits += 1
            cached["cache_hit"] = 1
            if mining_engine is not None:
                cached = mining_engine.apply_ner_gate(conn, item, cached)
                cached = mining_engine.apply_qwen_category_review(conn, item, cached)
            save_result(conn, item, cached)
        else:
            to_infer.append(item)

    start = time.perf_counter()
    inferred = infer_batch(model_ctx, to_infer) if to_infer else []
    inference_ms = int((time.perf_counter() - start) * 1000)
    for item, result in zip(to_infer, inferred):
        if mining_engine is not None:
            result = mining_engine.apply_ner_gate(conn, item, result)
            result = mining_engine.apply_qwen_category_review(conn, item, result)
        save_result(conn, item, result)
    conn.commit()
    write_progress(conn, cache_hits, inference_ms, model_ctx)
    return len(items), cache_hits, inference_ms


def repair_stale_tagging(conn: sqlite3.Connection) -> None:
    conn.execute(
        """
        UPDATE tagging_queue
        SET status = 'pending', last_error = 'recovered_stale_tagging'
        WHERE status = 'tagging'
          AND datetime(started_at) < datetime('now', '-10 minutes')
        """
    )
    conn.commit()


def exclude_low_quality_queue_items(conn: sqlite3.Connection) -> None:
    now = utc_now()
    conn.execute(
        """
        UPDATE tagging_queue
        SET status = 'excluded',
            finished_at = ?,
            last_error = 'low_quality_title_only_not_tagged'
        WHERE status IN ('pending', 'tagging')
          AND item_id IN (
              SELECT c.id
              FROM crawled_items c
              WHERE COALESCE(c.source_group, '') NOT IN ('truth', 'russia', 'x', 'market', 'test')
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
        """,
        (now,),
    )
    conn.commit()


def enqueue_items_missing_current_model(conn: sqlite3.Connection) -> None:
    rows = conn.execute(
        """
        SELECT c.id AS item_id
        FROM crawled_items c
        LEFT JOIN tag_results r ON r.item_id = c.id
        WHERE (r.item_id IS NULL OR r.model_version <> ?)
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
        """,
        (MODEL_VERSION,),
    ).fetchall()
    if not rows:
        return
    now = utc_now()
    for row in rows:
        conn.execute(
            """
            INSERT INTO tagging_queue (item_id, status, queued_at, last_error)
            VALUES (?, 'pending', ?, '')
            ON CONFLICT(item_id) DO UPDATE SET
                status = 'pending',
                queued_at = excluded.queued_at,
                last_error = ''
            """,
            (row["item_id"], now),
        )
    conn.commit()


def auto_review_excluded(conn: sqlite3.Connection) -> int:
    if not AUTO_REVIEW_EXCLUDED:
        return 0
    rows = conn.execute(
        """
        SELECT r.item_id, r.content_hash, r.primary_tag, r.sentiment_label,
               r.relevance_score, r.confidence, r.excluded, c.source_group
        FROM tag_results r
        JOIN crawled_items c ON c.id = r.item_id
        WHERE r.excluded = 1
          AND r.model_version = ?
        """,
        (MODEL_VERSION,),
    ).fetchall()
    changed = 0
    now = utc_now()
    for row in rows:
        relevance = float(row["relevance_score"] or 0)
        confidence = float(row["confidence"] or 0)
        source_group = (row["source_group"] or "").lower()
        should_restore = (
            relevance >= 0.24
            and confidence >= 0.52
            and source_group in {"gov", "news", "truth", "russia", "china"}
        )
        if not should_restore:
            continue
        conn.execute(
            """
            UPDATE tag_results
            SET excluded = 0,
                exclude_reason = 'builtin_ai_retag_review'
            WHERE item_id = ?
            """,
            (row["item_id"],),
        )
        conn.execute(
            """
            UPDATE tagging_queue
            SET status = 'tagged', finished_at = ?, last_error = ''
            WHERE item_id = ?
            """,
            (now, row["item_id"]),
        )
        conn.execute(
            """
            INSERT INTO label_feedback (
                item_id, content_hash, original_tag, original_sentiment,
                corrected_tag, corrected_sentiment, approved, reviewer,
                notes, source, created_at
            ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
            """,
            (
                row["item_id"],
                row["content_hash"],
                row["primary_tag"],
                row["sentiment_label"],
                row["primary_tag"],
                row["sentiment_label"],
                1,
                "builtin_ai_retag",
                "Excluded item restored by built-in AI review for training.",
                "builtin_ai_retag",
                utc_now(),
            ),
        )
        changed += 1
    if changed:
        conn.commit()
    return changed


def run_post_processors(conn: sqlite3.Connection) -> None:
    if mining_engine is None:
        return
    if ENABLE_QWEN_EXCLUDED_REVIEW:
        try:
            status = mining_engine.run_qwen_excluded_review(conn)
            mining_engine.set_engine_state(conn, "last_excluded_review_result", json.dumps(status, ensure_ascii=False))
            conn.commit()
        except Exception as exc:
            mining_engine.set_engine_state(conn, "last_excluded_review_result", f"error: {exc}")
            conn.commit()
    else:
        mining_engine.set_engine_state(conn, "last_excluded_review_result", "disabled: gpt55_excluded_review_off")
        conn.commit()
    if ENABLE_WON_ENTITY_UPDATE:
        try:
            status = mining_engine.run_won_entity_update(conn)
            mining_engine.set_engine_state(conn, "last_won_entity_update_result", json.dumps(status, ensure_ascii=False))
            conn.commit()
        except Exception as exc:
            mining_engine.set_engine_state(conn, "last_won_entity_update_result", f"error: {exc}")
            conn.commit()
    else:
        mining_engine.set_engine_state(conn, "last_won_entity_update_result", "disabled: gpt55_entity_update_off")
        conn.commit()
    if ENABLE_BGE_DEDUP:
        try:
            last_value = mining_engine.get_engine_state(conn, "last_bge_dedup_epoch")
            if not last_value or time.time() - float(last_value) >= BGE_DEDUP_INTERVAL_SECONDS:
                status = mining_engine.run_bge_dedup(conn)
                mining_engine.set_engine_state(conn, "last_bge_dedup_epoch", str(time.time()))
                mining_engine.set_engine_state(conn, "last_bge_dedup_result", json.dumps(status, ensure_ascii=False))
                conn.commit()
        except Exception as exc:
            mining_engine.set_engine_state(conn, "last_bge_dedup_result", f"error: {exc}")
            conn.commit()


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--once", action="store_true", help="Process one batch and exit")
    parser.add_argument("--loop", action="store_true", help="Run continuously")
    parser.add_argument("--interval", type=float, default=5.0, help="Polling interval in seconds")
    args = parser.parse_args()

    DB_PATH.parent.mkdir(parents=True, exist_ok=True)
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    conn = connect()
    init_db(conn)
    enqueue_items_missing_current_model(conn)
    exclude_low_quality_queue_items(conn)
    try:
        model_ctx = load_required_model_context(conn)
    except Exception as exc:
        append_system_log(f"[{utc_now()}] [NLP][MODEL_ERROR] tagging blocked: {exc}")
        conn.close()
        return 2

    try:
        while True:
            repair_stale_tagging(conn)
            processed, cache_hits, inference_ms = process_once(conn, model_ctx)
            if processed:
                auto_review_excluded(conn)
                run_post_processors(conn)
                write_progress(conn, cache_hits, inference_ms, model_ctx)
            if args.once or not args.loop:
                break
            if processed < BATCH_SIZE:
                time.sleep(args.interval)
    finally:
        write_progress(conn, model_ctx=model_ctx)
        conn.close()
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
