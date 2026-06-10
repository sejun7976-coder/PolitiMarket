from flask import Flask, render_template, jsonify, request, send_from_directory
import os
import sys
import json
import subprocess
import re
import requests
import signal
import sqlite3
import urllib3
from datetime import datetime, timedelta, timezone
from zoneinfo import ZoneInfo
from werkzeug.utils import secure_filename

urllib3.disable_warnings(urllib3.exceptions.InsecureRequestWarning)

app = Flask(__name__, template_folder='templates')


def env_int(name, default, minimum=None):
    try:
        value = int(os.environ.get(name, default))
    except Exception:
        value = int(default)
    if minimum is not None:
        value = max(minimum, value)
    return value


def crawler_lookback_where_sql(alias="c"):
    lookback_enabled = os.environ.get("CRAWLER_LOOKBACK_DAYS", "2") != "0"
    if not lookback_enabled:
        return ""
    lookback_days = env_int("CRAWLER_LOOKBACK_DAYS", "2", minimum=1)
    tz_name = os.environ.get("CRAWLER_DAY_TIMEZONE", "Asia/Seoul")
    try:
        local_tz = ZoneInfo(tz_name)
    except Exception:
        local_tz = ZoneInfo("Asia/Seoul")
    start_date = datetime.now(local_tz).date() - timedelta(days=lookback_days - 1)
    start_local = datetime.combine(start_date, datetime.min.time(), tzinfo=local_tz)
    start_utc = start_local.astimezone(timezone.utc).isoformat()
    start_ymd = start_local.strftime("%Y-%m-%d")
    return (
        " AND ("
        f"({alias}.source = 'News_PeopleCN_KO' "
        f"AND instr({alias}.url, '/n3/') > 0 "
        f"AND date(substr({alias}.url, instr({alias}.url, '/n3/') + 4, 4) || '-' || "
        f"substr({alias}.url, instr({alias}.url, '/n3/') + 9, 2) || '-' || "
        f"substr({alias}.url, instr({alias}.url, '/n3/') + 11, 2)) >= date('{start_ymd}'))"
        " OR "
        f"(COALESCE({alias}.source, '') <> 'News_PeopleCN_KO' "
        f"AND datetime(COALESCE(NULLIF({alias}.published_at, ''), {alias}.created_at)) >= datetime('{start_utc}'))"
        ")"
    )

crolling_dir = os.path.dirname(os.path.abspath(__file__))
project_dir = os.path.abspath(os.path.join(crolling_dir, ".."))
crawling_dir = os.path.abspath(os.path.join(crolling_dir, "../Crawling"))
filter_dir = os.path.abspath(os.path.join(crolling_dir, "../Filter"))
if crawling_dir not in sys.path:
    sys.path.insert(0, crawling_dir)

pid_path = os.path.join(crawling_dir, "crawler.pid")
log_path = os.path.join(crawling_dir, "crawler_loop.log")
crawler_status_path = os.path.join(crawling_dir, "crawler_status.json")
backfill_lock_path = os.path.join(crawling_dir, "backfill.lock")
crawler_config_path = os.path.join(crawling_dir, "crawler_config.json")
llm_config_path = os.path.join(crawling_dir, "llm_config.json")
won_config_path = llm_config_path
legacy_won_config_path = os.path.join(crawling_dir, "won_config.json")
summary_cache_path = os.path.join(filter_dir, "output", "summary_cache.json")
test_mode_path = os.path.join(crawling_dir, "test_mode.json")
test_dataset_dir = os.path.join(project_dir, "Test_dataset")
default_won_model = "gpt-5.5"
default_llm_reasoning_effort = "medium"
supported_won_models = {
    "gpt-5.5": "GPT-5.5 (reasoning: medium)",
}
collector_keys = ["x", "truth", "gov", "news", "thinktank", "axios", "market", "russia", "china"]
default_x_api_config = {
    "bearer_token": "",
    "accounts": ["realDonaldTrump", "Jaemyung_Lee", "mofa_kr", "ROK_MND"],
    "queries": [],
    "recent_lookback_days": 1,
    "backfill_days": 7,
    "use_full_archive": False,
    "exclude_retweets": True,
    "exclude_replies": True,
}
default_crawler_config = {
    "enabled": {key: True for key in collector_keys},
    "x_api": default_x_api_config,
}
CRAWL_INTERVAL_SECONDS = int(os.environ.get("CRAWLER_INTERVAL_SECONDS", "360"))
CRAWL_JITTER_RATIO = float(os.environ.get("CRAWLER_JITTER_RATIO", "0.2"))
MARKET_INTERVAL_SECONDS = int(os.environ.get("CRAWLER_MARKET_INTERVAL_SECONDS", "15"))

test_category_terms = {
    "IT": "AI semiconductor chip cloud data center",
    "Energy": "oil gas lng energy power electricity",
    "Finance": "bank treasury dollar finance stock rate",
    "Healthcare": "health pharma vaccine hospital bio",
    "Commodities": "gold silver copper wheat steel commodity",
    "Defense": "defense security military drone weapon",
    "Chemicals": "chemical battery petrochemical fertilizer lithium",
    "Shipbuilding": "ship vessel shipping shipbuilding naval",
}

test_scenarios = {
    "panic": {
        "label": "Panic",
        "items_per_category": 2,
        "sentiment_words": "risk sanction crisis fall decline conflict volatility",
        "summary": "악재가 여러 섹터에 동시에 쌓이는 공포 테스트 데이터",
    },
    "warning": {
        "label": "Warning",
        "items_per_category": 1,
        "sentiment_words": "risk",
        "summary": "약한 악재가 전반적으로 퍼지는 주의 테스트 데이터",
    },
    "neutral": {
        "label": "Neutral",
        "items_per_category": 1,
        "sentiment_words": "briefing update monitoring baseline",
        "summary": "방향성이 거의 없는 중립 테스트 데이터",
    },
    "positive": {
        "label": "Positive",
        "items_per_category": 1,
        "sentiment_words": "growth investment",
        "summary": "호재는 있지만 과열까지는 아닌 긍정 테스트 데이터",
    },
    "overheated": {
        "label": "Overheated",
        "items_per_category": 5,
        "sentiment_words": "growth deal agreement investment cooperation support profit record expand",
        "summary": "강한 호재가 반복되어 과열까지 올라가는 테스트 데이터",
    },
}

def load_crawler_config():
    config = json.loads(json.dumps(default_crawler_config))
    if os.path.exists(crawler_config_path):
        try:
            with open(crawler_config_path, 'r', encoding='utf-8') as f:
                saved = json.load(f)
            enabled = saved.get("enabled", {})
            for key in collector_keys:
                if key in enabled:
                    config["enabled"][key] = bool(enabled[key])
            x_api = saved.get("x_api", {})
            if isinstance(x_api, dict):
                for key in default_x_api_config:
                    if key in x_api:
                        config["x_api"][key] = x_api[key]
        except Exception as e:
            print(f"Error loading crawler config: {e}")
    env_token = os.environ.get("X_BEARER_TOKEN", "").strip()
    if env_token:
        config["x_api"]["bearer_token"] = env_token
    return config

def save_crawler_config(config):
    normalized = json.loads(json.dumps(default_crawler_config))
    enabled = config.get("enabled", {})
    for key in collector_keys:
        if key in enabled:
            normalized["enabled"][key] = bool(enabled[key])
    x_api = config.get("x_api", {})
    if isinstance(x_api, dict):
        for key in default_x_api_config:
            if key in x_api:
                normalized["x_api"][key] = x_api[key]
    with open(crawler_config_path, 'w', encoding='utf-8') as f:
        json.dump(normalized, f, ensure_ascii=False, indent=2)
    return normalized

def normalize_x_lines(value):
    if isinstance(value, list):
        values = value
    else:
        values = re.split(r"[\n,]+", str(value or ""))
    result = []
    seen = set()
    for raw in values:
        item = str(raw or "").strip()
        if not item:
            continue
        key = item.lower()
        if key in seen:
            continue
        seen.add(key)
        result.append(item)
    return result

def with_default_x_accounts(accounts):
    merged = normalize_x_lines(accounts)
    existing = {account.lower() for account in merged}
    for account in default_x_api_config["accounts"]:
        if account.lower() not in existing:
            merged.append(account)
            existing.add(account.lower())
    return merged

def public_x_api_config(include_token=False):
    config = load_crawler_config().get("x_api", {})
    token = str(config.get("bearer_token") or "")
    public = {
        "configured": bool(token),
        "token_masked": mask_secret(token) if token else "",
        "accounts": with_default_x_accounts(config.get("accounts")),
        "queries": normalize_x_lines(config.get("queries")),
        "recent_lookback_days": int(config.get("recent_lookback_days") or 1),
        "backfill_days": int(config.get("backfill_days") or 7),
        "use_full_archive": bool(config.get("use_full_archive")),
        "exclude_retweets": bool(config.get("exclude_retweets", True)),
        "exclude_replies": bool(config.get("exclude_replies", True)),
    }
    if include_token:
        public["bearer_token"] = token
    return public

def load_test_mode():
    if os.path.exists(test_mode_path):
        try:
            with open(test_mode_path, "r", encoding="utf-8") as f:
                data = json.load(f)
            return data if isinstance(data, dict) else {"active": False}
        except Exception:
            pass
    return {"active": False}

def list_test_dataset_files():
    os.makedirs(test_dataset_dir, exist_ok=True)
    files = []
    for name in sorted(os.listdir(test_dataset_dir)):
        path = os.path.join(test_dataset_dir, name)
        if os.path.isfile(path) and name.lower().endswith(".json"):
            files.append({
                "name": name,
                "size": os.path.getsize(path),
                "updated_at": datetime.fromtimestamp(os.path.getmtime(path)).isoformat(timespec="seconds"),
            })
    return files

def save_test_mode(active=False, scenario="", item_count=0):
    data = {
        "active": bool(active),
        "scenario": scenario or "",
        "item_count": int(item_count or 0),
        "updated_at": time_str(),
    }
    with open(test_mode_path, "w", encoding="utf-8") as f:
        json.dump(data, f, ensure_ascii=False, indent=2)
    return data

def read_pid():
    if not os.path.exists(pid_path):
        return None
    try:
        with open(pid_path, "r", encoding="utf-8") as f:
            return int((f.read() or "").strip())
    except Exception:
        return None

def build_test_items(scenario_key):
    scenario = test_scenarios.get(scenario_key)
    if not scenario:
        raise ValueError("알 수 없는 테스트 시나리오입니다.")
    items = []
    from datetime import timedelta
    now = datetime.now(timezone.utc)
    index = 0
    for category, terms in test_category_terms.items():
        for item_no in range(int(scenario["items_per_category"])):
            index += 1
            title = f"[{scenario['label']} TEST] {category} scenario signal {item_no + 1}"
            content = (
                f"{terms}. {scenario['sentiment_words']}. "
                f"This synthetic crawler item is designed for {category} tagging and {scenario['label']} market sentiment testing."
            )
            published_at = (now - timedelta(minutes=index)).isoformat(timespec="seconds")
            items.append({
                "source": f"TestMode_{scenario['label']}",
                "source_group": "test",
                "country": "TEST",
                "language": "en",
                "title": title,
                "content": content,
                "url": f"https://local.test/politimarket/{scenario_key}/{category.lower()}/{item_no + 1}",
                "raw_url": f"https://local.test/politimarket/{scenario_key}/{category.lower()}/{item_no + 1}",
                "published_at": published_at,
            })
    return items

def normalize_test_items(items):
    if not isinstance(items, list) or not items:
        raise ValueError("items 배열에 테스트 데이터를 1개 이상 넣어주세요.")
    normalized = []
    now = datetime.now(timezone.utc).isoformat(timespec="seconds")
    for idx, item in enumerate(items, 1):
        if not isinstance(item, dict):
            raise ValueError("items 배열은 객체 목록이어야 합니다.")
        title = str(item.get("title") or f"Custom test item {idx}")
        content = str(item.get("content") or "")
        if not content.strip():
            raise ValueError(f"{idx}번째 item의 content가 비어 있습니다.")
        source = str(item.get("source") or "TestMode_Custom")
        normalized.append({
            "source": source,
            "source_group": "test",
            "country": str(item.get("country") or "TEST"),
            "language": str(item.get("language") or "en"),
            "title": title,
            "content": content,
            "url": str(item.get("url") or f"https://local.test/politimarket/custom/{idx}"),
            "raw_url": str(item.get("raw_url") or item.get("url") or f"https://local.test/politimarket/custom/{idx}"),
            "published_at": str(item.get("published_at") or now),
        })
    return normalized

def delete_test_items(conn):
    rows = conn.execute("SELECT id, content_hash FROM crawled_items WHERE source_group = 'test'").fetchall()
    if not rows:
        return 0
    item_ids = [row["id"] for row in rows]
    hashes = [row["content_hash"] for row in rows]
    placeholders = ",".join("?" for _ in item_ids)
    conn.execute(f"DELETE FROM tag_results WHERE item_id IN ({placeholders})", item_ids)
    conn.execute(f"DELETE FROM tagging_queue WHERE item_id IN ({placeholders})", item_ids)
    conn.execute(f"DELETE FROM crawled_items WHERE id IN ({placeholders})", item_ids)
    hash_placeholders = ",".join("?" for _ in hashes)
    conn.execute(f"DELETE FROM tag_cache WHERE content_hash IN ({hash_placeholders})", hashes)
    return len(item_ids)

def reset_crawl_data():
    if crawling_dir not in sys.path:
        sys.path.insert(0, crawling_dir)
    import db_utils

    db_utils.init_db()
    db_path = os.path.join(crawling_dir, "crawler.db")
    tables = [
        "dedup_group_members",
        "dedup_groups",
        "item_embeddings",
        "llm_excluded_reviews",
        "ner_filter_events",
        "label_feedback",
        "tag_cache",
        "tag_results",
        "tagging_queue",
        "crawled_items",
        "engine_state",
    ]
    with db_utils.connect(db_path) as conn:
        before = conn.execute("SELECT COUNT(*) FROM crawled_items").fetchone()[0]
        for table in tables:
            conn.execute(f"DELETE FROM {table}")
        conn.commit()

    save_test_mode(False, "", 0)
    if os.path.exists(summary_cache_path):
        os.remove(summary_cache_path)

    progress_dir = os.path.join(filter_dir, "output")
    os.makedirs(progress_dir, exist_ok=True)
    with open(os.path.join(progress_dir, "tagging_progress.json"), "w", encoding="utf-8") as f:
        json.dump({
            "status": "idle",
            "total_count": 0,
            "completed_count": 0,
            "in_progress_count": 0,
            "excluded_count": 0,
            "failed_count": 0,
            "category_counts": {},
            "model_version": "",
            "generated_at": datetime.now(timezone.utc).isoformat(timespec="seconds"),
        }, f, ensure_ascii=False, indent=2)

    run_analysis_only()
    export_data_js_current_mode()

    status_data = apply_scheduler_state_to_crawler_status({}, load_crawler_config(), is_pid_running(read_pid()))
    save_crawler_status(status_data)
    with open(log_path, "a", encoding="utf-8") as f:
        f.write(f"[{time_str()}] [SYSTEM] reset crawl dataset; removed {before} crawled items and derivative analysis rows.\n")
    return before

def reset_tagging_work():
    if crawling_dir not in sys.path:
        sys.path.insert(0, crawling_dir)
    import db_utils

    db_utils.init_db()
    db_path = os.path.join(crawling_dir, "crawler.db")
    now = datetime.now(timezone.utc).isoformat(timespec="seconds")
    derivative_tables = [
        "llm_excluded_reviews",
        "ner_filter_events",
        "tag_cache",
        "tag_results",
        "tagging_queue",
    ]
    with db_utils.connect(db_path) as conn:
        item_rows = conn.execute("SELECT id FROM crawled_items").fetchall()
        for table in derivative_tables:
            conn.execute(f"DELETE FROM {table}")
        for row in item_rows:
            conn.execute(
                """
                INSERT INTO tagging_queue (item_id, status, attempts, queued_at, last_error)
                VALUES (?, 'pending', 0, ?, '')
                """,
                (row["id"], now),
            )
        conn.commit()
        total = len(item_rows)

    invalidate_summary_cache()
    progress_dir = os.path.join(filter_dir, "output")
    os.makedirs(progress_dir, exist_ok=True)
    with open(os.path.join(progress_dir, "tagging_progress.json"), "w", encoding="utf-8") as f:
        json.dump({
            "status": "running" if total else "complete",
            "total_count": total,
            "completed_count": 0,
            "in_progress_count": total,
            "excluded_count": 0,
            "failed_count": 0,
            "category_counts": {},
            "queue": {
                "pending": total,
                "tagging": 0,
                "tagged": 0,
                "excluded": 0,
                "failed": 0,
            },
            "model_version": "",
            "generated_at": now,
        }, f, ensure_ascii=False, indent=2)

    append_log(f"[{time_str()}] [NLP] reset tagging work; queued {total} crawled items for rework.")
    return total

def start_tagging_rework_worker():
    script_path = os.path.join(filter_dir, "run_tagging_rework.py")
    if not os.path.exists(script_path):
        raise RuntimeError("run_tagging_rework.py 파일을 찾을 수 없습니다.")

    creationflags = subprocess.CREATE_NO_WINDOW if sys.platform == "win32" else 0
    log_file = open(log_path, "a", encoding="utf-8", errors="replace")
    append_log(f"[{time_str()}] [NLP] background tagging rework started.")
    subprocess.Popen(
        [sys.executable, "-u", script_path],
        stdout=log_file,
        stderr=log_file,
        cwd=filter_dir,
        creationflags=creationflags,
    )

def tag_test_items(item_ids):
    if not item_ids:
        return 0
    if filter_dir not in sys.path:
        sys.path.insert(0, filter_dir)
    import realtime_tagger
    import sqlite3
    conn = sqlite3.connect(os.path.join(crawling_dir, "crawler.db"))
    conn.row_factory = sqlite3.Row
    realtime_tagger.init_db(conn)
    placeholders = ",".join("?" for _ in item_ids)
    rows = conn.execute(
        f"""
        SELECT c.id AS item_id, c.*
          FROM crawled_items c
         WHERE c.id IN ({placeholders})
         ORDER BY COALESCE(c.published_at, c.created_at) DESC
        """,
        item_ids,
    ).fetchall()
    tagged = 0
    try:
        model_ctx = realtime_tagger.load_required_model_context(conn)
        items = [dict(row) for row in rows]
        results = realtime_tagger.infer_batch(model_ctx, items)
        for item, result in zip(items, results):
            if getattr(realtime_tagger, "mining_engine", None) is not None:
                result = realtime_tagger.mining_engine.apply_ner_gate(conn, item, result)
                result = realtime_tagger.mining_engine.apply_qwen_category_review(conn, item, result)
            realtime_tagger.save_result(conn, item, result)
            tagged += 1
        conn.commit()
        realtime_tagger.write_progress(conn, model_ctx=model_ctx)
    finally:
        conn.close()
    return tagged

def run_analysis_only():
    script = os.path.join(filter_dir, "build_analysis.py")
    creationflags = subprocess.CREATE_NO_WINDOW if sys.platform == "win32" else 0
    subprocess.run([sys.executable, script], cwd=filter_dir, check=True, creationflags=creationflags)

def export_data_js_current_mode(limit=160):
    script = os.path.join(crawling_dir, "crawler.py")
    creationflags = subprocess.CREATE_NO_WINDOW if sys.platform == "win32" else 0
    subprocess.run([sys.executable, script, "--market-only", "--limit", str(limit)], cwd=crawling_dir, check=False, creationflags=creationflags)

def parse_dataset_payload(payload):
    if isinstance(payload, dict):
        items = payload.get("items")
        name = payload.get("name") or payload.get("scenario") or "uploaded"
    else:
        items = payload
        name = "uploaded"
    return str(name), normalize_test_items(items)

def apply_test_dataset_items(items, dataset_name):
    if crawling_dir not in sys.path:
        sys.path.insert(0, crawling_dir)
    import db_utils

    db_utils.init_db()
    with db_utils.connect() as conn:
        delete_test_items(conn)
        conn.commit()

    item_ids = []
    for item in items:
        result = db_utils.upsert_crawled_item(item)
        item_ids.append(result["id"])

    tagged = tag_test_items(item_ids)
    mode = save_test_mode(True, dataset_name, len(item_ids))
    run_analysis_only()
    export_data_js_current_mode()
    with open(log_path, "a", encoding="utf-8") as f:
        f.write(f"[{time_str()}] [TEST] applied {dataset_name} dataset ({len(item_ids)} items, tagged={tagged}).\n")
    return mode, len(item_ids), tagged

def set_platform_status(platform, status_type, error_msg=None):
    status_data = {"platforms": {}}
    if os.path.exists(crawler_status_path):
        try:
            with open(crawler_status_path, 'r', encoding='utf-8') as f:
                status_data = json.load(f)
        except:
            pass
    if "platforms" not in status_data:
        status_data["platforms"] = {}
    status_data["platforms"][platform] = {
        "status": status_type,
        "last_run": None,
        "error": error_msg
    }
    status_data["last_updated"] = time_str()
    with open(crawler_status_path, 'w', encoding='utf-8') as f:
        json.dump(status_data, f, ensure_ascii=False, indent=2)

def apply_scheduler_state_to_crawler_status(crawler_status, config, running):
    status_data = crawler_status if isinstance(crawler_status, dict) else {}
    platforms = status_data.setdefault("platforms", {})
    loop_data = status_data.setdefault("loop", {})
    loop_data["running"] = bool(running)
    if not running:
        loop_data["crawl_interval_seconds"] = CRAWL_INTERVAL_SECONDS
        loop_data["crawl_base_interval_seconds"] = CRAWL_INTERVAL_SECONDS
        loop_data["crawl_jitter_ratio"] = CRAWL_JITTER_RATIO
        loop_data["active_crawl_interval_seconds"] = None
        loop_data["next_crawl_at"] = None
        loop_data["seconds_until_next_crawl"] = 0
        loop_data["platform_next"] = {}
    else:
        loop_data.setdefault("crawl_interval_seconds", CRAWL_INTERVAL_SECONDS)
        loop_data.setdefault("crawl_base_interval_seconds", CRAWL_INTERVAL_SECONDS)
        loop_data.setdefault("crawl_jitter_ratio", CRAWL_JITTER_RATIO)

    for platform in collector_keys:
        enabled = config["enabled"].get(platform, True)
        current = platforms.get(platform) if isinstance(platforms.get(platform), dict) else {}

        if not enabled:
            current.update({
                "status": "disabled",
                "error": "Disabled by user setting"
            })
        elif not running:
            current.update({
                "status": "pending",
                "error": "Crawler loop is stopped"
            })

        current.setdefault("last_run", None)
        current.setdefault("count", 0)
        platforms[platform] = current

    status_data["last_updated"] = time_str()
    return status_data

def save_crawler_status(status_data):
    with open(crawler_status_path, 'w', encoding='utf-8') as f:
        json.dump(status_data, f, ensure_ascii=False, indent=2)

def mask_secret(value):
    if not value:
        return ""
    if len(value) <= 10:
        return value[:2] + "*" * max(0, len(value) - 4) + value[-2:]
    return value[:6] + "*" * (len(value) - 10) + value[-4:]

def validate_openai_api_key(api_key, allow_empty=False):
    token = str(api_key or "").strip()
    if not token:
        if allow_empty:
            return ""
        raise ValueError("OpenAI API 키가 필요합니다.")
    if any(ch.isspace() for ch in token):
        raise ValueError("OpenAI API 키에 공백/줄바꿈이 포함되어 있습니다. 키를 다시 복사해 주세요.")
    if not token.startswith("sk-"):
        raise ValueError(
            f"OpenAI API 키 형식이 아닙니다. OpenAI API 키는 보통 sk- 또는 sk-proj-로 시작합니다. 현재 입력: {mask_secret(token)}"
        )
    return token

def append_log(message):
    try:
        with open(log_path, 'a', encoding='utf-8', errors='replace') as f:
            f.write(message.rstrip() + "\n")
    except Exception as e:
        print(f"Error writing log: {e}")

def invalidate_summary_cache():
    try:
        if os.path.exists(summary_cache_path):
            os.remove(summary_cache_path)
            return True
    except Exception as e:
        append_log(f"[{time_str()}] [NLP] Summary cache clear failed: {e}")
    return False

def trigger_analysis_rebuild(reason):
    script_path = os.path.join(filter_dir, "build_analysis.py")
    if not os.path.exists(script_path):
        append_log(f"[{time_str()}] [NLP] build_analysis.py not found; rebuild skipped.")
        return False

    creationflags = 0
    if sys.platform == 'win32':
        creationflags = subprocess.CREATE_NO_WINDOW

    append_log(f"[{time_str()}] [NLP] Analysis rebuild started: {reason}")
    log_file = open(log_path, 'a', encoding='utf-8', errors='replace')
    subprocess.Popen(
        [sys.executable, "-u", script_path],
        stdout=log_file,
        stderr=log_file,
        cwd=filter_dir,
        creationflags=creationflags
    )
    return True

def load_won_config(include_key=False):
    config = {
        "model": default_won_model,
        "models": supported_won_models,
        "configured": False,
        "enabled": False,
        "reasoning_effort": default_llm_reasoning_effort,
        "api_url": "https://api.openai.com/v1/responses",
        "api_key_masked": "OpenAI API 키 미설정"
    }
    if os.path.exists(won_config_path):
        try:
            with open(won_config_path, 'r', encoding='utf-8') as f:
                saved = json.load(f)
            api_key = saved.get("openai_api_key", "") or saved.get("api_key", "")
            model = saved.get("model_id", default_won_model) or default_won_model
            api_url = saved.get("api_url", "https://api.openai.com/v1/responses")
            reasoning_effort = str(saved.get("reasoning_effort", default_llm_reasoning_effort)).lower()
            if model not in supported_won_models:
                model = default_won_model
            if reasoning_effort not in {"none", "low", "medium", "high", "xhigh"}:
                reasoning_effort = default_llm_reasoning_effort
            env_key = os.environ.get("OPENAI_API_KEY", "").strip()
            if env_key:
                api_key = env_key
            key_error = ""
            if api_key:
                try:
                    api_key = validate_openai_api_key(api_key)
                except ValueError as exc:
                    key_error = str(exc)
                    api_key = ""
            config.update({
                "model": model,
                "api_url": api_url,
                "reasoning_effort": reasoning_effort,
                "configured": bool(api_key),
                "enabled": bool(saved.get("enabled", bool(api_key)) and api_key),
                "api_key_masked": mask_secret(api_key) if api_key else "OpenAI API 키 미설정"
            })
            if key_error:
                config["key_error"] = key_error
                config["api_key_masked"] = "OpenAI API 키 형식 오류"
            if include_key:
                config["api_key"] = api_key
        except Exception as e:
            print(f"Error loading WON config: {e}")
    else:
        env_key = os.environ.get("OPENAI_API_KEY", "").strip()
        if env_key:
            key_error = ""
            try:
                env_key = validate_openai_api_key(env_key)
            except ValueError as exc:
                key_error = str(exc)
                env_key = ""
            config.update({
                "configured": bool(env_key),
                "enabled": bool(env_key),
                "api_key_masked": mask_secret(env_key) if env_key else "OpenAI API 키 형식 오류",
            })
            if key_error:
                config["key_error"] = key_error
            if include_key:
                config["api_key"] = env_key
    return config

def default_llm_usage():
    return {
        "available": False,
        "total_calls": 0,
        "estimated_calls": 0,
        "provider_reported_calls": 0,
        "total_prompt_tokens": 0,
        "total_completion_tokens": 0,
        "total_tokens": 0,
        "by_model": {},
        "last_call": None,
    }

def load_llm_usage():
    db_path = os.path.join(crawling_dir, "crawler.db")
    usage = default_llm_usage()
    if not os.path.exists(db_path):
        return usage
    try:
        with sqlite3.connect(db_path) as conn:
            conn.row_factory = sqlite3.Row
            row = conn.execute(
                "SELECT value FROM engine_state WHERE key = ?",
                ("llm_usage_summary",),
            ).fetchone()
        if not row:
            return usage
        saved = json.loads(row["value"] or "{}")
        if isinstance(saved, dict):
            usage.update(saved)
            usage["available"] = True
    except Exception as e:
        usage["error"] = str(e)
    return usage

def save_won_config(api_key="", model=None, enabled=True, api_url=""):
    existing = {}
    if os.path.exists(won_config_path):
        try:
            with open(won_config_path, 'r', encoding='utf-8') as f:
                existing = json.load(f)
        except Exception:
            existing = {}

    model = model or default_won_model
    if model not in supported_won_models:
        raise ValueError("지원하지 않는 OpenAI 모델입니다.")
    saved_token = existing.get("openai_api_key", "") or existing.get("api_key", "")
    provided_token = api_key.strip()
    next_token = provided_token if provided_token else saved_token
    if next_token:
        try:
            next_token = validate_openai_api_key(next_token)
        except ValueError:
            if provided_token:
                raise
            next_token = ""
            enabled = False
    next_api_url = api_url.strip() if api_url.strip() else existing.get("api_url", "https://api.openai.com/v1/responses")
    data = {
        "model_id": model,
        "openai_api_key": next_token,
        "api_url": next_api_url,
        "reasoning_effort": default_llm_reasoning_effort,
        "enabled": bool(enabled)
    }
    with open(won_config_path, 'w', encoding='utf-8') as f:
        json.dump(data, f, ensure_ascii=False, indent=2)
    return load_won_config()

def call_won_test(api_key, model, api_url=""):
    if model not in supported_won_models:
        raise ValueError("지원하지 않는 OpenAI 모델입니다.")
    token = api_key or os.environ.get("OPENAI_API_KEY", "").strip()
    token = validate_openai_api_key(token)
    endpoint = api_url or "https://api.openai.com/v1/responses"
    response = requests.post(
        endpoint,
        headers={"Authorization": f"Bearer {token}", "Content-Type": "application/json"},
        json={
            "model": model,
            "input": [
                {"role": "developer", "content": "Return strict JSON only."},
                {"role": "user", "content": "{\"status\":\"ok\"} 형태의 짧은 JSON만 출력하세요."},
            ],
            "reasoning": {"effort": default_llm_reasoning_effort},
            "max_output_tokens": 64,
        },
        timeout=60,
    )
    if response.status_code >= 400:
        try:
            payload = response.json()
            error = payload.get("error") or {}
            code = error.get("code")
            message = error.get("message") or response.text[:300]
        except Exception:
            code = ""
            message = response.text[:300]
        if response.status_code == 401 or code == "invalid_api_key":
            raise RuntimeError(
                f"OpenAI API 키가 유효하지 않습니다. 저장된 키를 삭제하고 platform.openai.com에서 새 sk- 키를 발급해 입력하세요. ({mask_secret(token)})"
            )
        raise RuntimeError(f"OpenAI API 호출 실패: HTTP {response.status_code} {message}")
    return f"{model} / reasoning={default_llm_reasoning_effort} Responses API 호출 확인 완료"

# Helper to check if a process PID is currently active
def is_pid_running(pid):
    if pid is None:
        return False
    try:
        pid = int(pid)
    except (TypeError, ValueError):
        return False
    if pid <= 0:
        return False
    if sys.platform == "win32":
        try:
            import ctypes
            PROCESS_QUERY_LIMITED_INFORMATION = 0x1000
            handle = ctypes.windll.kernel32.OpenProcess(PROCESS_QUERY_LIMITED_INFORMATION, False, int(pid))
            if not handle:
                return False
            exit_code = ctypes.c_ulong()
            ok = ctypes.windll.kernel32.GetExitCodeProcess(handle, ctypes.byref(exit_code))
            ctypes.windll.kernel32.CloseHandle(handle)
            return bool(ok) and exit_code.value == 259
        except Exception:
            return False
    try:
        os.kill(pid, 0)
        return True
    except OSError:
        return False

# Main Page
@app.route('/')
def index():
    return render_template('gui.html')

# Static Dashboard & Data Routing
@app.route('/code.html')
def dashboard():
    return send_from_directory(crolling_dir, 'code.html')

@app.route('/data.js')
def data_js():
    return send_from_directory(crolling_dir, 'data.js')

@app.route('/analysis.js')
def analysis_js():
    return send_from_directory(crolling_dir, 'analysis.js')

@app.route('/api/tagging/progress')
def tagging_progress():
    progress_path = os.path.join(filter_dir, "output", "tagging_progress.json")
    if not os.path.exists(progress_path):
        return jsonify({
            "total_count": 0,
            "in_progress_count": 0,
            "completed_count": 0,
            "excluded_count": 0,
            "category_counts": {},
            "status": "idle"
        })
    try:
        with open(progress_path, 'r', encoding='utf-8') as f:
            progress = json.load(f)
        progress = reconcile_progress_with_db(progress)
        training_path = os.path.join(filter_dir, "output", "training_status.json")
        if os.path.exists(training_path):
            try:
                with open(training_path, "r", encoding="utf-8") as tf:
                    training = json.load(tf)
                progress["training_status"] = training.get("status")
                progress["last_training_at"] = training.get("finished_at") or training.get("started_at")
                progress["training_sample_count"] = training.get("sample_count")
            except Exception:
                pass
        return jsonify(progress)
    except Exception as e:
        return jsonify({"status": "error", "message": str(e)})


def _safe_json_list(value):
    if not value:
        return []
    try:
        parsed = json.loads(value)
        return parsed if isinstance(parsed, list) else []
    except Exception:
        return []


def _tagging_audit_item(row):
    tags = _safe_json_list(row.get("tags_json"))
    primary_tag = row.get("primary_tag") or ""
    if primary_tag and not tags:
        tags = [{"tag": primary_tag, "score": row.get("relevance_score") or 0, "hits": 0}]
    is_excluded = bool(row.get("is_excluded") or row.get("queue_status") == "excluded")
    content = row.get("content") or ""
    exclude_reason = row.get("exclude_reason") or row.get("queue_error") or ""
    if is_excluded and not exclude_reason:
        exclude_reason = "사유 미기록"
    return {
        "item_id": row.get("item_id"),
        "content_hash": row.get("content_hash") or "",
        "source": row.get("source") or "",
        "source_group": row.get("source_group") or "",
        "country": row.get("country") or "",
        "language": row.get("language") or "",
        "title": row.get("title") or "",
        "url": row.get("url") or "",
        "published_at": row.get("published_at") or "",
        "crawled_at": row.get("created_at") or "",
        "tagged_at": row.get("tagged_at") or "",
        "queue_status": row.get("queue_status") or "not_queued",
        "crawl_status": row.get("crawl_status") or "",
        "crawl_quality_score": float(row.get("crawl_quality_score") or 0),
        "content_len": len(content),
        "snippet": " ".join(content.split())[:420],
        "model_version": row.get("model_version") or "",
        "primary_tag": primary_tag,
        "tags": tags,
        "sentiment_label": row.get("sentiment_label") or "",
        "sentiment_score": float(row.get("sentiment_score") or 0),
        "relevance_score": float(row.get("relevance_score") or 0),
        "confidence": float(row.get("confidence") or 0),
        "impact_type": row.get("impact_type") or "",
        "excluded": is_excluded,
        "exclude_reason": exclude_reason,
    }


@app.route("/api/tagging/audit")
def api_tagging_audit():
    try:
        if crawling_dir not in sys.path:
            sys.path.insert(0, crawling_dir)
        import db_utils

        status_filter = (request.args.get("status") or "all").strip().lower()
        tag_filter = (request.args.get("tag") or "all").strip()
        q = (request.args.get("q") or "").strip()
        limit = min(max(int(request.args.get("limit") or 120), 20), 300)
        offset = max(int(request.args.get("offset") or 0), 0)
        test_mode = load_test_mode()
        test_only = bool(test_mode.get("active"))
        lookback_where = "" if test_only else crawler_lookback_where_sql("c")
        active_model_version = (request.args.get("model_version") or "").strip()
        if not active_model_version:
            progress_path = os.path.join(filter_dir, "output", "tagging_progress.json")
            if os.path.exists(progress_path):
                try:
                    with open(progress_path, "r", encoding="utf-8") as pf:
                        active_model_version = str((json.load(pf) or {}).get("model_version") or "")
                except Exception:
                    active_model_version = ""

        base_where = [
            "((? = 1 AND c.source_group = 'test') OR (? = 0 AND COALESCE(c.source_group, '') <> 'test'))"
        ]
        params = [1 if test_only else 0, 1 if test_only else 0]

        if status_filter == "tagged":
            base_where.append("r.item_id IS NOT NULL AND COALESCE(r.excluded, 0) = 0")
        elif status_filter == "excluded":
            base_where.append("(COALESCE(r.excluded, 0) = 1 OR q.status = 'excluded')")
        elif status_filter == "pending":
            base_where.append("r.item_id IS NULL AND COALESCE(q.status, 'not_queued') IN ('pending', 'tagging', 'failed', 'not_queued')")

        if tag_filter and tag_filter.lower() != "all":
            base_where.append("r.primary_tag = ?")
            params.append(tag_filter)

        if q:
            base_where.append("(c.title LIKE ? OR c.content LIKE ? OR c.source LIKE ? OR c.url LIKE ?)")
            like = f"%{q}%"
            params.extend([like, like, like, like])

        where_sql = " AND ".join(base_where) + lookback_where
        db_utils.init_db()
        with db_utils.connect() as conn:
            rows = conn.execute(
                f"""
                SELECT c.id AS item_id, c.content_hash, c.source, c.source_group, c.country, c.language,
                       c.title, c.content, c.url, c.published_at, c.created_at,
                       c.crawl_status, c.crawl_quality_score,
                       q.status AS queue_status, q.last_error AS queue_error,
                       r.model_version, r.primary_tag, r.tags_json, r.sentiment_label,
                       r.sentiment_score, r.relevance_score, r.confidence, r.impact_type,
                       r.excluded AS is_excluded, r.exclude_reason, r.tagged_at
                  FROM crawled_items c
                  LEFT JOIN tagging_queue q ON q.item_id = c.id
                  LEFT JOIN tag_results r ON r.item_id = c.id
                   AND (? = '' OR r.model_version = ?)
                 WHERE {where_sql}
                 ORDER BY COALESCE(r.tagged_at, c.published_at, c.created_at) DESC
                 LIMIT ? OFFSET ?
                """,
                [active_model_version, active_model_version, *params, limit, offset],
            ).fetchall()
            total = conn.execute(
                f"""
                SELECT COUNT(*) AS count
                  FROM crawled_items c
                  LEFT JOIN tagging_queue q ON q.item_id = c.id
                  LEFT JOIN tag_results r ON r.item_id = c.id
                   AND (? = '' OR r.model_version = ?)
                 WHERE {where_sql}
                """,
                [active_model_version, active_model_version, *params],
            ).fetchone()["count"]
            summary = conn.execute(
                f"""
                SELECT
                    COUNT(*) AS total_items,
                    SUM(CASE WHEN r.item_id IS NOT NULL AND COALESCE(r.excluded, 0) = 0 THEN 1 ELSE 0 END) AS tagged_items,
                    SUM(CASE WHEN COALESCE(r.excluded, 0) = 1 OR q.status = 'excluded' THEN 1 ELSE 0 END) AS excluded_items,
                    SUM(CASE WHEN r.item_id IS NULL AND COALESCE(q.status, 'not_queued') IN ('pending', 'tagging', 'failed', 'not_queued') THEN 1 ELSE 0 END) AS pending_items
                  FROM crawled_items c
                  LEFT JOIN tagging_queue q ON q.item_id = c.id
                  LEFT JOIN tag_results r ON r.item_id = c.id
                   AND (? = '' OR r.model_version = ?)
                 WHERE ((? = 1 AND c.source_group = 'test') OR (? = 0 AND COALESCE(c.source_group, '') <> 'test'))
                   {lookback_where}
                """,
                (active_model_version, active_model_version, 1 if test_only else 0, 1 if test_only else 0),
            ).fetchone()
            tag_counts = [
                dict(row)
                for row in conn.execute(
                    f"""
                    SELECT r.primary_tag AS tag, COUNT(*) AS count
                      FROM tag_results r
                      JOIN crawled_items c ON c.id = r.item_id
                     WHERE r.primary_tag <> ''
                       AND COALESCE(r.excluded, 0) = 0
                       AND (? = '' OR r.model_version = ?)
                       AND ((? = 1 AND c.source_group = 'test') OR (? = 0 AND COALESCE(c.source_group, '') <> 'test'))
                       {lookback_where}
                     GROUP BY r.primary_tag
                     ORDER BY count DESC, tag ASC
                    """,
                    (active_model_version, active_model_version, 1 if test_only else 0, 1 if test_only else 0),
                ).fetchall()
            ]
            excluded_reasons = [
                {
                    "reason": row["reason"] or "unspecified",
                    "count": row["count"],
                }
                for row in conn.execute(
                    f"""
                    SELECT COALESCE(NULLIF(r.exclude_reason, ''), NULLIF(q.last_error, ''), 'unspecified') AS reason,
                           COUNT(*) AS count
                      FROM crawled_items c
                      LEFT JOIN tagging_queue q ON q.item_id = c.id
                      LEFT JOIN tag_results r ON r.item_id = c.id
                       AND (? = '' OR r.model_version = ?)
                     WHERE (COALESCE(r.excluded, 0) = 1 OR q.status = 'excluded')
                       AND ((? = 1 AND c.source_group = 'test') OR (? = 0 AND COALESCE(c.source_group, '') <> 'test'))
                       {lookback_where}
                     GROUP BY reason
                     ORDER BY count DESC
                    """,
                    (active_model_version, active_model_version, 1 if test_only else 0, 1 if test_only else 0),
                ).fetchall()
            ]

        return jsonify({
            "status": "success",
            "filters": {
                "status": status_filter,
                "tag": tag_filter,
                "q": q,
                "limit": limit,
                "offset": offset,
                "model_version": active_model_version,
            },
            "total": int(total or 0),
            "summary": {
                "total_items": int(summary["total_items"] or 0),
                "tagged_items": int(summary["tagged_items"] or 0),
                "excluded_items": int(summary["excluded_items"] or 0),
                "pending_items": int(summary["pending_items"] or 0),
            },
            "tag_counts": tag_counts,
            "excluded_reasons": excluded_reasons,
            "items": [_tagging_audit_item(dict(row)) for row in rows],
        })
    except Exception as e:
        return jsonify({"status": "error", "message": f"태깅 감사 데이터 조회 실패: {str(e)}"}), 500


@app.route("/api/dedup/audit")
def api_dedup_audit():
    try:
        if crawling_dir not in sys.path:
            sys.path.insert(0, crawling_dir)
        import db_utils

        limit = min(max(int(request.args.get("limit") or 80), 20), 200)
        db_utils.init_db()
        with db_utils.connect() as conn:
            summary = conn.execute(
                """
                SELECT
                    COUNT(*) AS candidate_pairs,
                    SUM(CASE WHEN audit_status = 'audited' THEN 1 ELSE 0 END) AS audited_pairs,
                    SUM(CASE WHEN COALESCE(gpt_is_duplicate, 0) = 1 THEN 1 ELSE 0 END) AS duplicate_pairs,
                    SUM(CASE WHEN within_group = 1 THEN 1 ELSE 0 END) AS grouped_pairs,
                    SUM(CASE WHEN audit_status LIKE 'blocked%' OR audit_status = 'audit_failed' THEN 1 ELSE 0 END) AS blocked_pairs
                  FROM dedup_candidate_pairs
                """
            ).fetchone()
            group_count = conn.execute("SELECT COUNT(*) AS count FROM dedup_groups").fetchone()["count"]
            member_count = conn.execute("SELECT COUNT(*) AS count FROM dedup_group_members").fetchone()["count"]
            rows = conn.execute(
                """
                SELECT p.*,
                       l.title AS left_title, l.source AS left_source, l.source_group AS left_source_group,
                       l.url AS left_url, l.published_at AS left_published_at,
                       r.title AS right_title, r.source AS right_source, r.source_group AS right_source_group,
                       r.url AS right_url, r.published_at AS right_published_at
                  FROM dedup_candidate_pairs p
                  JOIN crawled_items l ON l.id = p.left_item_id
                  JOIN crawled_items r ON r.id = p.right_item_id
                 ORDER BY COALESCE(p.gpt_is_duplicate, 0) DESC,
                          p.within_group DESC,
                          p.composite_score DESC,
                          p.similarity DESC
                 LIMIT ?
                """,
                (limit,),
            ).fetchall()

        return jsonify({
            "status": "success",
            "summary": {
                "candidate_pairs": int(summary["candidate_pairs"] or 0),
                "audited_pairs": int(summary["audited_pairs"] or 0),
                "duplicate_pairs": int(summary["duplicate_pairs"] or 0),
                "grouped_pairs": int(summary["grouped_pairs"] or 0),
                "blocked_pairs": int(summary["blocked_pairs"] or 0),
                "dedup_groups": int(group_count or 0),
                "dedup_members": int(member_count or 0),
            },
            "items": [
                {
                    "left_item_id": row["left_item_id"],
                    "right_item_id": row["right_item_id"],
                    "model_version": row["model_version"],
                    "similarity": float(row["similarity"] or 0),
                    "bge_similarity": float(row["bge_similarity"] or 0),
                    "lsa_similarity": float(row["lsa_similarity"] or 0),
                    "title_similarity": float(row["title_similarity"] or 0),
                    "url_similarity": float(row["url_similarity"] or 0),
                    "source_match": bool(row["source_match"]),
                    "time_delta_hours": float(row["time_delta_hours"] or 0),
                    "composite_score": float(row["composite_score"] or 0),
                    "audit_status": row["audit_status"] or "",
                    "audit_model": row["audit_model"] or "",
                    "gpt_is_duplicate": None if row["gpt_is_duplicate"] is None else bool(row["gpt_is_duplicate"]),
                    "gpt_confidence": None if row["gpt_confidence"] is None else float(row["gpt_confidence"]),
                    "gpt_rationale": row["gpt_rationale"] or "",
                    "audited_at": row["audited_at"] or "",
                    "within_group": bool(row["within_group"]),
                    "left_title": row["left_title"] or "",
                    "left_source": row["left_source"] or "",
                    "left_source_group": row["left_source_group"] or "",
                    "left_url": row["left_url"] or "",
                    "left_published_at": row["left_published_at"] or "",
                    "right_title": row["right_title"] or "",
                    "right_source": row["right_source"] or "",
                    "right_source_group": row["right_source_group"] or "",
                    "right_url": row["right_url"] or "",
                    "right_published_at": row["right_published_at"] or "",
                }
                for row in rows
            ],
        })
    except Exception as e:
        return jsonify({"status": "error", "message": f"중복 후보 감사 데이터 조회 실패: {str(e)}"}), 500


@app.route("/api/tagging/feedback", methods=["POST"])
def api_tagging_feedback():
    """Store reviewed labels for periodic validated fine-tuning."""
    try:
        payload = request.get_json(force=True) or {}
        if crawling_dir not in sys.path:
            sys.path.insert(0, crawling_dir)
        import db_utils

        item_id = payload.get("item_id")
        content_hash = payload.get("content_hash")
        if not item_id or not content_hash:
            return jsonify({"success": False, "error": "item_id and content_hash are required"}), 400

        feedback_id = db_utils.add_feedback(
            item_id=item_id,
            content_hash=content_hash,
            original_tag=payload.get("original_tag"),
            original_sentiment=payload.get("original_sentiment"),
            corrected_tag=payload.get("corrected_tag"),
            corrected_sentiment=payload.get("corrected_sentiment"),
            approved=bool(payload.get("approved", True)),
            reviewer=payload.get("reviewer", "dashboard"),
            notes=payload.get("notes"),
        )
        return jsonify({"success": True, "feedback_id": feedback_id})
    except Exception as e:
        return jsonify({"success": False, "error": str(e)}), 500


@app.route("/api/test_mode", methods=["GET"])
def api_test_mode_status():
    mode = load_test_mode()
    return jsonify({
        "active": bool(mode.get("active")),
        "scenario": mode.get("scenario", ""),
        "item_count": int(mode.get("item_count") or 0),
        "updated_at": mode.get("updated_at", ""),
        "scenarios": {
            key: {
                "label": value["label"],
                "item_count": value["items_per_category"] * len(test_category_terms),
                "summary": value["summary"],
            }
            for key, value in test_scenarios.items()
        },
        "dataset_dir": test_dataset_dir,
        "available_files": list_test_dataset_files(),
    })


@app.route("/api/test_mode/apply", methods=["POST"])
def api_test_mode_apply():
    try:
        payload = request.get_json(force=True) or {}
        scenario = str(payload.get("scenario") or "").strip().lower()
        custom_items = payload.get("items")
        if custom_items:
            scenario_name, items = parse_dataset_payload({"name": payload.get("name") or "custom", "items": custom_items})
        else:
            items = build_test_items(scenario)
            scenario_name = scenario

        mode, inserted, tagged = apply_test_dataset_items(items, scenario_name)
        return jsonify({
            "success": True,
            "mode": mode,
            "inserted": inserted,
            "tagged": tagged,
            "message": f"{scenario_name} 테스트 데이터 {inserted}개가 적용되었습니다.",
        })
    except Exception as e:
        return jsonify({"success": False, "error": str(e)}), 500


@app.route("/api/test_mode/upload", methods=["POST"])
def api_test_mode_upload():
    try:
        uploaded = request.files.get("dataset")
        if not uploaded or not uploaded.filename:
            return jsonify({"success": False, "error": "업로드할 JSON 데이터셋 파일을 선택해 주세요."}), 400
        filename = secure_filename(uploaded.filename)
        if not filename.lower().endswith(".json"):
            return jsonify({"success": False, "error": "JSON 파일만 업로드할 수 있습니다."}), 400

        raw = uploaded.read()
        try:
            payload = json.loads(raw.decode("utf-8-sig"))
        except Exception as exc:
            return jsonify({"success": False, "error": f"JSON 파싱 실패: {exc}"}), 400

        dataset_name, items = parse_dataset_payload(payload)
        if dataset_name == "uploaded":
            dataset_name = os.path.splitext(filename)[0]
        os.makedirs(test_dataset_dir, exist_ok=True)
        saved_path = os.path.join(test_dataset_dir, filename)
        with open(saved_path, "wb") as f:
            f.write(raw)

        mode, inserted, tagged = apply_test_dataset_items(items, dataset_name)
        return jsonify({
            "success": True,
            "mode": mode,
            "inserted": inserted,
            "tagged": tagged,
            "saved_file": filename,
            "available_files": list_test_dataset_files(),
            "message": f"{filename} 기반 테스트 데이터 {inserted}개 태깅/AI요약을 완료했습니다.",
        })
    except Exception as e:
        return jsonify({"success": False, "error": str(e)}), 500


@app.route("/api/test_mode/disable", methods=["POST"])
def api_test_mode_disable():
    try:
        if crawling_dir not in sys.path:
            sys.path.insert(0, crawling_dir)
        import db_utils
        db_utils.init_db()
        with db_utils.connect() as conn:
            deleted = delete_test_items(conn)
            conn.commit()
        mode = save_test_mode(False, "", 0)
        run_analysis_only()
        export_data_js_current_mode()
        with open(log_path, "a", encoding="utf-8") as f:
            f.write(f"[{time_str()}] [TEST] disabled test mode; removed {deleted} test items and restored live analysis view.\n")
        return jsonify({"success": True, "mode": mode, "deleted": deleted, "message": "테스트 모드를 해제했습니다."})
    except Exception as e:
        return jsonify({"success": False, "error": str(e)}), 500


@app.route("/api/stock_quotes")
def api_stock_quotes():
    symbols = [s.strip().upper() for s in request.args.get("symbols", "").split(",") if s.strip()]
    symbols = list(dict.fromkeys(symbols))[:80]
    if not symbols:
        return jsonify({"quotes": {}, "source": "Yahoo Finance chart", "errors": []})

    quotes = {}
    errors = []
    from concurrent.futures import ThreadPoolExecutor, as_completed

    naver_index_symbols = {"^KS11": "KOSPI", "^KQ11": "KOSDAQ"}

    def fetch_quote(symbol):
        headers = {"User-Agent": "Mozilla/5.0 (compatible; PolitiMarket/1.0)"}
        if symbol in naver_index_symbols:
            quote, error = fetch_naver_index_quote(naver_index_symbols[symbol])
            if quote:
                return symbol, quote, None
            if error:
                errors.append({"symbol": symbol, "source": "Naver Finance", "error": error})
        try:
            res = requests.get(
                f"https://query1.finance.yahoo.com/v8/finance/chart/{symbol}",
                params={"range": "5d", "interval": "1d"},
                headers=headers,
                timeout=8,
            )
            res.raise_for_status()
            payload = res.json()
            result = (payload.get("chart", {}).get("result") or [None])[0]
            if not result:
                raise RuntimeError("empty quote result")
            meta = result.get("meta") or {}
            price = meta.get("regularMarketPrice")
            timestamps = result.get("timestamp") or []
            closes = ((result.get("indicators") or {}).get("quote") or [{}])[0].get("close") or []
            valid_closes = [value for value in closes if isinstance(value, (int, float))]
            previous = (
                meta.get("regularMarketPreviousClose")
                or meta.get("previousClose")
                or meta.get("chartPreviousClose")
                or (valid_closes[-2] if len(valid_closes) >= 2 else None)
            )
            if price is None and valid_closes:
                price = valid_closes[-1]
            if price is None:
                raise RuntimeError("missing price")
            change = 0.0
            if previous:
                change = ((float(price) - float(previous)) / float(previous)) * 100
            return symbol, {
                "price": round(float(price), 2),
                "change": round(float(change), 2),
                "currency": meta.get("currency") or ("KRW" if symbol.endswith((".KS", ".KQ")) else "USD"),
                "exchange": meta.get("exchangeName") or "",
                "updated_at": meta.get("regularMarketTime") or (timestamps[-1] if timestamps else None),
            }, None
        except Exception as exc:
            return symbol, None, str(exc)

    with ThreadPoolExecutor(max_workers=10) as executor:
        futures = [executor.submit(fetch_quote, symbol) for symbol in symbols]
        for future in as_completed(futures):
            symbol, quote, error = future.result()
            if quote:
                quotes[symbol] = quote
            else:
                errors.append({"symbol": symbol, "error": error})
    return jsonify({"quotes": quotes, "source": "Yahoo Finance chart", "errors": errors})


def parse_market_number(value, default=None):
    try:
        cleaned = str(value or "").replace(",", "").replace("%", "").strip()
        if not cleaned:
            return default
        return float(cleaned)
    except Exception:
        return default


def yahoo_points(symbol, range_value, interval):
    headers = {"User-Agent": "Mozilla/5.0 (compatible; PolitiMarket/1.0)"}
    res = requests.get(
        f"https://query1.finance.yahoo.com/v8/finance/chart/{symbol}",
        params={"range": range_value, "interval": interval},
        headers=headers,
        timeout=10,
    )
    res.raise_for_status()
    payload = res.json()
    result = (payload.get("chart", {}).get("result") or [None])[0]
    if not result:
        return []
    timestamps = result.get("timestamp") or []
    closes = ((result.get("indicators") or {}).get("quote") or [{}])[0].get("close") or []
    points = []
    for ts, close in zip(timestamps, closes):
        if not isinstance(close, (int, float)):
            continue
        try:
            points.append({
                "x": datetime.fromtimestamp(int(ts), timezone.utc).isoformat(),
                "y": round(float(close), 4),
            })
        except Exception:
            continue
    return points


def fetch_yahoo_quote(symbol):
    headers = {"User-Agent": "Mozilla/5.0 (compatible; PolitiMarket/1.0)"}
    try:
        res = requests.get(
            f"https://query1.finance.yahoo.com/v8/finance/chart/{symbol}",
            params={"range": "5d", "interval": "1d"},
            headers=headers,
            timeout=8,
        )
        res.raise_for_status()
        payload = res.json()
        result = (payload.get("chart", {}).get("result") or [None])[0]
        if not result:
            raise RuntimeError("empty quote result")
        meta = result.get("meta") or {}
        price = meta.get("regularMarketPrice")
        timestamps = result.get("timestamp") or []
        closes = ((result.get("indicators") or {}).get("quote") or [{}])[0].get("close") or []
        valid_closes = [value for value in closes if isinstance(value, (int, float))]
        previous = (
            meta.get("regularMarketPreviousClose")
            or meta.get("previousClose")
            or meta.get("chartPreviousClose")
            or (valid_closes[-2] if len(valid_closes) >= 2 else None)
        )
        if price is None and valid_closes:
            price = valid_closes[-1]
        if price is None:
            raise RuntimeError("missing price")
        change = ((float(price) - float(previous)) / float(previous)) * 100 if previous else 0.0
        return {
            "price": round(float(price), 2),
            "change": round(float(change), 2),
            "currency": meta.get("currency") or "USD",
            "exchange": meta.get("exchangeName") or "",
            "updated_at": meta.get("regularMarketTime") or (timestamps[-1] if timestamps else None),
            "source": "Yahoo Finance chart",
        }, None
    except Exception as exc:
        return None, str(exc)


def fetch_naver_index_quote(code):
    try:
        res = requests.get(
            f"https://polling.finance.naver.com/api/realtime/domestic/index/{code}",
            headers={"User-Agent": "Mozilla/5.0 (compatible; PolitiMarket/1.0)", "Referer": "https://finance.naver.com/"},
            timeout=8,
        )
        res.raise_for_status()
        data = res.json()
        row = (data.get("datas") or [None])[0]
        if not row:
            raise RuntimeError("empty Naver index quote")
        price = parse_market_number(row.get("closePrice"))
        change = parse_market_number(row.get("fluctuationsRatio"), 0.0)
        if price is None:
            raise RuntimeError("missing Naver index closePrice")
        return {
            "price": round(float(price), 2),
            "change": round(float(change or 0), 2),
            "currency": "KRW",
            "exchange": code,
            "updated_at": row.get("localTradedAt") or row.get("tradeStopType") or datetime.now(ZoneInfo("Asia/Seoul")).isoformat(),
            "source": "Naver Finance realtime",
        }, None
    except Exception as exc:
        return None, str(exc)


def fetch_naver_index_daily(code, max_pages=5):
    rows = {}
    headers = {"User-Agent": "Mozilla/5.0 (compatible; PolitiMarket/1.0)", "Referer": "https://finance.naver.com/"}
    for page in range(1, max_pages + 1):
        try:
            res = requests.get(
                "https://finance.naver.com/sise/sise_index_day.naver",
                params={"code": code, "page": page},
                headers=headers,
                timeout=8,
            )
            res.raise_for_status()
            res.encoding = "euc-kr"
            for tr in re.findall(r"<tr[^>]*>(.*?)</tr>", res.text, re.S | re.I):
                if 'class="date"' not in tr:
                    continue
                cells = []
                for td in re.findall(r"<td[^>]*>(.*?)</td>", tr, re.S | re.I):
                    text = re.sub(r"<[^>]+>", " ", td)
                    text = re.sub(r"\s+", " ", text).strip()
                    if text:
                        cells.append(text)
                if len(cells) < 2 or not re.match(r"\d{4}\.\d{2}\.\d{2}$", cells[0]):
                    continue
                value = parse_market_number(cells[1])
                if value is None:
                    continue
                dt = datetime.strptime(cells[0], "%Y.%m.%d").replace(tzinfo=ZoneInfo("Asia/Seoul"))
                rows[cells[0]] = {"x": dt.isoformat(), "y": round(float(value), 4)}
        except Exception:
            continue
    return [rows[key] for key in sorted(rows.keys())]


@app.route("/api/market_indices")
def api_market_indices():
    from concurrent.futures import ThreadPoolExecutor, as_completed

    metric_symbols = {
        "kospi": "^KS11",
        "kosdaq": "^KQ11",
        "krwusd": "KRW=X",
        "sp500": "^GSPC",
        "nasdaq": "^IXIC",
        "dow": "^DJI",
    }
    naver_codes = {"kospi": "KOSPI", "kosdaq": "KOSDAQ"}

    def fetch_metric(metric, symbol):
        metric_errors = []
        minutes, daily, monthly = [], [], []
        quote = None

        if metric in naver_codes:
            quote, error = fetch_naver_index_quote(naver_codes[metric])
            if error:
                metric_errors.append({"source": "Naver realtime", "error": error})
            daily = fetch_naver_index_daily(naver_codes[metric], max_pages=5)

        for frame, range_value, interval in (
            ("minutes", "1d", "5m"),
            ("daily", "1mo", "1d"),
            ("monthly", "1y", "1mo"),
        ):
            if frame == "daily" and daily:
                continue
            try:
                points = yahoo_points(symbol, range_value, interval)
                if frame == "minutes":
                    minutes = points
                elif frame == "daily":
                    daily = points
                else:
                    monthly = points
            except Exception as exc:
                metric_errors.append({"source": f"Yahoo {range_value}/{interval}", "error": str(exc)})

        if not quote:
            quote, error = fetch_yahoo_quote(symbol)
            if error:
                metric_errors.append({"source": "Yahoo quote", "error": error})

        return metric, {
            "quote": quote,
            "market": {
                "minutes": minutes,
                "daily": daily,
                "monthly": monthly,
                "price": quote.get("price") if quote else None,
                "change": quote.get("change") if quote else None,
                "source": "Naver Finance + Yahoo Finance" if metric in naver_codes else "Yahoo Finance chart",
            },
            "errors": metric_errors,
        }

    market = {}
    quotes = {}
    errors = []
    with ThreadPoolExecutor(max_workers=6) as executor:
        futures = [executor.submit(fetch_metric, metric, symbol) for metric, symbol in metric_symbols.items()]
        for future in as_completed(futures):
            metric, payload = future.result()
            quote = payload["quote"]
            if quote:
                quotes[metric] = quote
            market[metric] = payload["market"]
            if payload["errors"]:
                errors.append({"metric": metric, "symbol": metric_symbols[metric], "errors": payload["errors"]})

    ordered_market = {metric: market.get(metric, {
            "minutes": [],
            "daily": [],
            "monthly": [],
            "price": None,
            "change": None,
            "source": "",
        }) for metric in metric_symbols}
    ordered_quotes = {metric: quotes[metric] for metric in metric_symbols if metric in quotes}

    return jsonify({"market": ordered_market, "quotes": ordered_quotes, "errors": errors})
def extract_json_block(content, var_name):
    idx = content.find(var_name)
    if idx == -1:
        return None
    start_char = None
    start_idx = -1
    for i in range(idx + len(var_name), len(content)):
        if content[i] == '[':
            start_char = '['
            start_idx = i
            break
        elif content[i] == '{':
            start_char = '{'
            start_idx = i
            break
    if start_idx == -1:
        return None
    
    end_char = ']' if start_char == '[' else '}'
    brace_count = 0
    in_string = False
    escape = False
    for i in range(start_idx, len(content)):
        char = content[i]
        if escape:
            escape = False
            continue
        if char == '\\':
            escape = True
            continue
        if char == '"':
            in_string = not in_string
            continue
        if not in_string:
            if char == start_char:
                brace_count += 1
            elif char == end_char:
                brace_count -= 1
                if brace_count == 0:
                    return content[start_idx:i+1]
    return None

def _pulse_match_key(item):
    content = item.get('content') or item.get('title') or ''
    return f"{item.get('source', '')}|{content[:240]}"

def _extract_crawled_pulse_data():
    js_path = os.path.join(crolling_dir, "data.js")
    if not os.path.exists(js_path):
        return []
    try:
        with open(js_path, 'r', encoding='utf-8', errors='replace') as f:
            content = f.read()
        block = extract_json_block(content, "crawledPulseData")
        if not block:
            return []
        data = json.loads(block)
        return data if isinstance(data, list) else []
    except Exception:
        return []

def reconcile_progress_with_current_feed(progress):
    if progress.get("status") in ("running", "scoring"):
        return progress

    analysis_path = os.path.join(filter_dir, "output", "category_analysis.json")
    current_items = _extract_crawled_pulse_data()
    if not current_items or not os.path.exists(analysis_path):
        return progress

    try:
        with open(analysis_path, 'r', encoding='utf-8') as f:
            analysis = json.load(f)
    except Exception:
        return progress

    classified = {}
    category_counts = {category: 0 for category in analysis.get("category_order", [])}

    def put(item, payload):
        if item.get("url"):
            classified[item["url"]] = payload
        if item.get("id"):
            classified[item["id"]] = payload
        classified[_pulse_match_key(item)] = payload

    for category, data in (analysis.get("categories") or {}).items():
        for item in data.get("all_items") or data.get("items") or []:
            payload = dict(item)
            payload["primary_tag"] = payload.get("primary_tag") or category
            put(item, payload)

    for item in analysis.get("excluded") or []:
        payload = dict(item)
        payload["excluded"] = True
        put(item, payload)

    completed = 0
    excluded = 0
    pending = 0
    for item in current_items:
        match = classified.get(item.get("url", "")) or classified.get(item.get("id", "")) or classified.get(_pulse_match_key(item))
        if not match:
            pending += 1
        elif match.get("excluded"):
            excluded += 1
        else:
            tag = match.get("primary_tag")
            if tag in category_counts:
                category_counts[tag] += 1
            completed += 1

    if pending == 0 and completed + excluded == int(progress.get("total_count", 0)):
        return progress

    reconciled = dict(progress)
    reconciled.update({
        "total_count": len(current_items),
        "in_progress_count": pending,
        "completed_count": completed,
        "excluded_count": excluded,
        "category_counts": category_counts,
        "status": "pending" if pending else progress.get("status", "complete")
    })
    return reconciled

def reconcile_progress_with_db(progress):
    """Return a DB-wide tagging snapshot using the same scope as tag audit."""
    try:
        if crawling_dir not in sys.path:
            sys.path.insert(0, crawling_dir)
        import db_utils

        test_mode = load_test_mode()
        test_only = bool(test_mode.get("active"))
        lookback_where = "" if test_only else crawler_lookback_where_sql("c")
        category_order = ["IT", "Energy", "Finance", "Healthcare", "Commodities", "Defense", "Chemicals", "Shipbuilding"]
        active_model_version = str(progress.get("model_version") or "")
        db_utils.init_db()
        with db_utils.connect() as conn:
            summary = conn.execute(
                f"""
                SELECT
                    COUNT(*) AS total_items,
                    SUM(CASE WHEN r.item_id IS NOT NULL AND COALESCE(r.excluded, 0) = 0 THEN 1 ELSE 0 END) AS tagged_items,
                    SUM(CASE WHEN COALESCE(r.excluded, 0) = 1 OR q.status = 'excluded' THEN 1 ELSE 0 END) AS excluded_items,
                    SUM(CASE WHEN r.item_id IS NULL AND COALESCE(q.status, 'not_queued') IN ('pending', 'tagging', 'failed', 'not_queued') THEN 1 ELSE 0 END) AS pending_items
                  FROM crawled_items c
                  LEFT JOIN tagging_queue q ON q.item_id = c.id
                  LEFT JOIN tag_results r ON r.item_id = c.id
                   AND (? = '' OR r.model_version = ?)
                 WHERE ((? = 1 AND c.source_group = 'test') OR (? = 0 AND COALESCE(c.source_group, '') <> 'test'))
                   {lookback_where}
                """,
                (active_model_version, active_model_version, 1 if test_only else 0, 1 if test_only else 0),
            ).fetchone()
            category_counts = {category: 0 for category in category_order}
            for row in conn.execute(
                f"""
                SELECT r.primary_tag AS tag, COUNT(*) AS count
                  FROM tag_results r
                  JOIN crawled_items c ON c.id = r.item_id
                 WHERE COALESCE(r.excluded, 0) = 0
                   AND COALESCE(r.primary_tag, '') <> ''
                   AND (? = '' OR r.model_version = ?)
                   AND ((? = 1 AND c.source_group = 'test') OR (? = 0 AND COALESCE(c.source_group, '') <> 'test'))
                   {lookback_where}
                 GROUP BY r.primary_tag
                """,
                (active_model_version, active_model_version, 1 if test_only else 0, 1 if test_only else 0),
            ).fetchall():
                tag = row["tag"]
                if tag in category_counts:
                    category_counts[tag] = int(row["count"] or 0)
            version_row = conn.execute(
                """
                SELECT model_version
                  FROM tag_results
                 WHERE COALESCE(model_version, '') <> ''
                 GROUP BY model_version
                 ORDER BY COUNT(*) DESC, MAX(tagged_at) DESC
                 LIMIT 1
                """
            ).fetchone()

        total = int(summary["total_items"] or 0)
        completed = int(summary["tagged_items"] or 0)
        excluded = int(summary["excluded_items"] or 0)
        pending = int(summary["pending_items"] or 0)
        reconciled = dict(progress)
        source_status = str(progress.get("status") or "")
        reconciled.update({
            "total_count": total,
            "in_progress_count": pending,
            "completed_count": completed,
            "excluded_count": excluded,
            "category_counts": category_counts,
            "status": source_status if source_status in {"blocked", "error"} else ("running" if pending else "complete"),
            "model_version": active_model_version or (version_row["model_version"] if version_row else ""),
            "count_scope": "db_lookback",
        })
        return reconciled
    except Exception:
        return reconcile_progress_with_current_feed(progress)

# API: Status
@app.route('/api/status')
def status():
    # 1. Check Scheduler Status
    running = False
    active_pid = 0
    if os.path.exists(pid_path):
        try:
            with open(pid_path, 'r') as f:
                pid = int(f.read().strip())
                if is_pid_running(pid):
                    running = True
                    active_pid = pid
                else:
                    # Clean up stale PID file
                    try:
                        os.remove(pid_path)
                    except:
                        pass
        except:
            pass
            
    # 2. Check Truth Social Tokens
    truth_tokens = []
    truth_file = os.path.join(crawling_dir, "truth_tokens.json")
    if os.path.exists(truth_file):
        try:
            with open(truth_file, 'r', encoding='utf-8') as f:
                truth_tokens = json.load(f)
        except:
            pass

    # 4. Check Crawler Platform Status
    crawler_status = {}
    if os.path.exists(crawler_status_path):
        try:
            with open(crawler_status_path, 'r', encoding='utf-8') as f:
                crawler_status = json.load(f)
        except:
            pass
    crawler_config = load_crawler_config()
    crawler_status = apply_scheduler_state_to_crawler_status(crawler_status, crawler_config, running)
    if not running:
        try:
            save_crawler_status(crawler_status)
        except Exception:
            pass
    return jsonify({
        "scheduler": {
            "running": running,
            "pid": active_pid
        },
        "accounts": {
            "x": [],
            "truth": truth_tokens
        },
        "won": load_won_config(),
        "llm_usage": load_llm_usage(),
        "x_api": public_x_api_config(),
        "crawler_config": crawler_config,
        "crawler_status": crawler_status
    })

@app.route('/api/won/usage')
def won_usage():
    return jsonify(load_llm_usage())

@app.route('/api/won/config', methods=['POST'])
def save_won_model_config():
    data = request.json or {}
    api_key = data.get("api_key", "").strip()
    model = data.get("model", default_won_model)
    api_url = data.get("api_url", "").strip()
    enabled = data.get("enabled", True)
    try:
        config = save_won_config(api_key, model, enabled, api_url)
        invalidate_summary_cache()
        rebuild_started = trigger_analysis_rebuild("LLM 설정 변경")
        suffix = " 요약 재생성을 백그라운드에서 시작했습니다." if rebuild_started else " 다음 NLP 실행 때 새 설정이 반영됩니다."
        return jsonify({
            "status": "success",
            "message": f"{supported_won_models[config['model']]} 설정이 저장되었습니다.{suffix}",
            "won": config
        })
    except Exception as e:
        return jsonify({"status": "error", "message": f"LLM 설정 저장 실패: {str(e)}"})

@app.route('/api/won/config', methods=['DELETE'])
def delete_won_model_config():
    try:
        if os.path.exists(won_config_path):
            os.remove(won_config_path)
        if os.path.exists(legacy_won_config_path):
            os.remove(legacy_won_config_path)
        invalidate_summary_cache()
        rebuild_started = trigger_analysis_rebuild("LLM 설정 삭제")
        suffix = " 기본 요약 재생성을 백그라운드에서 시작했습니다." if rebuild_started else ""
        return jsonify({
            "status": "success",
            "message": f"LLM 설정이 삭제되었습니다.{suffix}",
            "won": load_won_config()
        })
    except Exception as e:
        return jsonify({"status": "error", "message": f"LLM 설정 삭제 실패: {str(e)}"})

@app.route('/api/won/test', methods=['POST'])
def test_won_model_config():
    data = request.json or {}
    api_key = data.get("api_key", "").strip()
    model = data.get("model", "").strip()
    api_url = data.get("api_url", "").strip()
    saved_config = load_won_config(include_key=True)
    if not api_key:
        api_key = saved_config.get("api_key", "")
    if not model:
        model = saved_config.get("model", default_won_model)
    if not api_url:
        api_url = saved_config.get("api_url", "")
    try:
        text = call_won_test(api_key, model, api_url)
        return jsonify({
            "status": "success",
            "message": f"{supported_won_models[model]} 연동 확인 성공: {text}",
            "model": model
        })
    except Exception as e:
        return jsonify({
            "status": "error",
            "message": f"LLM 연동 실패 ({model}): {str(e)}"
        })

@app.route('/api/collectors/config', methods=['POST'])
def update_collector_config():
    data = request.json or {}
    platform = data.get('platform')
    enabled = data.get('enabled')

    if platform not in collector_keys:
        return jsonify({"status": "error", "message": "지원하지 않는 수집 대상입니다."})
    if enabled is None:
        return jsonify({"status": "error", "message": "enabled 값이 누락되었습니다."})

    config = load_crawler_config()
    config["enabled"][platform] = bool(enabled)
    config = save_crawler_config(config)

    try:
        if config["enabled"][platform]:
            set_platform_status(platform, "pending", "수집 재개됨. 다음 루프에서 반영됩니다.")
        else:
            set_platform_status(platform, "disabled", "사용자 설정으로 비활성화됨")
    except Exception as e:
        print(f"Error updating platform status: {e}")

    label = {
        "x": "X",
        "truth": "Truth Social",
        "gov": "정부 기관",
        "news": "뉴스",
        "thinktank": "싱크탱크",
        "axios": "Axios",
        "market": "시장지표",
        "russia": "러시아 Kremlin",
        "china": "중국 인민망"
    }[platform]
    state = "켜짐" if config["enabled"][platform] else "꺼짐"
    return jsonify({
        "status": "success",
        "message": f"{label} 수집이 {state}으로 변경되었습니다.",
        "crawler_config": config
    })

@app.route('/api/x/config', methods=['POST'])
def save_x_api_config():
    data = request.json or {}
    config = load_crawler_config()
    config.setdefault("enabled", {})["x"] = True
    x_api = config.setdefault("x_api", json.loads(json.dumps(default_x_api_config)))

    token = str(data.get("bearer_token") or data.get("api_key") or "").strip()
    if token:
        x_api["bearer_token"] = token
    if data.get("clear_token"):
        x_api["bearer_token"] = ""

    if "accounts" in data:
        x_api["accounts"] = normalize_x_lines(data.get("accounts"))
    if "queries" in data:
        x_api["queries"] = normalize_x_lines(data.get("queries"))
    if not x_api.get("accounts") and not x_api.get("queries"):
        x_api["accounts"] = list(default_x_api_config["accounts"])
    else:
        x_api["accounts"] = with_default_x_accounts(x_api.get("accounts"))

    for key, default, minimum in [
        ("recent_lookback_days", 1, 1),
        ("backfill_days", 7, 1),
    ]:
        try:
            x_api[key] = max(minimum, int(data.get(key, x_api.get(key, default)) or default))
        except Exception:
            x_api[key] = default

    for key in ("use_full_archive", "exclude_retweets", "exclude_replies"):
        if key in data:
            x_api[key] = bool(data.get(key))

    saved = save_crawler_config(config)
    try:
        set_platform_status("x", "pending", "X API configuration updated")
    except Exception:
        pass
    return jsonify({
        "status": "success",
        "message": "X API 설정이 저장되었습니다.",
        "x_api": public_x_api_config(),
        "crawler_config": saved,
    })

@app.route('/api/x/test', methods=['POST'])
def test_x_api_config():
    try:
        if crawling_dir not in sys.path:
            sys.path.insert(0, crawling_dir)
        import crawler

        config = load_crawler_config()
        items = crawler.collect_x_api(config, limit=1, days=1, backfill=False)
        return jsonify({
            "status": "success",
            "message": f"X API 연동 확인 성공: 샘플 {len(items)}건 조회",
            "sample_count": len(items),
            "x_api": public_x_api_config(),
        })
    except Exception as e:
        return jsonify({"status": "error", "message": f"X API 연동 실패: {str(e)}"})

@app.route('/api/crawl/backfill', methods=['POST'])
def run_crawl_backfill():
    data = request.json or {}
    config = load_crawler_config()
    try:
        days = max(1, int(data.get("days") or config.get("x_api", {}).get("backfill_days") or 7))
    except Exception:
        days = 7
    try:
        if os.path.exists(backfill_lock_path):
            return jsonify({"status": "error", "message": "이미 전체 백필이 실행 중입니다. 완료 후 다시 시도해 주세요."})
        with open(backfill_lock_path, "w", encoding="utf-8") as lock_file:
            json.dump({"days": days, "started_at": time_str(), "owner": "gui"}, lock_file, ensure_ascii=False)
        with open(log_path, 'a', encoding='utf-8') as f:
            f.write(f"[{time_str()}] [SYSTEM] all-source crawl backfill requested for {days} days.\n")

        creationflags = subprocess.CREATE_NO_WINDOW if sys.platform == 'win32' else 0
        proc = subprocess.run(
            [sys.executable, "crawler.py", "--backfill-days", str(days)],
            cwd=crawling_dir,
            capture_output=True,
            text=True,
            timeout=3600,
            creationflags=creationflags,
        )
        output = (proc.stdout or "") + (proc.stderr or "")
        with open(log_path, 'a', encoding='utf-8') as f:
            f.write(output)
        if proc.returncode != 0:
            if os.path.exists(backfill_lock_path):
                try:
                    os.remove(backfill_lock_path)
                except OSError:
                    pass
            return jsonify({
                "status": "error",
                "message": f"전체 백필 실패: {output[-1200:] or proc.returncode}",
            })
        return jsonify({
            "status": "success",
            "message": f"전체 수집원 {days}일 백필이 완료되었습니다.",
            "output": output[-1200:],
        })
    except subprocess.TimeoutExpired:
        if os.path.exists(backfill_lock_path):
            try:
                os.remove(backfill_lock_path)
            except OSError:
                pass
        return jsonify({"status": "error", "message": "전체 백필이 1시간 제한 시간을 초과했습니다."})
    except Exception as e:
        if os.path.exists(backfill_lock_path):
            try:
                os.remove(backfill_lock_path)
            except OSError:
                pass
        return jsonify({"status": "error", "message": f"전체 백필 실행 실패: {str(e)}"})

# API: Start Scheduler Loop
@app.route('/api/scheduler/start', methods=['POST'])
def start_scheduler():
    if os.path.exists(backfill_lock_path):
        return jsonify({"status": "error", "message": "전체 백필 실행 중에는 스케줄러를 시작할 수 없습니다."})
    if os.path.exists(pid_path):
        with open(pid_path, 'r') as f:
            try:
                pid = int(f.read().strip())
                if is_pid_running(pid):
                    return jsonify({"status": "error", "message": f"크롤러가 이미 실행 중입니다. (PID: {pid})"})
            except:
                pass
                
    try:
        # Clear log file on startup
        with open(log_path, 'w', encoding='utf-8') as f:
            f.write(f"[{time_str()}] [SYSTEM] GUI Dashboard starting crawler loop...\n")

        config = load_crawler_config()
        try:
            with open(crawler_status_path, 'w', encoding='utf-8') as f:
                json.dump({
                    "loop": {
                        "running": True,
                        "crawl_interval_seconds": CRAWL_INTERVAL_SECONDS,
                        "crawl_base_interval_seconds": CRAWL_INTERVAL_SECONDS,
                        "crawl_jitter_ratio": CRAWL_JITTER_RATIO,
                        "active_crawl_interval_seconds": None,
                        "last_crawl_at": None,
                        "next_crawl_at": None,
                        "seconds_until_next_crawl": 0,
                        "platform_next": {}
                    },
                    "platforms": {
                        p: {
                            "status": "pending" if config["enabled"].get(p, True) else "disabled",
                            "last_run": None,
                            "error": "루프 시작 후 첫 수집 대기 중" if config["enabled"].get(p, True) else "사용자 설정으로 비활성화됨"
                        }
                        for p in collector_keys
                    },
                    "last_updated": time_str()
                }, f, ensure_ascii=False, indent=2)
        except Exception as e:
            print(f"Error resetting crawler status: {e}")
            
        stdout_path = os.path.join(crawling_dir, "crawler_loop.stdout.log")
        log_file = open(stdout_path, 'a', encoding='utf-8', errors='replace')
        
        # Start run_crawler_loop.py as a separate background process redirecting output to log file
        creationflags = 0
        if sys.platform == 'win32':
            # Run without a console window to prevent console event propagation to parent batch script
            creationflags = subprocess.CREATE_NO_WINDOW

        subprocess.Popen(
            [sys.executable, "-u", "run_crawler_loop.py"],
            stdout=log_file,
            stderr=log_file,
            cwd=crawling_dir,
            creationflags=creationflags
        )
        return jsonify({"status": "success", "message": "실시간 크롤러 스케줄러가 성공적으로 시작되었습니다."})
    except Exception as e:
        return jsonify({"status": "error", "message": f"스케줄러 시작 실패: {str(e)}"})

# API: Stop Scheduler Loop
@app.route('/api/scheduler/stop', methods=['POST'])
def stop_scheduler():
    if not os.path.exists(pid_path):
        return jsonify({"status": "error", "message": "실행 중인 크롤러 프로세스가 없습니다."})
        
    try:
        with open(pid_path, 'r') as f:
            pid = int(f.read().strip())
            
        if is_pid_running(pid):
            # Kill process group or process directly on Windows
            if sys.platform == 'win32':
                subprocess.run(["taskkill", "/F", "/T", "/PID", str(pid)], capture_output=True)
            else:
                os.kill(pid, signal.SIGTERM)
                
            # Log termination
            with open(log_path, 'a', encoding='utf-8') as f:
                f.write(f"[{time_str()}] [SYSTEM] GUI Dashboard terminated crawler loop (PID: {pid}).\n")
                
            # Ensure PID file is removed
            if os.path.exists(pid_path):
                try:
                    os.remove(pid_path)
                except:
                    pass
            try:
                save_crawler_status(apply_scheduler_state_to_crawler_status({}, load_crawler_config(), False))
            except Exception as e:
                print(f"Error marking crawler status stopped: {e}")
            return jsonify({"status": "success", "message": "크롤러 프로세스가 성공적으로 정지되었습니다."})
        else:
            if os.path.exists(pid_path):
                os.remove(pid_path)
            try:
                save_crawler_status(apply_scheduler_state_to_crawler_status({}, load_crawler_config(), False))
            except Exception as e:
                print(f"Error marking crawler status stopped: {e}")
            return jsonify({"status": "error", "message": "PID 파일은 존재하나 실제 프로세스가 동작하고 있지 않아 PID 파일을 정리했습니다."})
    except Exception as e:
        return jsonify({"status": "error", "message": f"스케줄러 정지 실패: {str(e)}"})

@app.route('/api/crawl_data/reset', methods=['POST'])
def api_reset_crawl_data():
    try:
        deleted = reset_crawl_data()
        return jsonify({
            "status": "success",
            "deleted": deleted,
            "message": f"기존 크롤링 데이터 {deleted}건과 태깅/요약 파생 데이터를 삭제했습니다. 즉시 수집 또는 다음 루프에서 새로 수집됩니다.",
        })
    except Exception as e:
        return jsonify({"status": "error", "message": f"크롤링 데이터 초기화 실패: {str(e)}"}), 500

@app.route('/api/tagging/reset_rework', methods=['POST'])
def api_reset_tagging_rework():
    try:
        queued = reset_tagging_work()
        if queued:
            start_tagging_rework_worker()
            message = f"태깅 작업을 초기화하고 {queued}건을 다시 작업하도록 등록했습니다."
        else:
            message = "태깅할 수집 데이터가 없습니다. 먼저 수집을 실행해 주세요."
        return jsonify({
            "status": "success",
            "queued": queued,
            "message": message,
        })
    except Exception as e:
        return jsonify({"status": "error", "message": f"태깅 작업 초기화 실패: {str(e)}"}), 500

# API: Force One-off Crawl (Market-Only)
@app.route('/api/scheduler/force', methods=['POST'])
def force_crawl():
    try:
        config = load_crawler_config()
        if config["enabled"].get("market", True) is False:
            set_platform_status("market", "disabled", "사용자 설정으로 비활성화됨")
            return jsonify({"status": "success", "message": "시장지표 수집이 꺼져 있어 강제 수집을 건너뛰었습니다."})

                # Run the full real-time crawl/tag/analyse pipeline once synchronously.
        with open(log_path, 'a', encoding='utf-8') as f:
            f.write(f"[{time_str()}] [SYSTEM] GUI forced real-time crawl execution initiated...\n")

        creationflags = 0
        if sys.platform == 'win32':
            creationflags = subprocess.CREATE_NO_WINDOW

        p1 = subprocess.run([sys.executable, "crawler.py", "--once"], capture_output=True, text=True, cwd=crawling_dir, creationflags=creationflags)

        with open(log_path, 'a', encoding='utf-8') as f:
            f.write((p1.stdout or "") + (p1.stderr or ""))
            
        return jsonify({"status": "success", "message": "1회 강제 크롤링 수집이 성공적으로 완료되었습니다."})
    except Exception as e:
        return jsonify({"status": "error", "message": f"강제 수집 실패: {str(e)}"})

# API: Truth Social Auto Login (App registration + token generation)
@app.route('/api/login/truth_auto', methods=['POST'])
def login_truth_auto():
    data = request.json
    username = data.get('username')
    password = data.get('password')
    
    if not username or not password:
        return jsonify({"status": "error", "message": "이메일/아이디와 비밀번호가 필요합니다."})
        
    app_url = "https://truthsocial.com/api/v1/apps"
    app_data = {
        "client_name": "PolitiMarket Crawler",
        "redirect_uris": "urn:ietf:wg:oauth:2.0:oob",
        "scopes": "read"
    }
    
    try:
        # 1. Register Client App
        r = requests.post(app_url, data=app_data, timeout=10, verify=False)
        if r.status_code != 200:
            return jsonify({"status": "error", "message": f"앱 등록 실패: {r.status_code} - {r.text}"})
            
        app_info = r.json()
        client_id = app_info["client_id"]
        client_secret = app_info["client_secret"]
        
        # 2. Authenticate
        token_url = "https://truthsocial.com/oauth/token"
        token_data = {
            "client_id": client_id,
            "client_secret": client_secret,
            "grant_type": "password",
            "username": username,
            "password": password,
            "scope": "read"
        }
        
        r = requests.post(token_url, data=token_data, timeout=10, verify=False)
        if r.status_code != 200:
            return jsonify({"status": "error", "message": f"로그인 토큰 발급 실패: {r.status_code} - {r.text}"})
            
        token_info = r.json()
        access_token = token_info["access_token"]
        
        # Append token to truth_tokens.json
        tokens_file = os.path.join(crawling_dir, "truth_tokens.json")
        tokens = []
        if os.path.exists(tokens_file):
            try:
                with open(tokens_file, 'r', encoding='utf-8') as f:
                    tokens = json.load(f)
            except:
                pass
                
        if access_token not in tokens:
            tokens.append(access_token)
            
        with open(tokens_file, 'w', encoding='utf-8') as f:
            json.dump(tokens, f, ensure_ascii=False, indent=2)
            
        with open(log_path, 'a', encoding='utf-8') as f:
            f.write(f"[{time_str()}] [SYSTEM] Truth Social account logged in automatically. Token saved.\n")
            
        return jsonify({"status": "success", "message": "Truth Social API 로그인 및 액세스 토큰 등록이 성공적으로 완료되었습니다."})
    except Exception as e:
        return jsonify({"status": "error", "message": f"네트워크 오류: {str(e)}"})

# API: Truth Social Manual Token Paste
@app.route('/api/login/truth_manual', methods=['POST'])
def login_truth_manual():
    data = request.json
    token = data.get('token', '').strip()
    
    if not token:
        return jsonify({"status": "error", "message": "액세스 토큰을 입력해 주세요."})
        
    tokens_file = os.path.join(crawling_dir, "truth_tokens.json")
    tokens = []
    if os.path.exists(tokens_file):
        try:
            with open(tokens_file, 'r', encoding='utf-8') as f:
                tokens = json.load(f)
        except:
            pass
            
    if token not in tokens:
        tokens.append(token)
        
    try:
        with open(tokens_file, 'w', encoding='utf-8') as f:
            json.dump(tokens, f, ensure_ascii=False, indent=2)
            
        with open(log_path, 'a', encoding='utf-8') as f:
            f.write(f"[{time_str()}] [SYSTEM] Truth Social token added manually.\n")
            
        return jsonify({"status": "success", "message": "수동 입력된 Truth Social 인증 토큰이 성공적으로 추가되었습니다."})
    except Exception as e:
        return jsonify({"status": "error", "message": f"토큰 저장 실패: {str(e)}"})

# API: Delete Account Tokens
@app.route('/api/account/delete', methods=['POST'])
def delete_account():
    acct_type = request.args.get('type')
    slot = request.args.get('slot')
    
    if not acct_type:
        return jsonify({"status": "error", "message": "계정 타입이 누락되었습니다."})
        
    if acct_type == 'truth':
        tokens_file = os.path.join(crawling_dir, "truth_tokens.json")
        if os.path.exists(tokens_file):
            try:
                with open(tokens_file, 'r', encoding='utf-8') as f:
                    tokens = json.load(f)
                
                idx = int(slot)
                if 0 <= idx < len(tokens):
                    removed = tokens.pop(idx)
                    with open(tokens_file, 'w', encoding='utf-8') as f:
                        json.dump(tokens, f, ensure_ascii=False, indent=2)
                        
                    with open(log_path, 'a', encoding='utf-8') as f:
                        f.write(f"[{time_str()}] [SYSTEM] Truth Social token deleted.\n")
                        
                    return jsonify({"status": "success", "message": "해당 Truth Social 토큰이 성공적으로 제거되었습니다."})
                else:
                    return jsonify({"status": "error", "message": "해당 토큰 인덱스가 존재하지 않습니다."})
            except Exception as e:
                return jsonify({"status": "error", "message": f"토큰 삭제 실패: {str(e)}"})
        else:
            return jsonify({"status": "error", "message": "토큰 파일이 존재하지 않습니다."})

    return jsonify({"status": "error", "message": "지원하지 않는 계정 타입입니다."})

# API: Verify Account Tokens
@app.route('/api/account/verify', methods=['POST'])
def verify_account():
    acct_type = request.args.get('type')
    slot = request.args.get('slot')
    
    if not acct_type:
        return jsonify({"status": "error", "message": "계정 타입이 누락되었습니다."})
        
    if acct_type == 'truth':
        tokens_file = os.path.join(crawling_dir, "truth_tokens.json")
        if not os.path.exists(tokens_file):
            return jsonify({"status": "error", "message": "토큰 파일이 존재하지 않습니다."})
            
        try:
            with open(tokens_file, 'r', encoding='utf-8') as f:
                tokens = json.load(f)
            
            idx = int(slot)
            if 0 <= idx < len(tokens):
                token = tokens[idx]
                
                # Check using verify_credentials
                url = "https://truthsocial.com/api/v1/accounts/verify_credentials"
                headers = {
                    'Authorization': f'Bearer {token}',
                    'User-Agent': 'Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/120.0.0.0 Safari/537.36'
                }
                
                try:
                    r = requests.get(url, headers=headers, timeout=25, verify=False)
                except requests.Timeout:
                    return jsonify({
                        "status": "error",
                        "message": "Truth Social 응답 시간이 초과되었습니다. 인터넷 연결은 되어 있어도 truthsocial.com이 현재 IP/지역/VPN 요청을 지연하거나 차단하는 상태일 수 있습니다."
                    })
                except requests.RequestException as e:
                    return jsonify({"status": "error", "message": f"Truth Social 연결 실패: {str(e)}"})

                if r.status_code == 200:
                    info = r.json()
                    username = info.get('username', 'Unknown')
                    display_name = info.get('display_name', '')
                    name = f"@{username}"
                    if display_name:
                        name += f" ({display_name})"
                    return jsonify({"status": "success", "message": f"Truth Social 연동 확인 성공! 계정: {name}"})
                if r.status_code in (401, 403):
                    return jsonify({
                        "status": "error",
                        "message": f"Truth Social 연동 실패 (HTTP {r.status_code}). 토큰이 만료되었거나 Cloudflare/서비스 정책으로 현재 접속이 차단된 상태입니다. 브라우저에서 Truth Social 접속 가능 여부와 토큰 재발급을 확인해 주세요."
                    })
                else:
                    # Try lookup fallback
                    lookup_url = "https://truthsocial.com/api/v1/accounts/lookup?acct=realDonaldTrump"
                    try:
                        r2 = requests.get(lookup_url, headers=headers, timeout=25, verify=False)
                    except requests.Timeout:
                        return jsonify({
                            "status": "error",
                            "message": "Truth Social 공개 조회도 시간 초과되었습니다. 현재 네트워크/IP에서 truthsocial.com API 접근이 불안정합니다."
                        })
                    except requests.RequestException as e:
                        return jsonify({"status": "error", "message": f"Truth Social 공개 조회 실패: {str(e)}"})
                    if r2.status_code == 200:
                        return jsonify({"status": "success", "message": "Truth Social 연동 확인 성공! (Donald Trump 계정 조회 성공)"})
                    if r2.status_code in (401, 403):
                        return jsonify({
                            "status": "error",
                            "message": f"Truth Social 공개 조회 실패 (HTTP {r2.status_code}). 현재 IP/지역/VPN 또는 Cloudflare 보호 정책 때문에 API 접근이 차단된 것으로 보입니다."
                        })
                    else:
                        return jsonify({"status": "error", "message": f"연동 실패 (HTTP {r.status_code}): {r.text[:200]}"})
            else:
                return jsonify({"status": "error", "message": "해당 토큰 인덱스가 존재하지 않습니다."})
        except Exception as e:
            return jsonify({"status": "error", "message": f"연동 확인 중 오류 발생: {str(e)}"})
            
    return jsonify({"status": "error", "message": "지원하지 않는 계정 타입입니다."})

# API: Read logs from offset

@app.route('/api/logs')
def get_logs():
    try:
        offset = max(int(request.args.get('offset', 0)), 0)
    except Exception:
        offset = 0
    try:
        tail = max(int(request.args.get('tail', 0)), 0)
    except Exception:
        tail = 0
    session_only = str(request.args.get('session', '0')).lower() in {'1', 'true', 'yes'}
    hide_noise = str(request.args.get('hide_noise', '1')).lower() not in {'0', 'false', 'no'}
    if not os.path.exists(log_path):
        return jsonify({"lines": [], "last_line": 0, "start_line": 0})
        
    try:
        with open(log_path, 'rb') as f:
            raw = f.read()
        try:
            text = raw.decode('utf-8')
        except Exception:
            text = raw.decode('cp949', errors='replace')
        lines = text.splitlines()
            
        # If log was cleared or reset, adjust offset
        if offset > len(lines):
            offset = 0
        start = offset
        if offset == 0:
            if session_only:
                for idx in range(len(lines) - 1, -1, -1):
                    if '[SYSTEM] realtime crawler loop started' in lines[idx]:
                        start = idx
                        break
            if tail:
                start = max(start, len(lines) - tail)
        output_lines = []
        for line in lines[start:]:
            if hide_noise and 'skipped title-only article' in line:
                continue
            output_lines.append(line.strip())
            
        return jsonify({
            "lines": output_lines,
            "last_line": len(lines),
            "start_line": start,
        })
    except Exception as e:
        return jsonify({"lines": [f"Error reading log file: {str(e)}"], "last_line": 0, "start_line": 0})

# API: Read compiled market/news datasets from data.js
@app.route('/api/data')
def get_data():
    js_path = os.path.join(crolling_dir, "data.js")
    if not os.path.exists(js_path):
        return jsonify({"news": [], "market": {}})
        
    try:
        with open(js_path, 'r', encoding='utf-8') as f:
            content = f.read()
            
        news = []
        market = {}
        
        news_block = extract_json_block(content, "crawledPulseData")
        if news_block:
            news = json.loads(news_block)
            
        market_block = extract_json_block(content, "marketData")
        if market_block:
            market = json.loads(market_block)
            
        return jsonify({
            "news": news[:160],
            "market": market
        })
    except Exception as e:
        return jsonify({"error": str(e), "news": [], "market": {}})

# API: Per-source crawl statistics from DB
@app.route('/api/crawl_stats')
def get_crawl_stats():
    import sqlite3
    db_path = os.path.join(crawling_dir, "crawler.db")
    if not os.path.exists(db_path):
        return jsonify({"total": 0, "by_source": [], "by_category": []})
    try:
        conn = sqlite3.connect(db_path)
        conn.row_factory = sqlite3.Row

        # Total count
        total = conn.execute("SELECT COUNT(*) FROM crawled_items").fetchone()[0]

        # Per-source count, newest item timestamp
        rows = conn.execute("""
            SELECT source,
                   COUNT(*) AS cnt,
                   MAX(published_at) AS latest
            FROM crawled_items
            GROUP BY source
            ORDER BY cnt DESC
        """).fetchall()

        # Categorise sources into friendly groups
        def _category(src: str) -> str:
            s = src.lower()
            if s.startswith("x_"):
                return "X (Twitter)"
            if s.startswith("truthsocial"):
                return "Truth Social"
            if src == "Gov_Kremlin":
                return "러시아/정부"
            if src == "News_PeopleCN_KO":
                return "중국/인민망"
            if src == "Gov_Kremlin":
                return "러시아/정부"
            if src == "News_PeopleCN_KO":
                return "중국/인민망"
            if src == "Gov_Kremlin":
                return "러시아/정부"
            if src == "News_PeopleCN_KO":
                return "중국/인민망"
            if src == "Gov_Kremlin":
                return "러시아/정부"
            if src == "News_PeopleCN_KO":
                return "중국/인민망"
            if src == "Gov_Kremlin":
                return "러시아/정부"
            if src == "News_PeopleCN_KO":
                return "중국/인민망"
            if src == "Gov_Kremlin":
                return "러시아/정부"
            if src == "News_PeopleCN_KO":
                return "중국/인민망"
            if src == "Gov_Kremlin":
                return "러시아/정부"
            if src == "News_PeopleCN_KO":
                return "중국/인민망"
            if src == "Gov_Kremlin":
                return "러시아/정부"
            if src == "News_PeopleCN_KO":
                return "중국/인민망"
            if s.startswith("thinktank_"):
                return "싱크탱크"
            if s.startswith("axios"):
                return "Axios"
            if s.startswith("gov_"):
                return "정부/공공"
            if s.startswith("news_"):
                return "뉴스"
            if s.startswith("market"):
                return "시장지표"
            return "기타"

        by_source = []
        cat_agg: dict = {}
        for r in rows:
            src = r["source"]
            cnt = r["cnt"]
            latest = r["latest"] or ""
            cat = _category(src)
            by_source.append({"source": src, "count": cnt, "latest": latest, "category": cat})
            cat_agg[cat] = cat_agg.get(cat, 0) + cnt

        by_category = [{"category": k, "count": v} for k, v in sorted(cat_agg.items(), key=lambda x: -x[1])]

        conn.close()
        return jsonify({"total": total, "by_source": by_source, "by_category": by_category})
    except Exception as e:
        return jsonify({"error": str(e), "total": 0, "by_source": [], "by_category": []})


def time_str():
    import time
    return time.strftime('%Y-%m-%d %H:%M:%S')

if __name__ == '__main__':
    # Start web server on localhost:8080
    app.run(host='0.0.0.0', port=8080, debug=True, use_reloader=False)

