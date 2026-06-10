import os
import random
import signal
import sys
import time
import json
from datetime import datetime

import crawler
from db_utils import connect, init_db


BASE_DIR = os.path.dirname(os.path.abspath(__file__))
PID_PATH = os.path.join(BASE_DIR, "crawler.pid")
LOG_PATH = os.path.join(BASE_DIR, "crawler_loop.log")
STATUS_PATH = os.path.join(BASE_DIR, "crawler_status.json")
BACKFILL_LOCK_PATH = os.path.join(BASE_DIR, "backfill.lock")
CRAWL_INTERVAL_SECONDS = int(os.environ.get("CRAWLER_INTERVAL_SECONDS", "360"))
CRAWL_JITTER_RATIO = float(os.environ.get("CRAWLER_JITTER_RATIO", "0.2"))
LOOP_SLEEP_SECONDS = float(os.environ.get("CRAWLER_LOOP_SLEEP_SECONDS", "1"))
MARKET_INTERVAL_SECONDS = int(os.environ.get("CRAWLER_MARKET_INTERVAL_SECONDS", "15"))


running = True


def handle_stop(signum, frame):
    global running
    running = False


def log(message):
    with open(LOG_PATH, "a", encoding="utf-8", errors="replace") as f:
        f.write(f"[{crawler.time_str()}] {message}\n")
    print(message, flush=True)


def next_crawl_interval():
    jitter = max(0.0, min(0.95, CRAWL_JITTER_RATIO))
    min_interval = max(1.0, CRAWL_INTERVAL_SECONDS * (1.0 - jitter))
    return random.uniform(min_interval, float(CRAWL_INTERVAL_SECONDS))


def write_loop_timing(last_crawl, next_crawl, active_interval):
    data = {"platforms": {}}
    if os.path.exists(STATUS_PATH):
        try:
            with open(STATUS_PATH, "r", encoding="utf-8") as f:
                data = json.load(f)
        except Exception:
            data = {"platforms": {}}
    seconds_remaining = max(0, int(next_crawl - time.time())) if next_crawl else 0
    platform_next = {
        "x": next_crawl,
        "truth": next_crawl,
        "gov": next_crawl,
        "news": next_crawl,
        "thinktank": next_crawl,
        "axios": next_crawl,
        "market": time.time() + MARKET_INTERVAL_SECONDS,
        "russia": next_crawl,
        "china": next_crawl,
    }
    data["loop"] = {
        "running": True,
        "crawl_interval_seconds": CRAWL_INTERVAL_SECONDS,
        "crawl_base_interval_seconds": CRAWL_INTERVAL_SECONDS,
        "crawl_jitter_ratio": CRAWL_JITTER_RATIO,
        "active_crawl_interval_seconds": int(active_interval),
        "last_crawl_at": datetime.fromtimestamp(last_crawl).isoformat(timespec="seconds") if last_crawl else None,
        "next_crawl_at": datetime.fromtimestamp(next_crawl).isoformat(timespec="seconds") if next_crawl else None,
        "seconds_until_next_crawl": seconds_remaining,
        "platform_next": {
            key: {
                "next_at": datetime.fromtimestamp(value).isoformat(timespec="seconds") if value else None,
                "seconds_until_next": max(0, int(value - time.time())) if value else None,
            }
            for key, value in platform_next.items()
        },
    }
    data["last_updated"] = crawler.time_str()
    with open(STATUS_PATH, "w", encoding="utf-8") as f:
        json.dump(data, f, ensure_ascii=False, indent=2)


def has_pending_tagging():
    try:
        with connect() as conn:
            row = conn.execute(
                "SELECT COUNT(*) FROM tagging_queue WHERE status IN ('pending', 'tagging')"
            ).fetchone()
            return bool(row and row[0])
    except Exception:
        return True


def main():
    os.makedirs(BASE_DIR, exist_ok=True)
    with open(PID_PATH, "w", encoding="utf-8") as f:
        f.write(str(os.getpid()))
    signal.signal(signal.SIGTERM, handle_stop)
    if hasattr(signal, "SIGINT"):
        signal.signal(signal.SIGINT, handle_stop)

    init_db()
    log("[SYSTEM] realtime crawler loop started")
    last_crawl = 0.0
    active_interval = next_crawl_interval()
    next_crawl_at = time.time()
    write_loop_timing(last_crawl, next_crawl_at, active_interval)

    try:
        while running:
            config = crawler.load_config()
            now = time.time()
            did_crawl = False
            if os.path.exists(BACKFILL_LOCK_PATH):
                log("[SYSTEM] backfill lock active; scheduler crawl/tagging paused")
                write_loop_timing(last_crawl, next_crawl_at, active_interval)
                time.sleep(LOOP_SLEEP_SECONDS)
                continue
            if now >= next_crawl_at:
                crawler.collect_realtime(config)
                last_crawl = now
                active_interval = next_crawl_interval()
                next_crawl_at = last_crawl + active_interval
                did_crawl = True
            write_loop_timing(last_crawl, next_crawl_at, active_interval)
            if did_crawl or has_pending_tagging():
                crawler.run_tagger_and_analysis()
            time.sleep(LOOP_SLEEP_SECONDS)
    finally:
        data = {"platforms": {}}
        if os.path.exists(STATUS_PATH):
            try:
                with open(STATUS_PATH, "r", encoding="utf-8") as f:
                    data = json.load(f)
            except Exception:
                data = {"platforms": {}}
        data["loop"] = {
            "running": False,
            "crawl_interval_seconds": CRAWL_INTERVAL_SECONDS,
            "crawl_base_interval_seconds": CRAWL_INTERVAL_SECONDS,
            "crawl_jitter_ratio": CRAWL_JITTER_RATIO,
            "active_crawl_interval_seconds": int(active_interval),
            "last_crawl_at": datetime.fromtimestamp(last_crawl).isoformat(timespec="seconds") if last_crawl else None,
            "next_crawl_at": None,
            "seconds_until_next_crawl": 0,
            "platform_next": {},
        }
        data["last_updated"] = crawler.time_str()
        with open(STATUS_PATH, "w", encoding="utf-8") as f:
            json.dump(data, f, ensure_ascii=False, indent=2)
        if os.path.exists(PID_PATH):
            try:
                os.remove(PID_PATH)
            except OSError:
                pass
        log("[SYSTEM] realtime crawler loop stopped")


if __name__ == "__main__":
    main()
