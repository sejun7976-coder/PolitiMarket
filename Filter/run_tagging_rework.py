"""Reset-driven tagging worker used by the admin UI."""

from __future__ import annotations

import subprocess
import sys
from pathlib import Path

import realtime_tagger


BASE_DIR = Path(__file__).resolve().parents[1]
FILTER_DIR = BASE_DIR / "Filter"


def main() -> int:
    conn = realtime_tagger.connect()
    try:
        realtime_tagger.init_db(conn)
        model_ctx = realtime_tagger.load_required_model_context(conn)
        while True:
            realtime_tagger.repair_stale_tagging(conn)
            processed, cache_hits, inference_ms = realtime_tagger.process_once(conn, model_ctx)
            if processed:
                realtime_tagger.auto_review_excluded(conn)
                realtime_tagger.run_post_processors(conn)
                realtime_tagger.write_progress(conn, cache_hits, inference_ms)
                continue
            break
        realtime_tagger.write_progress(conn)
    finally:
        conn.close()

    subprocess.run([sys.executable, str(FILTER_DIR / "build_analysis.py")], cwd=FILTER_DIR, check=False)
    train_script = FILTER_DIR / "train_scheduler.py"
    if train_script.exists():
        subprocess.run([sys.executable, str(train_script)], cwd=FILTER_DIR, check=False)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
