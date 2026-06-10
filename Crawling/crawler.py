import argparse
import html
import json
import os
import queue
import random
import re
import subprocess
import sys
import threading
import time
import xml.etree.ElementTree as ET
from concurrent.futures import ThreadPoolExecutor
from datetime import datetime, timedelta, timezone
from email.utils import parsedate_to_datetime
from html.parser import HTMLParser
from urllib.parse import urljoin, urlparse
from zoneinfo import ZoneInfo

import requests

try:
    from bs4 import BeautifulSoup
except ImportError:
    BeautifulSoup = None

from db_utils import DB_PATH, connect, init_db, json_dumps, upsert_crawled_item, utc_now


BASE_DIR = os.path.dirname(os.path.abspath(__file__))
WEB_DIR = os.path.abspath(os.path.join(BASE_DIR, "../Web"))
FILTER_DIR = os.path.abspath(os.path.join(BASE_DIR, "../Filter"))
CONFIG_PATH = os.path.join(BASE_DIR, "crawler_config.json")
STATUS_PATH = os.path.join(BASE_DIR, "crawler_status.json")
LOG_PATH = os.path.join(BASE_DIR, "crawler_loop.log")
TEST_MODE_PATH = os.path.join(BASE_DIR, "test_mode.json")
BACKFILL_LOCK_PATH = os.path.join(BASE_DIR, "backfill.lock")

COLLECTOR_KEYS = ["x", "truth", "gov", "news", "thinktank", "axios", "market", "russia", "china"]
DEFAULT_X_API_CONFIG = {
    "bearer_token": "",
    "accounts": ["realDonaldTrump", "Jaemyung_Lee", "mofa_kr", "ROK_MND"],
    "queries": [],
    "recent_lookback_days": 1,
    "backfill_days": 7,
    "use_full_archive": False,
    "exclude_retweets": True,
    "exclude_replies": True,
}
DEFAULT_CONFIG = {
    "enabled": {key: True for key in COLLECTOR_KEYS},
    "x_api": DEFAULT_X_API_CONFIG,
}
X_ACCOUNT_COUNTRIES = {
    "realdonaldtrump": "US",
    "jaemyung_lee": "KR",
    "mofa_kr": "KR",
    "rok_mnd": "KR",
}
USER_AGENT = os.environ.get(
    "CRAWLER_USER_AGENT",
    "Mozilla/5.0 (Windows NT 10.0; Win64; x64) "
    "AppleWebKit/537.36 (KHTML, like Gecko) Chrome/125.0 Safari/537.36",
)
REQUEST_HEADERS = {
    "User-Agent": USER_AGENT,
    "Accept": "text/html,application/xhtml+xml,application/xml;q=0.9,application/rss+xml;q=0.8,text/xml;q=0.8,*/*;q=0.7",
    "Accept-Language": "ko-KR,ko;q=0.9,en-US;q=0.8,en;q=0.7",
    "Connection": "close",
}
JSON_REQUEST_HEADERS = dict(REQUEST_HEADERS)
JSON_REQUEST_HEADERS["Accept"] = "application/json, text/javascript, */*;q=0.8"
REQUEST_TIMEOUT = (4, 8)
ARTICLE_REQUEST_TIMEOUT = (4, 10)


def env_int(name, default, minimum=None):
    try:
        value = int(os.environ.get(name, default))
    except Exception:
        value = int(default)
    if minimum is not None:
        value = max(minimum, value)
    return value


LOOKBACK_DAYS = env_int("CRAWLER_LOOKBACK_DAYS", "2", minimum=1)
RETENTION_DAYS = env_int("CRAWLER_RETENTION_DAYS", str(LOOKBACK_DAYS), minimum=LOOKBACK_DAYS)
LOOKBACK_ENABLED = os.environ.get("CRAWLER_LOOKBACK_DAYS", "2") != "0"
LOG_SKIPPED_ARTICLES = os.environ.get("CRAWLER_LOG_SKIPPED_ARTICLES", "0") == "1"
LOCAL_TIMEZONE_NAME = os.environ.get("CRAWLER_DAY_TIMEZONE", "Asia/Seoul")
try:
    LOCAL_TIMEZONE = ZoneInfo(LOCAL_TIMEZONE_NAME)
except Exception:
    LOCAL_TIMEZONE = ZoneInfo("Asia/Seoul")
COLLECTOR_DEADLINES = {
    "x": 30,
    "gov": 20,
    "news": 20,
    "truth": 10,
    "thinktank": 30,
    "axios": 20,
    "china": 18,
    "russia": 10,
}
X_API_BASE = os.environ.get("X_API_BASE", "https://api.x.com/2")
X_RECENT_MAX_RESULTS = 100
X_FULL_ARCHIVE_MAX_RESULTS = 500
X_RECENT_MAX_DAYS = 7
X_RATE_LIMIT_MAX_SLEEP_SECONDS = env_int("X_RATE_LIMIT_MAX_SLEEP_SECONDS", "930", minimum=0)
ARTICLE_STOP_PATTERNS = [
    r"기사 제보와 오류 지적",
    r"<저작권자",
    r"저작권자\\(c\\)",
    r"원문 출처:",
    r"출처:",
    r"최신뉴스",
    r"뉴스 더보기",
    r"下一页",
    r"◎공감언론 뉴시스",
]
BOILERPLATE_PATTERNS = [
    r"^한국어 \\| 언어선택",
    r"^중국어 중국어 영어",
    r"^홈 정치 정부소식",
    r"^인민망 소개",
    r"^독자 제보",
]
EXTRACTION_BOILERPLATE_MARKERS = (
    "K-Artprice",
    "프라임뉴시스",
    "위클리뉴시스",
    "실시간 정치",
    "TV뉴시스",
    "제휴 콘텐츠",
    "공감언론 뉴시스 ::",
    "후속기사가 이어집니다",
    "많이 본 기사",
    "오늘의 헤드라인",
    "Subscribe",
    "Sign up",
)
GENERIC_ARTICLE_SELECTORS = (
    "#content_text_ALLBOX",
    "#postMainCont",
    ".post-txt",
    ".post-main-cont",
    ".detail-post",
    "article",
    "main",
    "[role='main']",
    ".entry-content",
    ".post-content",
    ".article-content",
    ".field--name-body",
    ".box_con",
    ".text_con",
    ".article",
    ".artDet",
    ".content",
    ".articleBody",
    ".view_cont",
    ".viewCont",
    ".cont_area",
    ".board_view",
    ".news_view",
    ".wb_txt",
    ".pic_c.gq_text",
    ".gq_text",
    ".p1_content",
)


class LinkParser(HTMLParser):
    def __init__(self):
        super().__init__()
        self.links = []
        self._href = None
        self._text = []

    def handle_starttag(self, tag, attrs):
        if tag.lower() == "a":
            attrs = dict(attrs)
            self._href = attrs.get("href")
            self._text = []

    def handle_data(self, data):
        if self._href is not None:
            self._text.append(data)

    def handle_endtag(self, tag):
        if tag.lower() == "a" and self._href is not None:
            text = " ".join(" ".join(self._text).split())
            self.links.append((self._href, html.unescape(text)))
            self._href = None
            self._text = []


class TextParser(HTMLParser):
    def __init__(self):
        super().__init__()
        self.parts = []
        self.skip = 0
        self.title = ""
        self._in_title = False

    def handle_starttag(self, tag, attrs):
        tag = tag.lower()
        if tag in ("script", "style", "noscript"):
            self.skip += 1
        if tag == "title":
            self._in_title = True
        if tag in ("p", "br", "div", "li", "h1", "h2"):
            self.parts.append("\n")

    def handle_endtag(self, tag):
        tag = tag.lower()
        if tag in ("script", "style", "noscript") and self.skip:
            self.skip -= 1
        if tag == "title":
            self._in_title = False

    def handle_data(self, data):
        if self.skip:
            return
        value = html.unescape(data.strip())
        if not value:
            return
        if self._in_title:
            self.title += value + " "
        self.parts.append(value)

    def text(self):
        return re.sub(r"\n{3,}", "\n\n", "\n".join(" ".join(self.parts).split("\n"))).strip()


def log(message):
    os.makedirs(BASE_DIR, exist_ok=True)
    with open(LOG_PATH, "a", encoding="utf-8", errors="replace") as f:
        f.write(f"[{time_str()}] {message}\n")
    print(message, flush=True)


def time_str():
    return time.strftime("%Y-%m-%d %H:%M:%S")


def load_config():
    config = json.loads(json.dumps(DEFAULT_CONFIG))
    if os.path.exists(CONFIG_PATH):
        try:
            with open(CONFIG_PATH, "r", encoding="utf-8") as f:
                saved = json.load(f)
            for key, value in (saved.get("enabled") or {}).items():
                if key in config["enabled"]:
                    config["enabled"][key] = bool(value)
            saved_x_api = saved.get("x_api") or {}
            if isinstance(saved_x_api, dict):
                config["x_api"].update(
                    {
                        key: saved_x_api.get(key, config["x_api"].get(key))
                        for key in config["x_api"]
                    }
                )
        except Exception as exc:
            log(f"[SYSTEM] config load failed: {exc}")
    token = os.environ.get("X_BEARER_TOKEN", "").strip()
    if token:
        config["x_api"]["bearer_token"] = token
    return config


def save_config(config):
    normalized = json.loads(json.dumps(DEFAULT_CONFIG))
    enabled = config.get("enabled") or {}
    for key in COLLECTOR_KEYS:
        if key in enabled:
            normalized["enabled"][key] = bool(enabled[key])
    x_api = config.get("x_api") or {}
    if isinstance(x_api, dict):
        normalized["x_api"].update(
            {
                key: x_api.get(key, normalized["x_api"].get(key))
                for key in normalized["x_api"]
            }
        )
    with open(CONFIG_PATH, "w", encoding="utf-8") as f:
        json.dump(normalized, f, ensure_ascii=False, indent=2)
    return normalized


def set_platform_status(platform, status, error="", count=0):
    data = {"platforms": {}}
    if os.path.exists(STATUS_PATH):
        try:
            with open(STATUS_PATH, "r", encoding="utf-8") as f:
                data = json.load(f)
        except Exception:
            data = {"platforms": {}}
    data.setdefault("platforms", {})
    data["platforms"][platform] = {
        "status": status,
        "last_run": time_str(),
        "error": error,
        "count": count,
    }
    data["last_updated"] = time_str()
    with open(STATUS_PATH, "w", encoding="utf-8") as f:
        json.dump(data, f, ensure_ascii=False, indent=2)


def fetch(url, timeout=REQUEST_TIMEOUT):
    last_exc = None
    for attempt in range(2):
        try:
            response = requests.get(url, headers=REQUEST_HEADERS, timeout=timeout)
            response.raise_for_status()
            break
        except requests.RequestException as exc:
            last_exc = exc
            status_code = getattr(getattr(exc, "response", None), "status_code", None)
            if status_code in {401, 403, 404, 410}:
                raise last_exc
            if attempt == 0:
                time.sleep(0.25)
                continue
            raise last_exc
    apparent = response.apparent_encoding or "utf-8"
    if not response.encoding or response.encoding.lower() in {"iso-8859-1", "ascii"}:
        response.encoding = apparent
    return response.text


def fetch_json(url, timeout=REQUEST_TIMEOUT, **kwargs):
    response = requests.get(url, headers=JSON_REQUEST_HEADERS, timeout=timeout, **kwargs)
    response.raise_for_status()
    return response.json()


def strip_html(value):
    if not value:
        return ""
    parser = TextParser()
    parser.feed(str(value))
    text = parser.text()
    if not text:
        text = re.sub(r"<[^>]+>", " ", str(value))
    return re.sub(r"\s+", " ", html.unescape(text)).strip()


def item_dedupe_key(item):
    url = re.sub(r"[?#].*$", "", str(item.get("url") or item.get("raw_url") or "")).strip().lower()
    if url:
        return f"url:{url}"
    title = re.sub(r"\s+", " ", str(item.get("title") or "")).strip().lower()
    content = re.sub(r"\s+", " ", str(item.get("content") or "")).strip().lower()
    return f"text:{title}|{content[:220]}"


def dedupe_items(items, limit=None):
    seen = set()
    unique = []
    for item in items:
        key = item_dedupe_key(item)
        if key in seen:
            continue
        seen.add(key)
        unique.append(item)
        if limit and len(unique) >= limit:
            break
    return unique


def is_probably_broken_korean(text, reference_text=""):
    sample = str(text or "")[:1200]
    reference = str(reference_text or "")
    if not re.search(r"[가-힣]", reference):
        return False
    hangul_count = len(re.findall(r"[가-힣]", sample))
    cjk_count = len(re.findall(r"[\u4e00-\u9fff]", sample))
    replacement_count = sample.count("\ufffd")
    if replacement_count >= 3:
        return True
    return hangul_count < 5 and cjk_count >= 20


def has_extraction_boilerplate(text):
    sample = re.sub(r"\s+", " ", str(text or "")).strip()[:900]
    return any(marker in sample for marker in EXTRACTION_BOILERPLATE_MARKERS)


def crawl_quality_payload(item, origin="", status="", error="", feed_content="", article_content=""):
    content = re.sub(r"\s+", " ", str(item.get("content") or "")).strip()
    title = re.sub(r"\s+", " ", str(item.get("title") or "")).strip()
    boilerplate = has_extraction_boilerplate(content) or any(
        marker in content[:500]
        for marker in ("언어선택", "뉴스 더보기", "기사 제보와 오류 지적")
    )
    content_len = len(content)
    score = 0.0
    if content_len >= 800:
        score = 1.0
    elif content_len >= 300:
        score = 0.82
    elif content_len >= 120:
        score = 0.58
    elif content_len >= 40:
        score = 0.35
    elif content_len >= 20:
        score = 0.2
    if not title:
        score -= 0.15
    if not item.get("url"):
        score -= 0.1
    if boilerplate:
        score -= 0.35
    if error and not content:
        score -= 0.2
    score = max(0.0, min(1.0, score))
    payload = {
        "validator": "crawler_quality_v1",
        "content_origin": origin,
        "crawl_status": status,
        "content_len": content_len,
        "title_len": len(title),
        "feed_content_len": len(feed_content or ""),
        "article_content_len": len(article_content or ""),
        "has_url": bool(item.get("url")),
        "has_boilerplate": boilerplate,
        "error": error or "",
    }
    return score, payload


def annotate_crawl_quality(item, origin="", status="", error="", feed_content="", article_content=""):
    score, payload = crawl_quality_payload(
        item,
        origin=origin,
        status=status,
        error=error,
        feed_content=feed_content,
        article_content=article_content,
    )
    item["content_origin"] = origin
    item["crawl_status"] = status
    item["crawl_error"] = error[:1000] if error else ""
    item["crawl_quality_score"] = score
    item["crawl_quality_json"] = json.dumps(payload, ensure_ascii=False)
    return item


def enrich_feed_item_with_article(item, source=""):
    """Always try the source URL, then choose the better body with audit metadata."""
    feed_title = item.get("title", "")
    feed_content = item.get("content", "")
    article_title = ""
    article_content = ""
    article_error = ""
    origin = "feed"
    status = "feed_only"
    url = item.get("url")

    if url:
        try:
            article_title, article_content = extract_article(
                url,
                feed_title,
                source=source,
                timeout=ARTICLE_REQUEST_TIMEOUT,
            )
        except requests.HTTPError as exc:
            code = getattr(exc.response, "status_code", None)
            article_error = f"HTTP {code}" if code else str(exc)
        except Exception as exc:
            article_error = str(exc)

    article_text = re.sub(r"\s+", " ", article_content or "").strip()
    feed_text = re.sub(r"\s+", " ", feed_content or "").strip()
    broken_article = is_probably_broken_korean(
        f"{article_title} {article_text}",
        f"{feed_title} {feed_text}",
    )
    article_boilerplate = has_extraction_boilerplate(article_text)
    article_good = has_enough_article_text(article_title or feed_title, article_text)
    if broken_article and not article_error:
        article_error = "article_text_mojibake"
    if article_boilerplate and not article_error:
        article_error = "article_text_boilerplate"
    article_short_usable = (
        source == "news"
        and len(article_text) >= 30
        and not has_extraction_boilerplate(feed_text)
        and len(article_text) >= len(feed_text)
    )
    if source == "news" and len(article_text) >= 30 and has_extraction_boilerplate(feed_text):
        article_short_usable = True

    if article_text and not broken_article and not article_boilerplate and (article_good or article_short_usable or len(article_text) >= max(len(feed_text), 80)):
        item["title"] = (article_title or feed_title)[:300]
        item["content"] = article_text[:3000]
        origin = "article"
        status = "article_extracted" if article_good else "article_short"
    elif feed_text:
        item["content"] = feed_text[:3000]
        origin = "feed_fallback" if article_error or article_text else "feed"
        status = "article_failed_feed_used" if article_error else "feed_used"
    else:
        item["content"] = feed_title[:3000]
        origin = "title_only"
        status = "no_body_title_only"

    return annotate_crawl_quality(
        item,
        origin=origin,
        status=status,
        error=article_error,
        feed_content=feed_content,
        article_content=article_content,
    )


def enrich_feed_items_parallel(items, source="", limit=None, max_workers=4):
    selected = items[:limit] if limit else list(items)
    if not selected:
        return []
    if len(selected) == 1:
        enriched = [enrich_feed_item_with_article(selected[0], source=source)]
    else:
        worker_count = max(1, min(max_workers, len(selected)))
        with ThreadPoolExecutor(max_workers=worker_count) as executor:
            enriched = list(executor.map(lambda row: enrich_feed_item_with_article(row, source=source), selected))
    return [
        item
        for item in enriched
        if item.get("crawl_status") != "no_body_title_only"
        and not is_title_only_content(item.get("title", ""), item.get("content", ""))
    ]


def clean_article_text(text, source=""):
    value = re.sub(r"\s+", " ", html.unescape(text or "")).strip()
    if source == "news":
        value = re.sub(r"\s*등록\s+\d{4}\.\d{2}\.\d{2}\s+\d{2}:\d{2}:\d{2}\s*작게\s+크게\s*", " ", value).strip()
        value = re.sub(r"후속기사가 이어집니다.*$", "", value).strip()
    for pattern in ARTICLE_STOP_PATTERNS:
        match = re.search(pattern, value)
        if match:
            if match.start() > 80 or source == "news":
                value = value[:match.start()].strip()
            elif source == "people":
                return ""
    for pattern in BOILERPLATE_PATTERNS:
        value = re.sub(pattern, "", value).strip()
    if source == "people":
        marker = re.search(r"\[인민망 한국어판[^\]]*\]", value)
        if marker:
            value = value[marker.start():].strip()
    return re.sub(r"\s+", " ", value).strip()


def parse_datetime(value):
    if not value:
        return datetime.now(timezone.utc).isoformat()
    raw = str(value).strip()
    try:
        parsed = parsedate_to_datetime(raw)
        if parsed.tzinfo is None:
            parsed = parsed.replace(tzinfo=timezone.utc)
        return parsed.astimezone(timezone.utc).isoformat()
    except Exception:
        pass
    try:
        normalized = raw.replace("Z", "+00:00")
        parsed = datetime.fromisoformat(normalized)
        if parsed.tzinfo is None:
            parsed = parsed.replace(tzinfo=timezone.utc)
        return parsed.astimezone(timezone.utc).isoformat()
    except Exception:
        return parse_iso_from_text(raw)


def parse_item_datetime(value):
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


def parse_people_url_datetime(url):
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


def lookback_start_local(days=None):
    days = LOOKBACK_DAYS if days is None else max(1, int(days))
    start_date = datetime.now(LOCAL_TIMEZONE).date() - timedelta(days=days - 1)
    return datetime.combine(start_date, datetime.min.time(), tzinfo=LOCAL_TIMEZONE)


def is_within_lookback_datetime(value, days=None):
    if not LOOKBACK_ENABLED:
        return True
    if value is None:
        return True
    if value.tzinfo is None:
        value = value.replace(tzinfo=timezone.utc)
    return value.astimezone(LOCAL_TIMEZONE) >= lookback_start_local(days)


def is_within_lookback_item(item, days=None):
    if not LOOKBACK_ENABLED:
        return True
    published = None
    if item.get("source") == "News_PeopleCN_KO":
        published = parse_people_url_datetime(item.get("url") or item.get("raw_url"))
    if published is None:
        published = parse_item_datetime(item.get("published_at") or item.get("created_at"))
    return is_within_lookback_datetime(published, days=days)


def filter_lookback_items(items):
    if not LOOKBACK_ENABLED:
        return list(items), 0
    kept_items = [item for item in items if is_within_lookback_item(item)]
    return kept_items, len(items) - len(kept_items)


def filter_items_within_days(items, days):
    kept_items = [item for item in items if is_within_lookback_item(item, days=days)]
    return kept_items, len(items) - len(kept_items)


def child_text(elem, names):
    wanted = {name.lower() for name in names}
    for child in list(elem):
        local = child.tag.split("}", 1)[-1].lower()
        if local not in wanted:
            continue
        if local == "link" and child.attrib.get("href"):
            return child.attrib.get("href", "").strip()
        return "".join(child.itertext()).strip()
    return ""


def parse_feed(feed_url, source, source_group, country, language, limit=10):
    xml_text = fetch(feed_url, timeout=REQUEST_TIMEOUT)
    root = ET.fromstring(xml_text.encode("utf-8"))
    nodes = root.findall(".//item")
    if not nodes:
        nodes = root.findall(".//{http://www.w3.org/2005/Atom}entry")

    items = []
    selected_nodes = nodes if limit is None else nodes[:limit]
    for node in selected_nodes:
        title = strip_html(child_text(node, ["title"]))
        link = child_text(node, ["link", "guid", "id"])
        description = strip_html(child_text(node, ["description", "summary", "content", "encoded"]))
        published = child_text(node, ["pubDate", "published", "updated", "dc:date"])
        if not title and not description:
            continue
        if link and not link.startswith("http"):
            link = urljoin(feed_url, link)
        content = description or title
        items.append(
            annotate_crawl_quality(
                {
                    "source": source,
                    "source_group": source_group,
                    "country": country,
                    "language": language,
                    "title": title[:300],
                    "content": content[:3000],
                    "url": link,
                    "raw_url": link,
                    "published_at": parse_datetime(published),
                },
                origin="feed",
                status="feed_parsed",
                feed_content=content,
            )
        )
    return items


def extract_links(page_url, html_text, patterns=None, limit=30):
    parser = LinkParser()
    parser.feed(html_text)
    seen = set()
    links = []
    for href, text in parser.links:
        if not href or not text:
            continue
        url = urljoin(page_url, href)
        if url in seen:
            continue
        if patterns and not any(re.search(pattern, url, re.I) for pattern in patterns):
            continue
        seen.add(url)
        links.append({"url": url, "title": text})
        if limit is not None and len(links) >= limit:
            break
    return links


GENERIC_LINK_TITLES = {
    "",
    "skip to main content",
    "topics",
    "regions",
    "banking",
    "economic outlook",
    "events",
    "research",
    "articles",
    "programs",
    "commentary",
    "newsletters",
    "for media",
    "about us",
}


def is_generic_link_title(title):
    normalized = re.sub(r"\s+", " ", html.unescape(title or "")).strip().lower()
    return normalized in GENERIC_LINK_TITLES


def is_thinktank_article_link(url, title):
    if is_generic_link_title(title):
        return False
    parsed = urlparse(url)
    path = parsed.path.rstrip("/")
    host = parsed.netloc.lower()
    if "piie.com" in host:
        return bool(
            re.search(r"/blogs/realtime-economics/20\d{2}/[^/]+$", path)
            or re.search(r"/research/piie-charts/20\d{2}/[^/]+$", path)
            or re.search(r"/publications/(policy-briefs|working-papers|piie-briefings)/20\d{2}/[^/]+$", path)
            or re.search(r"/events/20\d{2}/[^/]+$", path)
        )
    if "brookings.edu" in host:
        return bool(
            re.search(r"/articles/[^/]+$", path)
            or re.search(r"/events/[^/]+$", path)
        )
    if "csis.org" in host:
        return bool(re.search(r"/(analysis|events)/[^/]+$", path))
    return False


def has_enough_article_text(title, content):
    compact = re.sub(r"\s+", " ", content or "").strip()
    if len(compact) < 220:
        return False
    if is_title_only_content(title, compact):
        return False
    if is_generic_link_title(title):
        return False
    return True


def is_title_only_content(title, content):
    compact_title = re.sub(r"\s+", " ", html.unescape(title or "")).strip().lower()
    compact_content = re.sub(r"\s+", " ", html.unescape(content or "")).strip().lower()
    if not compact_content:
        return True
    if not compact_title:
        return False
    if len(compact_content) <= len(compact_title) + 20:
        return compact_content in compact_title or compact_title in compact_content
    return False


def extract_article(url, fallback_title="", source="", timeout=REQUEST_TIMEOUT):
    html_text = fetch(url, timeout=timeout)
    if BeautifulSoup is not None:
        soup = BeautifulSoup(html_text, "html.parser")
        for tag in soup(["script", "style", "noscript", "svg", "form", "header", "footer", "nav", "aside", "select"]):
            tag.decompose()
        for selector in [
            ".nav", ".footer", ".copyright", ".share", ".recommend", ".related",
            ".hotNews", ".menu", ".language", ".page_n", ".right-cont",
            ".tags-wrap", ".post-util", ".txt-another-lang", ".thumb-list",
            ".sns", ".util", ".breadcrumb", ".location",
        ]:
            for tag in soup.select(selector):
                tag.decompose()
        if source == "news":
            for selector in [
                ".quickArea", ".viewBottom", ".aside", ".article_photo", ".thumCont",
                ".photo", ".caption", ".ad", ".rankBox", ".rankMid", ".todayhead",
                ".botBox1", ".botBox2", ".botBox3", ".serviceMenu", ".viewBtn",
            ]:
                for tag in soup.select(selector):
                    tag.decompose()
        title = fallback_title or ""
        if not title:
            title_node = soup.select_one("meta[property='og:title'], meta[name='twitter:title']")
            if title_node and title_node.get("content"):
                title = title_node["content"]
        if not title:
            h1 = soup.find("h1")
            title = h1.get_text(" ", strip=True) if h1 else ""
        if not title and soup.title:
            title = soup.title.get_text(" ", strip=True)

        meta_description = ""
        description_node = soup.select_one("meta[name='description'], meta[property='og:description']")
        if description_node and description_node.get("content"):
            meta_description = clean_article_text(description_node["content"], source=source)

        def candidate_text(candidate):
            for br in candidate.find_all("br"):
                br.replace_with("\n")
            parts = []
            for node in candidate.find_all(["p", "li", "h2", "h3", "blockquote"]):
                value = node.get_text(" ", strip=True)
                if len(value) >= 20:
                    parts.append(value)
            full_text = candidate.get_text("\n", strip=True)
            if not parts:
                if full_text:
                    parts.append(full_text)
            text = " ".join(parts)
            if source == "news" and len(text) < 120 and len(full_text) > len(text):
                text = full_text
            return clean_article_text(text, source=source)

        if source == "people":
            for selector in [".wb_txt", ".article_text", ".pic_c.gq_text", ".gq_text", ".p1_content"]:
                for candidate in soup.select(selector):
                    text = candidate_text(candidate)
                    if len(text) >= 80:
                        return html.unescape(title).strip()[:300], text[:3000].strip()
            return html.unescape(title).strip()[:300], clean_article_text(title, source=source)[:3000].strip()

        if source == "news":
            short_text = ""
            for selector in [".viewer article", ".viewer", ".view", ".articleView"]:
                for candidate in soup.select(selector):
                    text = candidate_text(candidate)
                    if len(text) >= 80:
                        return html.unescape(title).strip()[:300], text[:3000].strip()
                    if len(text) >= 30 and not has_extraction_boilerplate(text):
                        short_text = text
            if short_text:
                return html.unescape(title).strip()[:300], short_text[:3000].strip()

        candidates = soup.select(", ".join(GENERIC_ARTICLE_SELECTORS))
        if not candidates:
            candidates = [soup]
        best_text = ""
        for candidate in candidates:
            text = candidate_text(candidate)
            if len(text) > len(best_text):
                best_text = text
        if len(best_text) >= 220:
            return html.unescape(title).strip()[:300], best_text[:3000].strip()
        if len(best_text) >= 80 and not is_title_only_content(title, best_text):
            return html.unescape(title).strip()[:300], best_text[:3000].strip()
        if len(meta_description) >= 120 and not is_title_only_content(title, meta_description):
            return html.unescape(title).strip()[:300], meta_description[:3000].strip()

    parser = TextParser()
    parser.feed(html_text)
    text = parser.text()
    title = fallback_title or " ".join(parser.title.split())
    if not title:
        title = text.split("\n", 1)[0][:120]
    content = clean_article_text(text, source=source)
    if is_title_only_content(title, content) and BeautifulSoup is not None:
        # Keep a useful snippet if the visible page text was mostly navigation.
        soup = BeautifulSoup(html_text, "html.parser")
        description_node = soup.select_one("meta[name='description'], meta[property='og:description']")
        if description_node and description_node.get("content"):
            description = clean_article_text(description_node["content"], source=source)
            if len(description) > len(content):
                content = description
    return title.strip(), content[:3000].strip()


def parse_iso_from_text(value):
    value = value or ""
    month_map = {
        "Jan": "01",
        "Feb": "02",
        "Mar": "03",
        "Apr": "04",
        "May": "05",
        "Jun": "06",
        "Jul": "07",
        "Aug": "08",
        "Sep": "09",
        "Oct": "10",
        "Nov": "11",
        "Dec": "12",
    }
    match = re.search(r"(Jan|Feb|Mar|Apr|May|Jun|Jul|Aug|Sep|Oct|Nov|Dec)\.?\s+(\d{1,2}),\s+(\d{4})(?:\s+(\d{1,2}):(\d{2}))?", value)
    if match:
        mon, day, year, hour, minute = match.groups()
        return f"{year}-{month_map[mon]}-{int(day):02d}T{int(hour or 0):02d}:{int(minute or 0):02d}:00Z"
    return datetime.now(timezone.utc).isoformat()


def collect_kremlin(limit=10):
    page = ""
    base = ""
    list_failures = 0
    for candidate in (
        "https://en.kremlin.ru/events/president/news",
        "https://www.en.kremlin.ru/events/president/news",
    ):
        try:
            page = fetch(candidate, timeout=(3, 5))
            base = candidate
            break
        except Exception:
            list_failures += 1
    if not page:
        log(f"[RUSSIA] skipped_unreachable_lists={list_failures}")
        return []
    links = extract_links(base, page, patterns=[r"/events/president/news/\d+$"], limit=limit)
    items = []
    article_failures = 0
    for link in links:
        url = str(link["url"] or "").replace("http://en.kremlin.ru", "https://en.kremlin.ru")
        try:
            title, content = extract_article(url, link["title"], timeout=(3, 5))
            published = parse_iso_from_text(content[:500])
            items.append(
                annotate_crawl_quality(
                    {
                        "source": "Gov_Kremlin",
                        "source_group": "russia",
                        "country": "RU",
                        "language": "en",
                        "title": title,
                        "content": content,
                        "url": url,
                        "raw_url": url,
                        "published_at": published,
                    },
                    origin="article",
                    status="article_extracted" if has_enough_article_text(title, content) else "article_short",
                    article_content=content,
                )
            )
        except Exception:
            article_failures += 1
    if article_failures:
        log(f"[RUSSIA] skipped_unreachable_articles={article_failures}")
    return items


def collect_gov_kr(limit=10):
    feeds = [
        ("Gov_KoreaPolicy", "https://www.korea.kr/rss/policy.xml"),
        ("Gov_KoreaBriefing", "https://www.korea.kr/rss/ebriefing.xml"),
        ("Gov_KoreaNetBriefing", "https://www.korea.net/koreanet/rss/government/briefing-room/109"),
        ("Gov_KoreaNetPolicy", "https://www.korea.net/koreanet/rss/news/3"),
    ]
    items = []
    failures = []
    per_feed = None if limit is None else max(1, limit // len(feeds) + 1)
    for source, feed_url in feeds:
        try:
            language = "en" if "korea.net" in feed_url else "ko"
            parsed_items = parse_feed(feed_url, source, "gov", "KR", language, per_feed)
            if source.startswith("Gov_KoreaNet"):
                parsed_items = [
                    item for item in parsed_items
                    if "articleId=" in str(item.get("url") or "")
                ]
            items.extend(parsed_items)
        except Exception as exc:
            failures.append(f"{feed_url}: {exc}")
    if failures and not items:
        for failure in failures:
            log(f"[GOV] feed failed {failure}")
    return enrich_feed_items_parallel(dedupe_items(items, limit), source="gov", limit=limit)


def collect_news_kr(limit=10):
    feeds = [
        ("News_NewsisPolitics", "https://nwww.newsis.com/RSS/politics.xml"),
        ("News_NewsisEconomy", "https://nwww.newsis.com/RSS/economy.xml"),
        ("News_NewsisWorld", "https://nwww.newsis.com/RSS/international.xml"),
    ]
    items = []
    per_feed = None if limit is None else max(1, limit // len(feeds) + 1)
    for source, feed_url in feeds:
        try:
            items.extend(parse_feed(feed_url, source, "news", "KR", "ko", per_feed))
        except Exception as exc:
            log(f"[NEWS] feed failed {feed_url}: {exc}")
    return enrich_feed_items_parallel(dedupe_items(items, limit), source="news", limit=limit)


def collect_thinktank(limit=12):
    pages = [
        ("ThinkTank_PIIE_Trade", "https://www.piie.com/research/trade-investment", "US", "en"),
        ("ThinkTank_PIIE_Research", "https://www.piie.com/research", "US", "en"),
        ("ThinkTank_Brookings_Economy", "https://www.brookings.edu/topics/u-s-economy/", "US", "en"),
        ("ThinkTank_Brookings_ForeignPolicy", "https://www.brookings.edu/programs/foreign-policy/", "US", "en"),
        ("ThinkTank_CSIS_Korea", "https://www.csis.org/programs/korea-chair", "US", "en"),
    ]
    patterns = [
        r"/blogs/realtime-economics/20\d{2}/",
        r"/research/piie-charts/20\d{2}/",
        r"/publications/(policy-briefs|working-papers|piie-briefings)/20\d{2}/",
        r"/articles/",
        r"/analysis/",
        r"/events/[^/?#]+",
    ]
    items = []
    seen = set()
    per_page = None if limit is None else max(1, limit // len(pages) + 1)
    for source, page_url, country, language in pages:
        try:
            page = fetch(page_url, timeout=REQUEST_TIMEOUT)
            page_host = urlparse(page_url).netloc.lower().removeprefix("www.")
            raw_link_limit = None if per_page is None else per_page * 8
            raw_links = extract_links(page_url, page, patterns=patterns, limit=raw_link_limit)
            links = []
            for link in raw_links:
                link_host = urlparse(link["url"]).netloc.lower().removeprefix("www.")
                if link_host != page_host:
                    continue
                if is_thinktank_article_link(link["url"], link["title"]):
                    links.append(link)
                if per_page is not None and len(links) >= per_page:
                    break
        except Exception as exc:
            log(f"[THINKTANK] list failed {page_url}: {exc}")
            continue
        for link in links:
            if link["url"] in seen or (limit is not None and len(items) >= limit):
                break
            seen.add(link["url"])
            try:
                title, content = extract_article(link["url"], link["title"])
                if is_title_only_content(title, content):
                    if LOG_SKIPPED_ARTICLES:
                        log(f"[THINKTANK] skipped title-only article {link['url']}")
                    continue
                status = "article_extracted" if has_enough_article_text(title, content) else "article_short"
                items.append(
                    annotate_crawl_quality(
                        {
                            "source": source,
                            "source_group": "thinktank",
                            "country": country,
                            "language": language,
                            "title": title[:300],
                            "content": content[:3000],
                            "url": link["url"],
                            "raw_url": link["url"],
                            "published_at": parse_iso_from_text(content[:700]),
                        },
                        origin="article",
                        status=status,
                        article_content=content,
                    )
                )
            except Exception as exc:
                log(f"[THINKTANK] article failed {link['url']}: {exc}")
    return items if limit is None else items[:limit]


def collect_axios(limit=8):
    feeds = [
        "https://api.axios.com/feed/top/",
        "https://api.axios.com/feed/",
    ]
    items = []
    seen = set()
    for feed_url in feeds:
        try:
            feed_limit = None if limit is None else limit * 3
            feed_items = parse_feed(feed_url, "Axios", "axios", "US", "en", feed_limit)
        except Exception as exc:
            log(f"[AXIOS] feed failed {feed_url}: {exc}")
            continue
        for item in feed_items:
            if item["url"] in seen or (limit is not None and len(items) >= limit):
                continue
            seen.add(item["url"])
            items.append(item)
    return enrich_feed_items_parallel(items, source="axios", limit=limit)


def normalize_x_lines(value):
    if isinstance(value, list):
        values = value
    else:
        values = re.split(r"[\n,]+", str(value or ""))
    normalized = []
    seen = set()
    for raw in values:
        item = str(raw or "").strip()
        if not item:
            continue
        key = item.lower()
        if key in seen:
            continue
        seen.add(key)
        normalized.append(item)
    return normalized


def x_config(config=None):
    config = config or load_config()
    saved = config.get("x_api") or {}
    merged = json.loads(json.dumps(DEFAULT_X_API_CONFIG))
    if isinstance(saved, dict):
        for key in merged:
            if key in saved:
                merged[key] = saved[key]
    token = os.environ.get("X_BEARER_TOKEN", "").strip()
    if token:
        merged["bearer_token"] = token
    merged["accounts"] = normalize_x_lines(merged.get("accounts"))
    merged["queries"] = normalize_x_lines(merged.get("queries"))
    if not merged["accounts"] and not merged["queries"]:
        merged["accounts"] = list(DEFAULT_X_API_CONFIG["accounts"])
    else:
        existing_accounts = {account.lower() for account in merged["accounts"]}
        for account in DEFAULT_X_API_CONFIG["accounts"]:
            if account.lower() not in existing_accounts:
                merged["accounts"].append(account)
                existing_accounts.add(account.lower())
    try:
        merged["recent_lookback_days"] = max(1, int(merged.get("recent_lookback_days") or 1))
    except Exception:
        merged["recent_lookback_days"] = 1
    try:
        merged["backfill_days"] = max(1, int(merged.get("backfill_days") or 7))
    except Exception:
        merged["backfill_days"] = 7
    merged["use_full_archive"] = bool(merged.get("use_full_archive"))
    merged["exclude_retweets"] = bool(merged.get("exclude_retweets"))
    merged["exclude_replies"] = bool(merged.get("exclude_replies"))
    return merged


def mask_secret(value):
    value = str(value or "")
    if len(value) <= 8:
        return "configured" if value else ""
    return f"{value[:4]}...{value[-4:]}"


def build_x_queries(config):
    cfg = x_config({"x_api": config})
    suffix_parts = []
    if cfg.get("exclude_retweets"):
        suffix_parts.append("-is:retweet")
    if cfg.get("exclude_replies"):
        suffix_parts.append("-is:reply")
    suffix = " ".join(suffix_parts)

    queries = []
    for account in cfg.get("accounts") or []:
        username = str(account).strip().lstrip("@")
        if not username:
            continue
        query = f"from:{username}"
        if suffix:
            query = f"{query} {suffix}"
        queries.append(query)
    for query in cfg.get("queries") or []:
        query = str(query).strip()
        if query:
            queries.append(query)
    return queries


def x_iso_time(value):
    if isinstance(value, datetime):
        dt = value
    else:
        dt = parse_item_datetime(value)
    if not dt:
        dt = datetime.now(timezone.utc)
    if dt.tzinfo is None:
        dt = dt.replace(tzinfo=timezone.utc)
    return dt.astimezone(timezone.utc).isoformat(timespec="seconds").replace("+00:00", "Z")


def x_api_request(endpoint, token, params, wait_on_rate_limit=False):
    url = f"{X_API_BASE.rstrip('/')}/{endpoint.lstrip('/')}"
    headers = {"Authorization": f"Bearer {token}", "User-Agent": USER_AGENT}
    while True:
        response = requests.get(url, headers=headers, params=params, timeout=(5, 30))
        if response.status_code == 429 and wait_on_rate_limit:
            reset_header = response.headers.get("x-rate-limit-reset")
            try:
                reset_at = int(reset_header or "0")
            except Exception:
                reset_at = 0
            sleep_seconds = max(5, reset_at - int(time.time()) + 2) if reset_at else 60
            sleep_seconds = min(sleep_seconds, X_RATE_LIMIT_MAX_SLEEP_SECONDS)
            if sleep_seconds <= 0:
                raise RuntimeError("X API rate limit reached and waiting is disabled by max sleep setting.")
            log(f"[X] rate limited; sleeping {sleep_seconds}s until reset")
            time.sleep(sleep_seconds)
            continue
        if response.status_code >= 400:
            raise RuntimeError(f"X API HTTP {response.status_code}: {response.text[:500]}")
        data = response.json()
        if isinstance(data, dict) and data.get("errors") and not data.get("data"):
            raise RuntimeError(f"X API returned errors: {json.dumps(data.get('errors'), ensure_ascii=False)[:500]}")
        return data


def x_tweets_to_items(payload, query=""):
    users = {
        str(user.get("id")): user
        for user in (payload.get("includes") or {}).get("users", [])
        if isinstance(user, dict)
    }
    items = []
    for tweet in payload.get("data") or []:
        if not isinstance(tweet, dict):
            continue
        text = strip_html(tweet.get("text") or "")
        if not text:
            continue
        author = users.get(str(tweet.get("author_id"))) or {}
        username = author.get("username") or tweet.get("author_id") or "unknown"
        country = X_ACCOUNT_COUNTRIES.get(str(username).lstrip("@").lower(), "")
        display_name = author.get("name") or username
        tweet_id = str(tweet.get("id") or "")
        url = f"https://x.com/{username}/status/{tweet_id}" if tweet_id else ""
        metrics = tweet.get("public_metrics") or {}
        quality_json = {
            "validator": "x_api_v2",
            "query": query,
            "tweet_id": tweet_id,
            "author_id": tweet.get("author_id"),
            "username": username,
            "public_metrics": metrics,
        }
        items.append(
            annotate_crawl_quality(
                {
                    "id": f"x_{tweet_id}" if tweet_id else "",
                    "source": f"X_@{username}",
                    "source_group": "x",
                    "country": country,
                    "language": tweet.get("lang") or "en",
                    "title": text[:160],
                    "content": text[:3000],
                    "url": url,
                    "raw_url": url,
                    "published_at": x_iso_time(tweet.get("created_at") or utc_now()),
                    "crawl_quality_json": json_dumps(quality_json),
                },
                origin="api",
                status="api_text",
                article_content=text,
            )
        )
        items[-1]["crawl_quality_json"] = json_dumps(quality_json)
        items[-1]["title"] = f"@{username}: {text[:140]}"
        items[-1]["content"] = f"{display_name} (@{username})\n{text}"[:3000]
    return items


def collect_x_api(config=None, limit=10, days=None, backfill=False):
    cfg = x_config(config)
    token = str(cfg.get("bearer_token") or "").strip()
    if not token:
        raise RuntimeError("X_BEARER_TOKEN or crawler_config.json x_api.bearer_token is required.")
    queries = build_x_queries(cfg)
    if not queries:
        raise RuntimeError("No X API accounts or queries configured.")

    days = max(1, int(days or (cfg["backfill_days"] if backfill else cfg["recent_lookback_days"])))
    use_full_archive = bool(cfg.get("use_full_archive") or days > X_RECENT_MAX_DAYS)
    endpoint = "tweets/search/all" if use_full_archive else "tweets/search/recent"
    max_results = X_FULL_ARCHIVE_MAX_RESULTS if use_full_archive else X_RECENT_MAX_RESULTS
    if not use_full_archive and days > X_RECENT_MAX_DAYS:
        days = X_RECENT_MAX_DAYS
    start_time = datetime.now(timezone.utc) - timedelta(days=days)
    end_time = datetime.now(timezone.utc) - timedelta(seconds=10)

    all_items = []
    per_query_limit = None if backfill or limit is None else max(1, int(limit))
    for query in queries:
        next_token = None
        query_count = 0
        while True:
            page_size = max_results
            if per_query_limit is not None:
                remaining = per_query_limit - query_count
                if remaining <= 0:
                    break
                page_size = max(10, min(max_results, remaining))
            params = {
                "query": query,
                "max_results": str(page_size),
                "start_time": x_iso_time(start_time),
                "end_time": x_iso_time(end_time),
                "tweet.fields": "id,text,created_at,lang,author_id,conversation_id,public_metrics,entities,referenced_tweets,source,possibly_sensitive",
                "expansions": "author_id",
                "user.fields": "id,name,username,verified",
            }
            if next_token:
                params["next_token"] = next_token
            payload = x_api_request(endpoint, token, params, wait_on_rate_limit=backfill)
            items = x_tweets_to_items(payload, query=query)
            all_items.extend(items)
            query_count += len(items)
            meta = payload.get("meta") or {}
            next_token = meta.get("next_token")
            if not next_token:
                break
            if not backfill and per_query_limit is not None and query_count >= per_query_limit:
                break
    return dedupe_items(all_items, limit=None if backfill else limit)


def collect_truth_social(limit=10):
    lookup = fetch_json(
        "https://truthsocial.com/api/v1/accounts/lookup",
        timeout=REQUEST_TIMEOUT,
        params={"acct": "realDonaldTrump"},
    )
    account_id = lookup.get("id")
    if not account_id:
        raise RuntimeError("Truth Social account lookup did not return an id")
    statuses = []
    max_id = None
    page_size = 40 if limit is None else max(1, min(40, int(limit)))
    while True:
        params = {"exclude_replies": "true", "limit": str(page_size)}
        if max_id:
            params["max_id"] = max_id
        page = fetch_json(
            f"https://truthsocial.com/api/v1/accounts/{account_id}/statuses",
            timeout=REQUEST_TIMEOUT,
            params=params,
        )
        if not page:
            break
        statuses.extend(page)
        if limit is not None or len(page) < page_size:
            break
        last_id = str(page[-1].get("id") or "")
        if not last_id or last_id == max_id:
            break
        max_id = last_id
    items = []
    selected_statuses = statuses if limit is None else statuses[:limit]
    for status in selected_statuses:
        content = strip_html(status.get("content", ""))
        if not content:
            continue
        published = status.get("created_at") or datetime.now(timezone.utc).isoformat()
        items.append(
            annotate_crawl_quality(
                {
                    "source": "TruthSocial_DonaldTrump",
                    "source_group": "truth",
                    "country": "US",
                    "language": "en",
                    "title": content[:120],
                    "content": content[:3000],
                    "url": status.get("url") or status.get("uri") or "",
                    "raw_url": status.get("url") or status.get("uri") or "",
                    "published_at": parse_datetime(published),
                },
                origin="api",
                status="api_text",
                article_content=content,
            )
        )
    return items


def collect_people_ko(limit=12, days=None):
    start_urls = [
        "https://kr.people.com.cn/",
        "https://kr.people.com.cn/203280/index.html",
        "https://kr.people.com.cn/203281/index.html",
    ]
    links = []
    seen = set()
    list_failures = 0
    for start_url in start_urls:
        try:
            page = fetch(start_url)
        except Exception:
            list_failures += 1
            continue
        link_limit = None if limit is None else limit * 5
        for link in extract_links(start_url, page, patterns=[r"/n3/\d{4}/\d{4}/c\d+-\d+\.html"], limit=link_limit):
            url = str(link["url"] or "").replace("http://kr.people.com.cn", "https://kr.people.com.cn")
            people_published = parse_people_url_datetime(url)
            if people_published and not is_within_lookback_datetime(people_published, days=days):
                continue
            if url not in seen:
                seen.add(url)
                links.append({**link, "url": url})
            if limit is not None and len(links) >= limit:
                break
        if limit is not None and len(links) >= limit:
            break
    if list_failures:
        log(f"[CHINA] skipped_unreachable_lists={list_failures}")

    items = []
    selected_links = links if limit is None else links[:limit]
    article_failures = 0
    for link in selected_links:
        try:
            title, content = extract_article(link["url"], link["title"], source="people")
            if is_title_only_content(title, content):
                if LOG_SKIPPED_ARTICLES:
                    log(f"[CHINA] skipped title-only article {link['url']}")
                continue
            people_published = parse_people_url_datetime(link["url"])
            published = people_published.isoformat() if people_published else parse_iso_from_text(content[:700])
            items.append(
                annotate_crawl_quality(
                    {
                        "source": "News_PeopleCN_KO",
                        "source_group": "china",
                        "country": "CN",
                        "language": "ko",
                        "title": title,
                        "content": content,
                        "url": link["url"],
                        "raw_url": link["url"],
                        "published_at": published,
                    },
                    origin="article",
                    status="article_extracted" if has_enough_article_text(title, content) else "article_short",
                    article_content=content,
                )
            )
        except Exception:
            article_failures += 1
    if article_failures:
        log(f"[CHINA] skipped_unreachable_articles={article_failures}")
    return items


def import_existing_data_once():
    data_path = os.path.join(WEB_DIR, "data.js")
    if not os.path.exists(data_path):
        return 0
    with connect() as conn:
        existing = conn.execute("SELECT COUNT(*) FROM crawled_items").fetchone()[0]
    if existing:
        return 0
    try:
        with open(data_path, "r", encoding="utf-8", errors="replace") as f:
            text = f.read()
        match = re.search(r"window\.crawledPulseData\s*=\s*(\[.*?\]);", text, re.S)
        if not match:
            return 0
        rows = json.loads(match.group(1))
    except Exception as exc:
        log(f"[SYSTEM] legacy import failed: {exc}")
        return 0
    count = 0
    for row in rows:
        source = row.get("source", "")
        row.setdefault("source_group", "legacy")
        if source.startswith("Gov_"):
            row["source_group"] = "gov"
        elif source.startswith("News_"):
            row["source_group"] = "news"
        elif source.startswith("TruthSocial_"):
            row["source_group"] = "truth"
        elif source.startswith("X_"):
            row["source_group"] = "x"
        text_value = (row.get("title") or "") + (row.get("content") or "")
        row.setdefault("language", "ko" if re.search(r"[가-힣]", text_value) else "en")
        result = upsert_crawled_item(row)
        count += 1 if result["created"] else 0
    if count:
        log(f"[SYSTEM] imported {count} existing display items into realtime DB")
    return count


def preserve_market_data():
    path = os.path.join(WEB_DIR, "data.js")
    if not os.path.exists(path):
        return {}
    try:
        text = open(path, "r", encoding="utf-8", errors="replace").read()
        match = re.search(r"window\.marketData\s*=\s*(\{.*?\});", text, re.S)
        return json.loads(match.group(1)) if match else {}
    except Exception:
        return {}


def purge_old_items():
    if not LOOKBACK_ENABLED:
        return 0
    init_db()
    with connect() as conn:
        rows = conn.execute(
            """
            SELECT id, content_hash, source, url, raw_url, published_at, created_at
              FROM crawled_items
             WHERE COALESCE(source_group, '') <> 'test'
            """
        ).fetchall()
        old_rows = [
            dict(row)
            for row in rows
            if not is_within_lookback_item(dict(row), days=RETENTION_DAYS)
        ]
        if not old_rows:
            return 0
        old_ids = [row["id"] for row in old_rows]
        old_hashes = [row["content_hash"] for row in old_rows if row.get("content_hash")]
        id_placeholders = ",".join("?" for _ in old_ids)
        conn.execute(f"DELETE FROM tag_results WHERE item_id IN ({id_placeholders})", old_ids)
        conn.execute(f"DELETE FROM tagging_queue WHERE item_id IN ({id_placeholders})", old_ids)
        conn.execute(f"DELETE FROM ner_filter_events WHERE item_id IN ({id_placeholders})", old_ids)
        conn.execute(f"DELETE FROM label_feedback WHERE item_id IN ({id_placeholders})", old_ids)
        conn.execute(f"DELETE FROM item_embeddings WHERE item_id IN ({id_placeholders})", old_ids)
        conn.execute(f"DELETE FROM dedup_group_members WHERE item_id IN ({id_placeholders})", old_ids)
        conn.execute(f"DELETE FROM llm_excluded_reviews WHERE item_id IN ({id_placeholders})", old_ids)
        if old_hashes:
            hash_placeholders = ",".join("?" for _ in old_hashes)
            conn.execute(f"DELETE FROM tag_cache WHERE content_hash IN ({hash_placeholders})", old_hashes)
            conn.execute(f"DELETE FROM label_feedback WHERE content_hash IN ({hash_placeholders})", old_hashes)
            conn.execute(f"DELETE FROM ner_filter_events WHERE content_hash IN ({hash_placeholders})", old_hashes)
            conn.execute(f"DELETE FROM llm_excluded_reviews WHERE content_hash IN ({hash_placeholders})", old_hashes)
        conn.execute(f"DELETE FROM crawled_items WHERE id IN ({id_placeholders})", old_ids)
        conn.execute(
            """
            DELETE FROM dedup_groups
             WHERE group_id NOT IN (SELECT DISTINCT group_id FROM dedup_group_members)
            """
        )
        conn.commit()
        return len(old_rows)


def purge_items_outside_days(days):
    if not LOOKBACK_ENABLED:
        return 0
    init_db()
    days = max(1, int(days))
    with connect() as conn:
        rows = conn.execute(
            """
            SELECT id, content_hash, source, url, raw_url, published_at, created_at
              FROM crawled_items
             WHERE COALESCE(source_group, '') <> 'test'
            """
        ).fetchall()
        old_rows = [
            dict(row)
            for row in rows
            if not is_within_lookback_item(dict(row), days=days)
        ]
        if not old_rows:
            return 0
        old_ids = [row["id"] for row in old_rows]
        old_hashes = [row["content_hash"] for row in old_rows if row.get("content_hash")]
        id_placeholders = ",".join("?" for _ in old_ids)
        for table in (
            "tag_results",
            "tagging_queue",
            "ner_filter_events",
            "label_feedback",
            "item_embeddings",
            "dedup_group_members",
            "llm_excluded_reviews",
        ):
            conn.execute(f"DELETE FROM {table} WHERE item_id IN ({id_placeholders})", old_ids)
        if old_hashes:
            hash_placeholders = ",".join("?" for _ in old_hashes)
            for table in ("tag_cache", "label_feedback", "ner_filter_events", "llm_excluded_reviews"):
                conn.execute(f"DELETE FROM {table} WHERE content_hash IN ({hash_placeholders})", old_hashes)
        conn.execute(f"DELETE FROM crawled_items WHERE id IN ({id_placeholders})", old_ids)
        conn.execute(
            """
            DELETE FROM dedup_groups
             WHERE group_id NOT IN (SELECT DISTINCT group_id FROM dedup_group_members)
            """
        )
        conn.commit()
        return len(old_rows)


def export_data_js(limit=160, days=None):
    os.makedirs(WEB_DIR, exist_ok=True)
    init_db()
    test_only = False
    try:
        if os.path.exists(TEST_MODE_PATH):
            with open(TEST_MODE_PATH, "r", encoding="utf-8") as f:
                test_only = bool(json.load(f).get("active"))
    except Exception:
        test_only = False
    query_limit = max(limit * 10, 500) if LOOKBACK_ENABLED and not test_only else max(limit * 3, limit)
    with connect() as conn:
        rows = conn.execute(
            """
            SELECT c.id, c.source, c.title, c.content, c.published_at, c.url, c.raw_url,
                   c.language, c.country, c.source_group, c.content_hash,
                   c.content_origin, c.crawl_status, c.crawl_error,
                   c.crawl_quality_score, c.crawl_quality_json, c.last_seen_at
             FROM crawled_items c
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
             ORDER BY COALESCE(c.published_at, c.created_at) DESC, c.created_at DESC
             LIMIT ?
            """,
            (1 if test_only else 0, 1 if test_only else 0, query_limit),
        ).fetchall()
    filtered_rows = [row for row in [dict(row) for row in rows] if is_within_lookback_item(row, days=days)]
    news = dedupe_items(filtered_rows, limit)
    market = preserve_market_data()
    body = "// Automatically generated by Crawling/crawler.py. Do not edit.\n"
    body += "window.crawledPulseData = " + json_dumps(news) + ";\n"
    body += "window.marketData = " + json_dumps(market) + ";\n"
    with open(os.path.join(WEB_DIR, "data.js"), "w", encoding="utf-8") as f:
        f.write(body)
    return len(news)


def collect_realtime(config, limit=10):
    init_db()
    purged = purge_old_items()
    if purged:
        log(f"[SYSTEM] purged_old_items={purged} retention_days={RETENTION_DAYS}")
    if os.environ.get("CRAWLER_IMPORT_LEGACY") == "1":
        import_existing_data_once()
    totals = {}

    def run_with_deadline(func, seconds, **kwargs):
        result_queue = queue.Queue(maxsize=1)

        def target():
            try:
                result_queue.put(("ok", func(**kwargs)))
            except Exception as exc:
                result_queue.put(("error", exc))

        thread = threading.Thread(target=target, daemon=True)
        thread.start()
        thread.join(seconds)
        if thread.is_alive():
            raise TimeoutError(f"collector exceeded {seconds}s deadline")
        try:
            status, payload = result_queue.get_nowait()
        except queue.Empty as exc:
            raise RuntimeError("collector finished without a result") from exc
        if status == "error":
            raise payload
        return payload

    def run_collector(platform, label, collect_fn):
        if not config["enabled"].get(platform, True):
            set_platform_status(platform, "disabled", "사용자 설정으로 비활성화됨")
            return
        try:
            raw_items = run_with_deadline(collect_fn, COLLECTOR_DEADLINES.get(platform, 20), limit=limit)
            items, skipped_outside_window = filter_lookback_items(raw_items)
            results = [upsert_crawled_item(item) for item in items]
            created = sum(1 for result in results if result.get("created"))
            updated = sum(1 for result in results if result.get("updated"))
            totals[platform] = created
            status = "online" if items else "partial"
            error = "" if items else f"No {label} items found in last {LOOKBACK_DAYS} days ({LOCAL_TIMEZONE_NAME})"
            set_platform_status(platform, status, error, len(items))
            log(f"[{label}] collected={len(raw_items)} window={len(items)} skipped_outside_window={skipped_outside_window} new={created} updated={updated}")
        except Exception as exc:
            set_platform_status(platform, "offline", str(exc))
            log(f"[{label}] failed: {exc}")

    run_collector("gov", "GOV", collect_gov_kr)
    run_collector("news", "NEWS", collect_news_kr)
    run_collector("thinktank", "THINKTANK", collect_thinktank)
    run_collector("axios", "AXIOS", collect_axios)
    run_collector("truth", "TRUTH", collect_truth_social)

    if config["enabled"].get("x", True):
        if not str(x_config(config).get("bearer_token") or "").strip():
            set_platform_status("x", "partial", "X API Bearer Token not configured", 0)
            log("[X] skipped: X API Bearer Token not configured")
        else:
            try:
                raw_items = run_with_deadline(
                    lambda limit: collect_x_api(config, limit=limit, backfill=False),
                    COLLECTOR_DEADLINES["x"],
                    limit=limit,
                )
                items, skipped_outside_window = filter_lookback_items(raw_items)
                results = [upsert_crawled_item(item) for item in items]
                created = sum(1 for result in results if result.get("created"))
                updated = sum(1 for result in results if result.get("updated"))
                totals["x"] = created
                status = "online" if items else "partial"
                error = "" if items else "No X API items found in the configured lookback window"
                set_platform_status("x", status, error, len(items))
                log(f"[X] api_collected={len(raw_items)} window={len(items)} skipped_outside_window={skipped_outside_window} new={created} updated={updated}")
            except Exception as exc:
                set_platform_status("x", "offline", str(exc))
                log(f"[X] API failed: {exc}")
    else:
        set_platform_status("x", "disabled", "Disabled by user setting")

    if config["enabled"].get("russia", True):
        try:
            raw_items = run_with_deadline(collect_kremlin, COLLECTOR_DEADLINES["russia"], limit=limit)
            items, skipped_outside_window = filter_lookback_items(raw_items)
            results = [upsert_crawled_item(item) for item in items]
            created = sum(1 for result in results if result.get("created"))
            updated = sum(1 for result in results if result.get("updated"))
            totals["russia"] = created
            set_platform_status("russia", "online" if items else "partial", "" if items else f"No Kremlin items found in last {LOOKBACK_DAYS} days ({LOCAL_TIMEZONE_NAME})", len(items))
            log(f"[RUSSIA] collected={len(raw_items)} window={len(items)} skipped_outside_window={skipped_outside_window} new={created} updated={updated}")
        except Exception as exc:
            set_platform_status("russia", "offline", str(exc))
            log(f"[RUSSIA] failed: {exc}")
    else:
        set_platform_status("russia", "disabled", "사용자 설정으로 비활성화됨")

    if config["enabled"].get("china", True):
        try:
            raw_items = run_with_deadline(collect_people_ko, COLLECTOR_DEADLINES["china"], limit=limit)
            items, skipped_outside_window = filter_lookback_items(raw_items)
            results = [upsert_crawled_item(item) for item in items]
            created = sum(1 for result in results if result.get("created"))
            updated = sum(1 for result in results if result.get("updated"))
            totals["china"] = created
            set_platform_status("china", "online" if items else "partial", "" if items else f"No People's Daily Korean items found in last {LOOKBACK_DAYS} days ({LOCAL_TIMEZONE_NAME})", len(items))
            log(f"[CHINA] collected={len(raw_items)} window={len(items)} skipped_outside_window={skipped_outside_window} new={created} updated={updated}")
        except Exception as exc:
            set_platform_status("china", "offline", str(exc))
            log(f"[CHINA] failed: {exc}")
    else:
        set_platform_status("china", "disabled", "사용자 설정으로 비활성화됨")

    if config["enabled"].get("market", True):
        set_platform_status("market", "online", "시장 데이터는 기존 data.js 값을 보존합니다.")
    else:
        set_platform_status("market", "disabled", "사용자 설정으로 비활성화됨")
    export_data_js()
    return totals


def collect_x_backfill(config, days=None, run_analysis=True):
    init_db()
    cfg = x_config(config)
    days = max(1, int(days or cfg.get("backfill_days") or 7))
    if not str(cfg.get("bearer_token") or "").strip():
        set_platform_status("x", "partial", "X API Bearer Token not configured", 0)
        log(f"[X] backfill skipped: X API Bearer Token not configured; days={days}")
        return {"collected": 0, "created": 0, "updated": 0, "days": days, "skipped": "missing_token"}
    raw_items = collect_x_api(config, limit=None, days=days, backfill=True)
    results = [upsert_crawled_item(item) for item in raw_items]
    created = sum(1 for result in results if result.get("created"))
    updated = sum(1 for result in results if result.get("updated"))
    set_platform_status("x", "online" if raw_items else "partial", "" if raw_items else "No X API backfill items found", len(raw_items))
    log(f"[X] backfill_days={days} collected={len(raw_items)} new={created} updated={updated}")
    export_data_js(limit=1000, days=days)
    if run_analysis:
        run_tagger_and_analysis()
    return {"collected": len(raw_items), "created": created, "updated": updated, "days": days}


def collect_backfill(config, days=None, platforms=None, run_analysis=True, respect_enabled=False):
    init_db()
    days = max(1, int(days or x_config(config).get("backfill_days") or LOOKBACK_DAYS))
    os.makedirs(BASE_DIR, exist_ok=True)
    with open(BACKFILL_LOCK_PATH, "w", encoding="utf-8") as f:
        f.write(json.dumps({"days": days, "started_at": time_str()}, ensure_ascii=False))
    requested_platforms = set(platforms or [])
    platform_order = ["gov", "news", "thinktank", "axios", "truth", "x", "russia", "china"]
    collectors = {
        "gov": ("GOV", collect_gov_kr),
        "news": ("NEWS", collect_news_kr),
        "thinktank": ("THINKTANK", collect_thinktank),
        "axios": ("AXIOS", collect_axios),
        "truth": ("TRUTH", collect_truth_social),
        "russia": ("RUSSIA", collect_kremlin),
        "china": ("CHINA", collect_people_ko),
    }
    totals = {}
    errors = {}
    backfill_deadlines = {
        "gov": 60,
        "news": 90,
        "thinktank": 120,
        "axios": 60,
        "truth": 30,
        "x": 1800,
        "russia": 60,
        "china": 60,
    }

    def run_backfill_collector(platform, func, **kwargs):
        result_queue = queue.Queue(maxsize=1)

        def target():
            try:
                result_queue.put(("ok", func(**kwargs)))
            except Exception as exc:
                result_queue.put(("error", exc))

        thread = threading.Thread(target=target, daemon=True)
        thread.start()
        thread.join(backfill_deadlines.get(platform, 60))
        if thread.is_alive():
            raise TimeoutError(f"{platform} backfill exceeded {backfill_deadlines.get(platform, 60)}s deadline")
        try:
            status, payload = result_queue.get_nowait()
        except queue.Empty as exc:
            raise RuntimeError(f"{platform} backfill finished without a result") from exc
        if status == "error":
            raise payload
        return payload

    def should_run(platform):
        if requested_platforms and platform not in requested_platforms:
            return False
        if respect_enabled:
            return config["enabled"].get(platform, True)
        return True

    try:
        for platform in platform_order:
            if not should_run(platform):
                set_platform_status(platform, "disabled", "Skipped by backfill configuration")
                continue
            try:
                if platform == "x":
                    if not str(x_config(config).get("bearer_token") or "").strip():
                        set_platform_status("x", "partial", "X API Bearer Token not configured", 0)
                        log(f"[X] backfill skipped: X API Bearer Token not configured; days={days}")
                        totals[platform] = {
                            "collected": 0,
                            "kept": 0,
                            "created": 0,
                            "updated": 0,
                            "skipped_outside_window": 0,
                            "skipped": "missing_token",
                        }
                        continue
                    raw_items = run_backfill_collector("x", collect_x_api, config=config, limit=None, days=days, backfill=True)
                    items = raw_items
                    skipped_outside_window = 0
                    label = "X"
                else:
                    label, collect_fn = collectors[platform]
                    if platform == "china":
                        raw_items = run_backfill_collector(platform, collect_fn, limit=None, days=days)
                    else:
                        raw_items = run_backfill_collector(platform, collect_fn, limit=None)
                    items, skipped_outside_window = filter_items_within_days(raw_items, days)
                results = [upsert_crawled_item(item) for item in items]
                created = sum(1 for result in results if result.get("created"))
                updated = sum(1 for result in results if result.get("updated"))
                totals[platform] = {
                    "collected": len(raw_items),
                    "kept": len(items),
                    "created": created,
                    "updated": updated,
                    "skipped_outside_window": skipped_outside_window,
                }
                set_platform_status(platform, "online" if items else "partial", "" if items else f"No {label} items found in last {days} days", len(items))
                log(f"[{label}] backfill_days={days} collected={len(raw_items)} kept={len(items)} skipped_outside_window={skipped_outside_window} new={created} updated={updated}")
            except Exception as exc:
                errors[platform] = str(exc)
                set_platform_status(platform, "offline", str(exc))
                log(f"[{platform.upper()}] backfill failed: {exc}")

        purged = purge_items_outside_days(days)
        if purged:
            log(f"[SYSTEM] backfill_days={days} purged_outside_window={purged}")
        export_data_js(limit=1000, days=days)
        if run_analysis:
            run_tagger_and_analysis()
        return {"days": days, "totals": totals, "errors": errors, "purged_outside_window": purged}
    finally:
        try:
            if os.path.exists(BACKFILL_LOCK_PATH):
                os.remove(BACKFILL_LOCK_PATH)
        except OSError:
            pass


def run_tagger_and_analysis():
    scripts = [
        os.path.join(FILTER_DIR, "realtime_tagger.py"),
        os.path.join(FILTER_DIR, "build_analysis.py"),
        os.path.join(FILTER_DIR, "train_scheduler.py"),
    ]
    for script in scripts:
        if not os.path.exists(script):
            continue
        cmd = [sys.executable, script, "--once"] if script.endswith("realtime_tagger.py") else [sys.executable, script]
        try:
            creationflags = subprocess.CREATE_NO_WINDOW if sys.platform == "win32" else 0
            env = os.environ.copy()
            if script.endswith("realtime_tagger.py"):
                env.setdefault("FILTER_BATCH_SIZE", "256")
            subprocess.run(cmd, cwd=os.path.dirname(script), check=False, creationflags=creationflags, env=env)
        except Exception as exc:
            log(f"[NLP] failed running {os.path.basename(script)}: {exc}")


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--once", action="store_true")
    parser.add_argument("--market-only", action="store_true")
    parser.add_argument("--backfill-days", type=int, default=0)
    parser.add_argument("--x-backfill-days", type=int, default=0)
    parser.add_argument("--x-only", action="store_true")
    parser.add_argument("--no-analysis", action="store_true")
    parser.add_argument("--limit", type=int, default=10)
    args = parser.parse_args()
    config = load_config()
    if args.market_only:
        export_data_js()
        log("[MARKET] market-only run preserved existing market data")
        return
    if args.backfill_days:
        collect_backfill(config, days=args.backfill_days, run_analysis=not args.no_analysis)
        return
    if args.x_backfill_days:
        collect_x_backfill(config, days=args.x_backfill_days, run_analysis=not args.no_analysis)
        return
    if args.x_only:
        items = collect_x_api(config, limit=args.limit, backfill=False)
        results = [upsert_crawled_item(item) for item in items]
        created = sum(1 for result in results if result.get("created"))
        updated = sum(1 for result in results if result.get("updated"))
        set_platform_status("x", "online" if items else "partial", "" if items else "No X API items found", len(items))
        log(f"[X] one_shot collected={len(items)} new={created} updated={updated}")
        export_data_js()
        if not args.no_analysis:
            run_tagger_and_analysis()
        return
    collect_realtime(config, limit=args.limit)
    run_tagger_and_analysis()


if __name__ == "__main__":
    main()
