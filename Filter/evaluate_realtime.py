import sys
import os
import sqlite3
import json
import time
import re
import datetime
from pathlib import Path
import numpy as np
import requests

# Reconfigure stdout to UTF-8 to prevent encoding issues when printing
sys.stdout.reconfigure(encoding='utf-8')

# Paths
PROJECT_ROOT = Path("c:/Users/sejun/Desktop/Project/NLP ver.2")
DB_PATH = PROJECT_ROOT / "Crawling" / "crawler.db"
OUTPUT_DIR = PROJECT_ROOT / "Filter" / "output"
OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

# Add to path
sys.path.insert(0, str(PROJECT_ROOT))
sys.path.insert(0, str(PROJECT_ROOT / "Filter"))

import realtime_tagger
from sentence_transformers import SentenceTransformer
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity

# Load LLM config for OpenAI API key
llm_config_path = PROJECT_ROOT / "Crawling" / "llm_config.json"
with open(llm_config_path, "r", encoding="utf-8") as f:
    llm_config = json.load(f)
OPENAI_API_KEY = llm_config["openai_api_key"]

def call_llm(prompt: str, system_prompt: str = "You are a helpful assistant.") -> str:
    headers = {
        "Authorization": f"Bearer {OPENAI_API_KEY}",
        "Content-Type": "application/json"
    }
    payload = {
        "model": "gpt-4o-mini",
        "messages": [
            {"role": "system", "content": system_prompt},
            {"role": "user", "content": prompt}
        ],
        "temperature": 0.0,
        "response_format": {"type": "json_object"}
    }
    # Retry logic
    for attempt in range(3):
        try:
            r = requests.post("https://api.openai.com/v1/chat/completions", headers=headers, json=payload, timeout=60)
            if r.status_code == 200:
                res = r.json()
                return res["choices"][0]["message"]["content"]
            else:
                print(f"API Error (HTTP {r.status_code}): {r.text}. Retrying...")
                time.sleep(2)
        except Exception as e:
            print(f"Network error: {e}. Retrying...")
            time.sleep(2)
    raise RuntimeError("Failed to call OpenAI API after 3 attempts.")

def evaluate_items_batch(batch: list[dict]) -> list[dict]:
    items_input = []
    for item in batch:
        items_input.append({
            "id": item["id"],
            "title": item.get("title", ""),
            "content": item.get("content", "")[:1200]
        })
    
    prompt = f"""Evaluate the following news items for category, Korea/US relevance, and sentiment.
Return a JSON object with a single key "evaluations" which is a list of objects, one for each item.
Each evaluation object MUST have the following keys:
- "id": string (the exact id of the item)
- "category": string (MUST be one of: "IT", "Energy", "Finance", "Healthcare", "Commodities", "Defense", "Chemicals", "Shipbuilding", "Unclear")
- "korea_us_relevance": boolean (true if the item is relevant to South Korea or United States financial, economic, political, or geopolitical markets; false otherwise)
- "sentiment": string (MUST be one of: "Positive", "Neutral", "Warning", "Panic")
- "rationale": string (brief explanation of the classification)

Items to evaluate:
{json.dumps(items_input, ensure_ascii=False, indent=2)}
"""
    
    system_prompt = "You are an expert financial analyst. Return a JSON object with the requested evaluations. Return strict JSON only, with no markdown formatting."
    
    res_text = call_llm(prompt, system_prompt)
    try:
        data = json.loads(res_text)
        return data["evaluations"]
    except Exception as e:
        print(f"Failed to parse LLM response: {res_text}. Error: {e}")
        raise e

def evaluate_duplicate_pairs_batch(batch: list[dict], items_dict: dict) -> list[dict]:
    pairs_input = []
    for pair in batch:
        id_a = pair["id_a"]
        id_b = pair["id_b"]
        pairs_input.append({
            "pair_index": pair["index"],
            "item_a": {
                "title": items_dict[id_a].get("title", ""),
                "content": items_dict[id_a].get("content", "")[:600]
            },
            "item_b": {
                "title": items_dict[id_b].get("title", ""),
                "content": items_dict[id_b].get("content", "")[:600]
            }
        })
        
    prompt = f"""Compare each pair of articles and determine if they are duplicates or near-duplicates (i.e. describing the exact same event/story with substantially the same content, or one is a republished/translated version of the other) or if they are distinct (different events, different news, even if on a similar topic).
Return a JSON object with a single key "duplicate_evaluations" which is a list of objects, one for each pair.
Each object MUST have the following keys:
- "pair_index": integer (the exact pair_index provided)
- "is_duplicate": boolean (true if the articles are duplicates or near-duplicates; false if they are distinct)
- "rationale": string (brief explanation)

Pairs to compare:
{json.dumps(pairs_input, ensure_ascii=False, indent=2)}
"""
    
    system_prompt = "You are a professional news editor. Determine if the pairs of articles are duplicates or near-duplicates. Return strict JSON only."
    
    res_text = call_llm(prompt, system_prompt)
    try:
        data = json.loads(res_text)
        return data["duplicate_evaluations"]
    except Exception as e:
        print(f"Failed to parse duplicate LLM response: {res_text}. Error: {e}")
        raise e

def parse_time(val):
    if not val:
        return 0.0
    try:
        return datetime.datetime.fromisoformat(val.replace("Z", "+00:00")).timestamp()
    except Exception:
        try:
            return datetime.datetime.strptime(val[:19], "%Y-%m-%d %H:%M:%S").timestamp()
        except Exception:
            return 0.0

def main():
    print("Connecting to database...")
    conn = sqlite3.connect(DB_PATH)
    conn.row_factory = sqlite3.Row
    
    # 1. Fetch crawled items and production model results
    query = """
    SELECT c.id, c.source, c.country, c.language, c.title, c.content, c.published_at, c.created_at, c.source_group,
           r.primary_tag as prod_tag, r.sentiment_label as prod_sentiment, r.excluded as prod_excluded, r.exclude_reason as prod_exclude_reason
    FROM crawled_items c
    LEFT JOIN tag_results r ON r.item_id = c.id AND r.model_version = 'bge-finbert-ner-keyword-v1'
    """
    rows = conn.execute(query).fetchall()
    items = [dict(row) for row in rows]
    print(f"Loaded {len(items)} items from database.")
    
    # Create items lookup dictionary
    items_dict = {item["id"]: item for item in items}
    
    # 2. Fetch production deduplication groups
    dup_members_rows = conn.execute("SELECT item_id, group_id, representative_item_id, similarity FROM dedup_group_members").fetchall()
    dup_members_dict = {row["item_id"]: dict(row) for row in dup_members_rows}
    print(f"Loaded {len(dup_members_dict)} items belonging to duplicate groups.")
    
    # 3. Compute baseline predictions for each item (keyword-only)
    print("Running baseline keyword-only model on items...")
    for item in items:
        # We need to replicate the structure expected by infer_keyword
        baseline_res = realtime_tagger.infer_keyword(item)
        item["baseline_tag"] = baseline_res["primary_tag"]
        item["baseline_sentiment"] = baseline_res["sentiment_label"]
        item["baseline_excluded"] = 1 if baseline_res["is_excluded"] else 0
    
    # 4. Run Gemini evaluations (in batches)
    print("Evaluating items with Gemini (gpt-4o-mini)...")
    gemini_evals = {}
    batch_size = 15
    for start in range(0, len(items), batch_size):
        batch = items[start:start+batch_size]
        print(f"Processing batch {start//batch_size + 1} / {(len(items)-1)//batch_size + 1}...")
        try:
            evals = evaluate_items_batch(batch)
            for ev in evals:
                gemini_evals[ev["id"]] = ev
        except Exception as exc:
            print(f"Error in batch {start}: {exc}")
            # Wait and retry once
            time.sleep(5)
            evals = evaluate_items_batch(batch)
            for ev in evals:
                gemini_evals[ev["id"]] = ev
                
    # Align Gemini results to items
    for item in items:
        ev = gemini_evals.get(item["id"], {})
        item["gemini_tag"] = ev.get("category", "Unclear")
        item["gemini_korea_us_relevance"] = ev.get("korea_us_relevance", False)
        item["gemini_sentiment"] = ev.get("sentiment", "Neutral")
        item["gemini_rationale"] = ev.get("rationale", "")
        
    # 5. Determine candidate duplicate pairs (sim >= 0.70 BGE-M3, time <= 12 hours)
    print("Calculating candidate duplicate pairs using BGE-M3 embeddings...")
    model = SentenceTransformer('BAAI/bge-m3', local_files_only=True)
    texts = []
    for item in items:
        t = item.get("title") or ""
        c = item.get("content") or ""
        if t and t not in c[:150]:
            texts.append(f"{t}. {c}"[:1000])
        else:
            texts.append(c[:1000])
            
    embs = model.encode(texts, normalize_embeddings=True, show_progress_bar=False)
    bge_sim = embs @ embs.T
    
    # Also calculate TF-IDF for baseline similarity
    vectorizer = TfidfVectorizer(analyzer="word", ngram_range=(1, 2), min_df=1, lowercase=True)
    tfidf_sparse = vectorizer.fit_transform(texts)
    tfidf_sim = cosine_similarity(tfidf_sparse)
    
    candidate_pairs = []
    pair_index = 0
    for i in range(len(items)):
        t_i = parse_time(items[i].get("published_at") or items[i].get("created_at"))
        for j in range(i + 1, len(items)):
            t_j = parse_time(items[j].get("published_at") or items[j].get("created_at"))
            time_diff = abs(t_i - t_j)
            s_bge = float(bge_sim[i, j])
            s_tfidf = float(tfidf_sim[i, j])
            
            # If BGE similarity is >= 0.70 and within 12 hours, treat as candidate pair
            if s_bge >= 0.70 and time_diff <= 12 * 3600:
                candidate_pairs.append({
                    "index": pair_index,
                    "id_a": items[i]["id"],
                    "id_b": items[j]["id"],
                    "bge_sim": s_bge,
                    "tfidf_sim": s_tfidf,
                    "time_diff_hours": round(time_diff / 3600.0, 2),
                    # Production prediction: True if both belong to same group
                    "prod_is_duplicate": (
                        items[i]["id"] in dup_members_dict and 
                        items[j]["id"] in dup_members_dict and 
                        dup_members_dict[items[i]["id"]]["group_id"] == dup_members_dict[items[j]["id"]]["group_id"]
                    ),
                    # Baseline prediction: True if TF-IDF similarity >= 0.70
                    "baseline_is_duplicate": s_tfidf >= 0.70
                })
                pair_index += 1
                
    print(f"Found {len(candidate_pairs)} candidate pairs. Evaluating duplicates with Gemini...")
    
    # Evaluate candidate duplicate pairs in batches
    gemini_dup_evals = {}
    batch_size_dup = 15
    for start in range(0, len(candidate_pairs), batch_size_dup):
        batch = candidate_pairs[start:start+batch_size_dup]
        print(f"Processing duplicate pairs batch {start//batch_size_dup + 1} / {(len(candidate_pairs)-1)//batch_size_dup + 1}...")
        try:
            evals = evaluate_duplicate_pairs_batch(batch, items_dict)
            for ev in evals:
                gemini_dup_evals[int(ev["pair_index"])] = ev
        except Exception as exc:
            print(f"Error in duplicate batch {start}: {exc}")
            time.sleep(5)
            evals = evaluate_duplicate_pairs_batch(batch, items_dict)
            for ev in evals:
                gemini_dup_evals[int(ev["pair_index"])] = ev
                
    # Align Gemini results to candidate pairs
    for pair in candidate_pairs:
        ev = gemini_dup_evals.get(pair["index"], {})
        pair["gemini_is_duplicate"] = ev.get("is_duplicate", False)
        pair["gemini_rationale"] = ev.get("rationale", "")
        
    # Save the raw results to file
    out_payload = {
        "items": items,
        "candidate_pairs": candidate_pairs
    }
    eval_results_path = OUTPUT_DIR / "eval_results.json"
    with open(eval_results_path, "w", encoding="utf-8") as f:
        json.dump(out_payload, f, ensure_ascii=False, indent=2)
    print(f"Saved evaluation results to {eval_results_path}")
    
    # 6. Analyze and print performance metrics
    print("\n" + "="*50)
    print("EVALUATION METRICS SUMMARY")
    print("="*50)
    
    # A. Category Tagging (Classification)
    # Filter items where Gemini did not predict "Unclear" as the ground truth
    cat_items = [item for item in items if item["gemini_tag"] != "Unclear"]
    total_cat = len(cat_items)
    if total_cat > 0:
        baseline_cat_correct = sum(1 for item in cat_items if item["baseline_tag"] == item["gemini_tag"])
        prod_cat_correct = sum(1 for item in cat_items if item["prod_tag"] == item["gemini_tag"])
        print(f"Category Tagging (N={total_cat}):")
        print(f"  Keyword Baseline Accuracy: {baseline_cat_correct / total_cat:.4f} ({baseline_cat_correct}/{total_cat})")
        print(f"  Production BGE-M3 Accuracy: {prod_cat_correct / total_cat:.4f} ({prod_cat_correct}/{total_cat})")
    
    # B. Korea/US Relevance (Inclusion Filter)
    # Gemini ground truth: korea_us_relevance (True/False)
    # Baseline prediction: not baseline_excluded (True = relevant, False = excluded)
    # Production prediction: not prod_excluded (True = relevant, False = excluded)
    # Metrics: Precision, Recall, F1 for the "Relevant" class
    y_true_rel = [item["gemini_korea_us_relevance"] for item in items]
    y_pred_base_rel = [not item["baseline_excluded"] for item in items]
    y_pred_prod_rel = [not item["prod_excluded"] for item in items]
    
    def calc_clf_metrics(y_true, y_pred):
        tp = sum(1 for t, p in zip(y_true, y_pred) if t and p)
        fp = sum(1 for t, p in zip(y_true, y_pred) if not t and p)
        fn = sum(1 for t, p in zip(y_true, y_pred) if t and not p)
        tn = sum(1 for t, p in zip(y_true, y_pred) if not t and not p)
        precision = tp / (tp + fp) if (tp + fp) > 0 else 0
        recall = tp / (tp + fn) if (tp + fn) > 0 else 0
        f1 = 2 * precision * recall / (precision + recall) if (precision + recall) > 0 else 0
        accuracy = (tp + tn) / len(y_true)
        return precision, recall, f1, accuracy, tp, fp, fn, tn

    base_p, base_r, base_f1, base_acc, b_tp, b_fp, b_fn, b_tn = calc_clf_metrics(y_true_rel, y_pred_base_rel)
    prod_p, prod_r, prod_f1, prod_acc, p_tp, p_fp, p_fn, p_tn = calc_clf_metrics(y_true_rel, y_pred_prod_rel)
    
    print("\nKorea/US Relevance Classification (Relevant = Included):")
    print(f"  Keyword Baseline: Prec={base_p:.4f}, Rec={base_r:.4f}, F1={base_f1:.4f}, Acc={base_acc:.4f} (TP={b_tp}, FP={b_fp}, FN={b_fn}, TN={b_tn})")
    print(f"  Production Model: Prec={prod_p:.4f}, Rec={prod_r:.4f}, F1={prod_f1:.4f}, Acc={prod_acc:.4f} (TP={p_tp}, FP={p_fp}, FN={p_fn}, TN={p_tn})")
    
    # C. Sentiment Analysis
    # Labels: Positive, Neutral, Warning, Panic
    # Map labels to ordinal levels to calculate tolerance accuracy
    sent_levels = {"Panic": 0, "Warning": 1, "Neutral": 2, "Positive": 3}
    
    def calc_sent_metrics(items_list):
        total = len(items_list)
        if total == 0:
            return 0.0, 0.0, 0.0, 0.0
            
        base_exact = 0
        base_tolerant = 0
        prod_exact = 0
        prod_tolerant = 0
        
        for item in items_list:
            g_lbl = item["gemini_sentiment"]
            b_lbl = item["baseline_sentiment"]
            p_lbl = item["prod_sentiment"]
            
            g_lv = sent_levels.get(g_lbl, 2)
            b_lv = sent_levels.get(b_lbl, 2)
            p_lv = sent_levels.get(p_lbl, 2)
            
            # Exact Match
            if b_lbl == g_lbl:
                base_exact += 1
            if p_lbl == g_lbl:
                prod_exact += 1
                
            # Tolerant Match (diff <= 1 level)
            if abs(b_lv - g_lv) <= 1:
                base_tolerant += 1
            if abs(p_lv - g_lv) <= 1:
                prod_tolerant += 1
                
        return base_exact / total, base_tolerant / total, prod_exact / total, prod_tolerant / total

    # Compute sentiment overall and also on non-excluded (active) items only
    base_ex_all, base_tol_all, prod_ex_all, prod_tol_all = calc_sent_metrics(items)
    active_items = [item for item in items if not item["prod_excluded"]]
    base_ex_act, base_tol_act, prod_ex_act, prod_tol_act = calc_sent_metrics(active_items)
    
    print("\nSentiment Analysis:")
    print(f"  Overall Dataset (N={len(items)}):")
    print(f"    Keyword Baseline: Exact={base_ex_all:.4f}, Tolerant={base_tol_all:.4f}")
    print(f"    Production Model: Exact={prod_ex_all:.4f}, Tolerant={prod_tol_all:.4f}")
    print(f"  Active Items (N={len(active_items)}):")
    print(f"    Keyword Baseline: Exact={base_ex_act:.4f}, Tolerant={base_tol_act:.4f}")
    print(f"    Production Model: Exact={prod_ex_act:.4f}, Tolerant={prod_tol_act:.4f}")
    
    # D. Duplicate Detection
    # Evaluated on the candidate pairs (N = 111)
    y_true_dup = [p["gemini_is_duplicate"] for p in candidate_pairs]
    y_pred_base_dup = [p["baseline_is_duplicate"] for p in candidate_pairs]
    y_pred_prod_dup = [p["prod_is_duplicate"] for p in candidate_pairs]
    
    d_base_p, d_base_r, d_base_f1, d_base_acc, db_tp, db_fp, db_fn, db_tn = calc_clf_metrics(y_true_dup, y_pred_base_dup)
    d_prod_p, d_prod_r, d_prod_f1, d_prod_acc, dp_tp, dp_fp, dp_fn, dp_tn = calc_clf_metrics(y_true_dup, y_pred_prod_dup)
    
    print(f"\nDuplicate Detection on Candidate Pairs (N={len(candidate_pairs)}):")
    print(f"  Keyword TF-IDF Baseline: Prec={d_base_p:.4f}, Rec={d_base_r:.4f}, F1={d_base_f1:.4f}, Acc={d_base_acc:.4f} (TP={db_tp}, FP={db_fp}, FN={db_fn}, TN={db_tn})")
    print(f"  Production BGE-M3 Model: Prec={d_prod_p:.4f}, Rec={d_prod_r:.4f}, F1={d_prod_f1:.4f}, Acc={d_prod_acc:.4f} (TP={dp_tp}, FP={dp_fp}, FN={dp_fn}, TN={dp_tn})")
    
    conn.close()

if __name__ == "__main__":
    main()
