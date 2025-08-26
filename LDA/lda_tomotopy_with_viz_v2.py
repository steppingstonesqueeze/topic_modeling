#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
lda_tomotopy_with_viz.py '97 Short-text LDA with auto-K (log-likelihood sweep) + visualizations

Inputs:
  - TSV / JSON / JSONL
  - For TSV: choose id/text columns via CLI or prompt
  - For JSON/JSONL: choose id/text fields via CLI or prompt

Behavior:
  - Tokenize, drop docs with < 3 tokens
  - Auto-K sweep (default 2..30 step 5) selects K by ll_per_word
  - Retrain at best K and emit final outputs

Outputs:
  - doc_assignments.tsv   # id  topic  doc  weight  (sorted by topic)
  - cluster_labels.tsv    # topic  label  keywords_json
  - topics_full.tsv       # topic  rank  word  score
  - topics.json           # {"0":[{"word":...,"score":...},...], ...}
  - metrics.json          # preprocessing + model selection (K vs ll_per_word, chosen_k)
  - clusters_2d.tsv       # PCA coords of doc-topic distributions
  - clusters_pca.png      # scatter of topic-distribution PCA colored by topic
  - clusters_2d_raw.tsv   # PCA coords of raw TF-IDF document vectors
  - clusters_pca_raw.png  # scatter of raw TF-IDF PCA (grey) + topic PCA overlay (colored)
  - clusters_pca_overlay.png # same as above (explicit overlay image)

Requires: tomotopy, pandas, numpy
Optional (for PNG plots): matplotlib
"""

import argparse
import csv
import json
import re
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

import numpy as np
import pandas as pd
import tomotopy as tp

# Optional plotting (safe fallback)
try:
    import matplotlib.pyplot as plt  # type: ignore
    _HAS_MPL = True
except Exception:
    _HAS_MPL = False

# --------------------------- Tokenization ---------------------------

_WORD_RE = re.compile(r"[A-Za-z0-9']+")
_STOP = {
    "the","a","an","and","or","but","if","then","else","when","while","of","to","for","in","on","at","by","from",
    "with","as","is","are","was","were","be","been","being","it","its","this","that","these","those","we","you",
    "they","he","she","him","her","them","our","your","their","i","me","my","mine","ours","yours","theirs",
    "so","not","no","very","can","could","should","would","will","just","than","too","also","there","here",
    "about","into","over","under","up","down","out"
}

def tokenize(text: str) -> List[str]:
    toks = [t.lower() for t in _WORD_RE.findall(str(text))]
    return [t for t in toks if t not in _STOP and len(t) > 1]

def clean_doc_text(s: str) -> str:
    return " ".join(str(s).replace("t"," ").replace("r"," ").replace("n"," ").split())

# --------------------------- IO helpers ---------------------------

def infer_format(path: Path) -> str:
    ext = path.suffix.lower()
    if ext in (".tsv", ".tab"):
        return "tsv"
    if ext in (".jsonl", ".ndjson"):
        return "jsonl"
    if ext == ".json":
        return "json"
    return ""

def prompt_choice(name: str, options: List[str]) -> str:
    print(f"nSelect the {name} from available options:")
    for i, opt in enumerate(options, 1):
        print(f"  {i}. {opt}")
    while True:
        raw = input(f"Enter number (1-{len(options)}): ").strip()
        if raw.isdigit():
            idx = int(raw)
            if 1 <= idx <= len(options):
                return options[idx - 1]
        print("Invalid choice. Try again.")

def load_tsv(path: Path, id_col: Optional[str], text_col: Optional[str]) -> Tuple[List[str], List[str]]:
    df = pd.read_csv(
        path, sep="t", dtype=str, keep_default_na=False,
        engine="python", quoting=csv.QUOTE_MINIMAL, escapechar="", on_bad_lines="warn"
    )
    cols = list(df.columns)
    if text_col is None or text_col not in cols:
        text_col = prompt_choice("TSV text column", cols)
    if id_col is None or id_col not in cols:
        print("No valid TSV id column provided; choose one or auto-generate.")
        chosen = prompt_choice("TSV id column", cols + ["<auto-generate>"])
        if chosen == "<auto-generate>":
            ids = [str(i) for i in range(len(df))]
        else:
            id_col = chosen
            ids = df[id_col].astype(str).tolist()
    else:
        ids = df[id_col].astype(str).tolist()
    texts = df[text_col].astype(str).tolist()
    return ids, texts

def load_jsonl(path: Path, json_text_field: Optional[str], json_id_field: Optional[str]) -> Tuple[List[str], List[str]]:
    ids, texts = [], []
    keys: List[str] = []
    with path.open("r", encoding="utf-8") as f:
        first = f.readline().strip()
        if first:
            try:
                probe = json.loads(first)
                if isinstance(probe, dict):
                    keys = list(probe.keys())
            except Exception:
                pass
    if json_text_field is None or (keys and json_text_field not in keys):
        json_text_field = prompt_choice("JSONL text field", keys) if keys else input("Enter JSONL text field: ").strip()
    if json_id_field is None or (keys and json_id_field not in keys):
        opts = (keys + ["<auto-generate>"]) if keys else ["<auto-generate>"]
        chosen = prompt_choice("JSONL id field", opts) if opts else "<auto-generate>"
        json_id_field = None if chosen == "<auto-generate>" else chosen

    with path.open("r", encoding="utf-8") as f:
        for i, line in enumerate(f):
            line = line.strip()
            if not line:
                continue
            try:
                rec = json.loads(line)
                txt = str(rec.get(json_text_field, "")).strip()
                rid = str(rec.get(json_id_field, i)) if json_id_field else str(i)
                ids.append(rid)
                texts.append(txt)
            except Exception:
                continue
    return ids, texts

def load_json(path: Path, json_text_field: Optional[str], json_id_field: Optional[str]) -> Tuple[List[str], List[str]]:
    data = json.loads(path.read_text(encoding="utf-8"))
    if isinstance(data, dict) and "data" in data and isinstance(data["data"], list):
        data = data["data"]
    if not isinstance(data, list):
        data = [data]
    keys = list(data[0].keys()) if (data and isinstance(data[0], dict)) else []
    if json_text_field is None or (keys and json_text_field not in keys):
        json_text_field = prompt_choice("JSON text field", keys) if keys else input("Enter JSON text field: ").strip()
    if json_id_field is None or (keys and json_id_field not in keys):
        opts = (keys + ["<auto-generate>"]) if keys else ["<auto-generate>"]
        chosen = prompt_choice("JSON id field", opts) if opts else "<auto-generate>"
        json_id_field = None if chosen == "<auto-generate>" else chosen

    ids, texts = [], []
    for i, rec in enumerate(data):
        if not isinstance(rec, dict):
            continue
        txt = str(rec.get(json_text_field, "")).strip()
        rid = str(rec.get(json_id_field, i)) if json_id_field else str(i)
        ids.append(rid)
        texts.append(txt)
    return ids, texts

# --------------------------- Corpus ---------------------------

def build_corpus(ids: List[str], raw_texts: List[str]) -> Tuple[List[str], List[str], List[List[str]], Dict[str, Any]]:
    tokenized = [tokenize(t) for t in raw_texts]
    before = len(tokenized)
    keep_mask = [len(toks) >= 3 for toks in tokenized]  # keep docs with >=3 tokens
    ids_kept = [i for i, k in zip(ids, keep_mask) if k]
    texts_kept = [clean_doc_text(t) for t, k in zip(raw_texts, keep_mask) if k]
    toks_kept = [t for t, k in zip(tokenized, keep_mask) if k]
    stats = {
        "docs_before": before,
        "docs_after": len(ids_kept),
        "docs_dropped_lt_3_tokens": before - len(ids_kept),
    }
    return ids_kept, texts_kept, toks_kept, stats

# --------------------------- Training ---------------------------

def train_lda_k(toks: List[List[str]], k: int, iters: int, seed: int) -> tp.LDAModel:
    mdl = tp.LDAModel(k=k, tw=tp.TermWeight.IDF, alpha=50.0 / k, eta=0.01, seed=seed)
    for words in toks:
        mdl.add_doc(words)
    mdl.burn_in = min(100, iters // 5)
    mdl.train(0)
    for _ in range(iters):
        mdl.train(1)
    return mdl

def auto_k_sweep(
    toks: List[List[str]],
    iters: int,
    seed: int,
    k_min: int,
    k_max: int,
    k_step: int,
    grid: Optional[List[int]] = None
) -> Tuple[tp.LDAModel, Dict[str, Any]]:
    scores: Dict[int, float] = {}
    best_mdl = None
    best_k = None
    best_ll = -1e18
    ks = grid if grid else list(range(k_min, k_max + 1, k_step))
    for k in ks:
        print(f"[auto-K] Training LDA with K={k} ...")
        mdl = train_lda_k(toks, k, iters, seed)
        ll = float(mdl.ll_per_word)
        scores[k] = ll
        print(f"[auto-K] K={k}, ll_per_word={ll:.6f}")
        if ll > best_ll:
            best_ll = ll
            best_k = k
            best_mdl = mdl
    meta = {
        "method": "sweep_ll",
        "k_candidates": ks,
        "chosen_k": int(best_k),
        "ll_per_word_scores": {str(kk): vv for kk, vv in scores.items()},
        "best_ll_per_word": float(best_ll)
    }
    return best_mdl, meta

# --------------------------- Topics & Outputs ---------------------------

def topic_terms(mdl, topk: int) -> Dict[int, List[Tuple[str, float]]]:
    res: Dict[int, List[Tuple[str, float]]] = {}
    for t in range(int(mdl.k)):
        words = mdl.get_topic_words(t, top_n=topk)
        res[t] = [(w, float(p)) for (w, p) in words]
    return res

def save_outputs(
    out_dir: Path,
    ids: List[str],
    docs: List[str],
    mdl,
    tterms: Dict[int, List[Tuple[str, float]]],
    stats: Dict[str, Any],
    meta: Dict[str, Any],
) -> None:
    out_dir.mkdir(parents=True, exist_ok=True)

    # doc assignments
    rows = []
    n_docs = min(len(ids), len(docs), len(getattr(mdl, "docs", [])))
    for i in range(n_docs):
        rid = ids[i]
        doc_text = docs[i]
        try:
            dist = mdl.docs[i].get_topic_dist()
        except Exception:
            dist = []
        if len(dist) > 0:
            top_tid = int(np.argmax(dist))
            top_p = float(dist[top_tid])
        else:
            top_tid, top_p = -1, 0.0
        rows.append((rid, top_tid, doc_text, top_p))

    df = pd.DataFrame(rows, columns=["id", "topic", "doc", "weight"]).sort_values(by=["topic"], kind="stable")
    df.to_csv(out_dir / "doc_assignments.tsv", sep="t", index=False)

    # labels + topics tables
    label_rows = []
    topics_json: Dict[str, List[Dict[str, float]]] = {}
    topics_rows = []
    for tid, pairs in tterms.items():
        words = [w for (w, _s) in pairs]
        label = ", ".join(words[: min(5, len(words))])
        kw_json = [{"word": w, "score": float(s)} for (w, s) in pairs]
        label_rows.append((tid, label, json.dumps(kw_json, ensure_ascii=False)))
        topics_json[str(tid)] = kw_json
        for r, (w, s) in enumerate(pairs, start=1):
            topics_rows.append((tid, r, w, float(s)))

    pd.DataFrame(label_rows, columns=["topic", "label", "keywords_json"]).to_csv(
        out_dir / "cluster_labels.tsv", sep="t", index=False
    )
    pd.DataFrame(topics_rows, columns=["topic", "rank", "word", "score"]).to_csv(
        out_dir / "topics_full.tsv", sep="t", index=False
    )
    with (out_dir / "topics.json").open("w", encoding="utf-8") as f:
        json.dump(topics_json, f, ensure_ascii=False, indent=2)

    metrics = {"preprocess_stats": stats, "model_selection": meta}
    with (out_dir / "metrics.json").open("w", encoding="utf-8") as f:
        json.dump(metrics, f, ensure_ascii=False, indent=2)

# --------------------------- Visualization ---------------------------

def _compute_2d_embedding(matrix: np.ndarray) -> np.ndarray:
    # PCA via SVD (no sklearn dependency)
    X = matrix - matrix.mean(axis=0, keepdims=True)
    U, S, Vt = np.linalg.svd(X, full_matrices=False)
    return U[:, :2] * S[:2]

def _build_tfidf_pca_coords(toks: List[List[str]], max_vocab: int = 5000) -> Tuple[np.ndarray, Dict[str,int]]:
    # Build simple TF-IDF (no sklearn), cap vocab
    from collections import Counter
    freq = Counter()
    for doc in toks:
        freq.update(doc)
    vocab = [w for w, _ in freq.most_common(max_vocab)]
    idx = {w: i for i, w in enumerate(vocab)}
    V = len(vocab)
    N = len(toks)
    if V == 0 or N == 0:
        return np.zeros((N, 2), dtype=float), idx

    # Term Frequency
    X = np.zeros((N, V), dtype=float)
    df = np.zeros(V, dtype=float)
    for r, doc in enumerate(toks):
        seen = set()
        for w in doc:
            j = idx.get(w)
            if j is None: continue
            X[r, j] += 1.0
            if j not in seen:
                df[j] += 1.0
                seen.add(j)
    # Normalize TF
    lens = X.sum(axis=1, keepdims=True); lens[lens == 0] = 1.0
    X = X / lens
    # IDF (smooth)
    idf = np.log((N + 1.0) / (df + 1.0)) + 1.0
    X = X * idf
    # PCA
    coords = _compute_2d_embedding(X)
    return coords, idx

def save_plots(out_dir: Path, ids: List[str], docs: List[str], toks: List[List[str]], mdl, seed: int) -> None:
    n = min(len(ids), len(getattr(mdl, "docs", [])))
    if n == 0:
        return
    # Doc-topic distributions
    dists = [mdl.docs[i].get_topic_dist() for i in range(n)]
    D = np.array(dists)
    labels = np.argmax(D, axis=1)

    # Topic PCA coords
    topic_coords = _compute_2d_embedding(D)
    pd.DataFrame({"id": ids[:n], "topic": labels, "x": topic_coords[:,0], "y": topic_coords[:,1]}).to_csv(
        out_dir / "clusters_2d.tsv", sep="t", index=False
    )
    if _HAS_MPL:
        fig = plt.figure(figsize=(8,6)); ax = fig.add_subplot(111)
        sc = ax.scatter(topic_coords[:,0], topic_coords[:,1], c=labels, s=30, alpha=0.85)
        ax.set_title("PCA of Topic Distributions")
        cb = plt.colorbar(sc, ax=ax); cb.set_label("Topic")
        fig.tight_layout(); fig.savefig(out_dir / "clusters_pca.png", dpi=150); plt.close(fig)

    # Raw TF-IDF PCA
    raw_coords, _ = _build_tfidf_pca_coords(toks[:n])
    pd.DataFrame({"id": ids[:n], "topic": labels, "x": raw_coords[:,0], "y": raw_coords[:,1]}).to_csv(
        out_dir / "clusters_2d_raw.tsv", sep="t", index=False
    )
    if _HAS_MPL:
        # raw-only scatter + overlay in one image
        fig = plt.figure(figsize=(8,6)); ax = fig.add_subplot(111)
        ax.scatter(raw_coords[:,0], raw_coords[:,1], c="#bbbbbb", s=20, alpha=0.5, label="Raw TF-IDF PCA")
        sc = ax.scatter(topic_coords[:,0], topic_coords[:,1], c=labels, s=35, alpha=0.9, label="Topic PCA")
        ax.set_title("Raw TF-IDF PCA with Topic PCA Overlay")
        ax.legend()
        cb = plt.colorbar(sc, ax=ax); cb.set_label("Topic")
        fig.tight_layout(); fig.savefig(out_dir / "clusters_pca_raw.png", dpi=150); plt.close(fig)

        # explicit overlay image (same content; separate filename)
        fig = plt.figure(figsize=(8,6)); ax = fig.add_subplot(111)
        ax.scatter(raw_coords[:,0], raw_coords[:,1], c="#bbbbbb", s=18, alpha=0.45, label="Raw TF-IDF PCA")
        sc = ax.scatter(topic_coords[:,0], topic_coords[:,1], c=labels, s=40, alpha=0.95, label="Topic PCA")
        ax.set_title("Overlay: Raw TF-IDF PCA + Topic PCA")
        ax.legend()
        cb = plt.colorbar(sc, ax=ax); cb.set_label("Topic")
        fig.tight_layout(); fig.savefig(out_dir / "clusters_pca_overlay.png", dpi=150); plt.close(fig)

# --------------------------- Main ---------------------------

def parse_args():
    ap = argparse.ArgumentParser(description="Short-text LDA (tomotopy) with auto-K sweep + visualizations")
    ap.add_argument("--input", required=True, help="Path to TSV / JSON / JSONL")
    ap.add_argument("--format", choices=["tsv","json","jsonl"], help="If omitted, inferred from file extension")
    ap.add_argument("--id-col", help="TSV id column (optional)")
    ap.add_argument("--text-col", help="TSV text column (optional)")
    ap.add_argument("--json-id-field", help="JSON/JSONL id field (optional)")
    ap.add_argument("--json-text-field", help="JSON/JSONL text field (optional)")
    ap.add_argument("--iters", type=int, default=500, help="Training iterations per model")
    ap.add_argument("--seed", type=int, default=42)
    ap.add_argument("--topk", type=int, default=10, help="Top keywords per topic")
    ap.add_argument("--k-min", type=int, default=2)
    ap.add_argument("--k-max", type=int, default=30)
    ap.add_argument("--k-step", type=int, default=5)
    ap.add_argument("--k-grid", help="Comma-separated list of K values (overrides min/max/step), e.g. '2,5,10,15,25,30'")
    ap.add_argument("--out-dir", required=True, help="Output directory")
    return ap.parse_args()

def main():
    args = parse_args()
    in_path = Path(args.input)
    fmt = args.format or infer_format(in_path)
    if fmt not in {"tsv","json","jsonl"}:
        raise SystemExit("Could not infer format. Use --format tsv|json|jsonl.")

    # Load
    if fmt == "tsv":
        ids, texts = load_tsv(in_path, args.id_col, args.text_col)
    elif fmt == "jsonl":
        ids, texts = load_jsonl(in_path, args.json_text_field, args.json_id_field)
    else:
        ids, texts = load_json(in_path, args.json_text_field, args.json_id_field)

    if not texts:
        raise SystemExit("No texts loaded.")

    # Build corpus
    ids_kept, docs_kept, toks_kept, stats = build_corpus(ids, texts)
    if len(ids_kept) == 0:
        raise SystemExit("All documents were dropped (<3 tokens).")

    # Auto-K grid
    grid = None
    if args.k_grid:
        grid = [int(x.strip()) for x in args.k_grid.split(",") if x.strip().isdigit()]
        if not grid:
            grid = None

    # Train via sweep
    mdl, meta = auto_k_sweep(
        toks_kept, iters=args.iters, seed=args.seed,
        k_min=args.k_min, k_max=args.k_max, k_step=args.k_step, grid=grid
    )

    # Topics
    tterms = topic_terms(mdl, topk=args.topk)

    # Save outputs
    out_dir = Path(args.out_dir)
    save_outputs(out_dir, ids_kept, docs_kept, mdl, tterms, stats, meta)

    # Viz
    try:
        save_plots(out_dir, ids_kept, docs_kept, toks_kept, mdl, args.seed)
    except Exception as e:
        with (out_dir / "viz_error.txt").open("w", encoding="utf-8") as f:
            f.write(str(e))

    print(f"Done. Outputs written to: {out_dir}")

if __name__ == "__main__":
    main()
