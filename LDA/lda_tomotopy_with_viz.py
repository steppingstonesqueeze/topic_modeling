#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
lda_tomotopy_fixed.py — Short‑text LDA/HDP‑LDA using tomotopy with clean TSV/JSON outputs

What it does
------------
- Reads TSV / JSON / JSONL
- Prompts for text/id fields if missing
- Keeps short texts (drops ONLY docs with < 3 tokens)
- Auto‑K via:
    * --auto-method hdp  (HDP‑LDA, nonparametric; default)
    * --auto-method sweep (choose K by log‑likelihood over a range)
- Outputs:
    out_dir/
      ├─ doc_assignments.tsv   # id  topic  doc  weight  (sorted by topic)
      ├─ cluster_labels.tsv    # topic  label  keywords_json
      ├─ topics_full.tsv       # topic  rank  word  score
      ├─ topics.json           # { "0": [{"word": ..., "score": ...}, ...], ... }
      └─ metrics.json          # preprocessing + model selection stats

Usage examples
--------------
python lda_tomotopy_fixed.py \
  --input data.tsv --format tsv --text-col Summary --out-dir out_lda

python lda_tomotopy_fixed.py \
  --input data.jsonl --format jsonl --json-text-field text --out-dir out_lda

python lda_tomotopy_fixed.py \
  --input data.tsv --format tsv --text-col text --n-topics 20 --out-dir out_lda

Requirements
------------
- tomotopy==0.12.6
- pandas==2.2.2
"""

import argparse
import csv
import json
import re
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

import pandas as pd
import tomotopy as tp

import numpy as np

# Optional plotting; we'll fall back gracefully if matplotlib isn't installed
try:
    import matplotlib.pyplot as plt  # type: ignore
    _HAS_MPL = True
except Exception:
    _HAS_MPL = False


# --------------------------- Utils ---------------------------

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
    print(f"\nSelect the {name} from available options:")
    for i, opt in enumerate(options, 1):
        print(f"  {i}. {opt}")
    while True:
        raw = input(f"Enter number (1-{len(options)}): ").strip()
        if raw.isdigit():
            idx = int(raw)
            if 1 <= idx <= len(options):
                return options[idx - 1]
        print("Invalid choice. Try again.")


_WORD_RE = re.compile(r"[A-Za-z0-9']+")
_STOP = {
    "the","a","an","and","or","but","if","then","else","when","while","of","to","for","in","on",
    "at","by","from","with","as","is","are","was","were","be","been","being","it","its","this",
    "that","these","those","we","you","they","he","she","him","her","them","our","your","their",
    "i","me","my","mine","ours","yours","theirs","so","not","no","very","can","could","should",
    "would","will","just","than","too","also","there","here","about","into","over","under","up",
    "down","out"
}


def tokenize(text: str) -> List[str]:
    toks = [t.lower() for t in _WORD_RE.findall(text)]
    return [t for t in toks if t not in _STOP and len(t) > 1]


def clean_doc_text(s: str) -> str:
    # sanitize for TSV: remove tabs/newlines and collapse spaces
    txt = str(s).replace("\t", " ").replace("\r", " ").replace("\n", " ")
    return " ".join(txt.split())


# --------------------------- Loaders ---------------------------

def load_tsv(path: Path, id_col: Optional[str], text_col: Optional[str]) -> Tuple[List[str], List[str]]:
    df = pd.read_csv(
        path,
        sep="\t",
        dtype=str,
        keep_default_na=False,
        engine="python",
        quoting=csv.QUOTE_MINIMAL,
        escapechar="\\",
        on_bad_lines="warn",
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
    keep_mask = [len(toks) >= 3 for toks in tokenized]
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

def train_hdp(toks: List[List[str]], iters: int, seed: int) -> Tuple[tp.HDPModel, Dict[str, Any]]:
    mdl = tp.HDPModel(tw=tp.TermWeight.IDF, seed=seed)
    for words in toks:
        mdl.add_doc(words)
    mdl.burn_in = min(100, iters // 5)
    mdl.train(0)
    for _ in range(iters):
        mdl.train(1)
    try:
        k_est = mdl.num_topics
    except AttributeError:
        k_est = sum(mdl.get_count_by_topics() > 0)
    meta = {
        "method": "hdp",
        "iters": iters,
        "estimated_topics": int(k_est),
        "ll_per_word": float(mdl.ll_per_word),
    }
    return mdl, meta


def train_lda_k(toks: List[List[str]], k: int, iters: int, seed: int) -> tp.LDAModel:
    mdl = tp.LDAModel(k=k, tw=tp.TermWeight.IDF, alpha=50.0 / k, eta=0.01, seed=seed)
    for words in toks:
        mdl.add_doc(words)
    mdl.burn_in = min(100, iters // 5)
    mdl.train(0)
    for _ in range(iters):
        mdl.train(1)
    return mdl


def auto_k_sweep(toks: List[List[str]], k_min: int, k_max: int, k_step: int, iters: int, seed: int):
    scores: Dict[int, float] = {}
    best_mdl = None
    best_k = None
    best_ll = -1e18
    for k in range(k_min, k_max + 1, k_step):
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
        "k_min": k_min,
        "k_max": k_max,
        "k_step": k_step,
        "chosen_k": int(best_k),
        "ll_per_word_scores": {str(kk): vv for kk, vv in scores.items()},
    }
    return best_mdl, meta


# --------------------------- Topics & Outputs ---------------------------

def topic_terms(mdl, topk: int) -> Dict[int, List[Tuple[str, float]]]:
    res: Dict[int, List[Tuple[str, float]]] = {}
    tcount = getattr(mdl, "k", None) or getattr(mdl, "num_topics", None)
    if tcount is None:
        tcount = 0
        while True:
            try:
                mdl.get_topic_words(tcount, top_n=1)
                tcount += 1
            except Exception:
                break
    for t in range(int(tcount)):
        words = mdl.get_topic_words(t, top_n=topk)
        res[t] = [(w, float(p)) for (w, p) in words]
    return res


def save_outputs(
    out_dir: Path,
    ids: List[str],
    docs: List[str],
    toks: List[List[str]],
    mdl,
    tterms: Dict[int, List[Tuple[str, float]]],
    stats: Dict[str, Any],
    meta: Dict[str, Any],
) -> None:
    out_dir.mkdir(parents=True, exist_ok=True)

    # Doc assignments: use model's own docs (added in order)
    rows = []
    # Ensure we have the same count
    n_docs = min(len(ids), len(docs), len(getattr(mdl, "docs", [])))
    for i in range(n_docs):
        rid = ids[i]
        doc_text = docs[i]
        try:
            dist = mdl.docs[i].get_topic_dist()
        except Exception:
            dist = []
        if len(dist) > 0:
            top_tid = int(max(range(len(dist)), key=lambda t: dist[t]))
            top_p = float(dist[top_tid])
        else:
            top_tid, top_p = -1, 0.0
        rows.append((rid, top_tid, doc_text, top_p))

    df = pd.DataFrame(rows, columns=["id", "topic", "doc", "weight"])
    df = df.sort_values(by=["topic"], kind="stable")
    df.to_csv(out_dir / "doc_assignments.tsv", sep="\t", index=False)

    # Labels + topics files
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
        out_dir / "cluster_labels.tsv", sep="\t", index=False
    )

    pd.DataFrame(topics_rows, columns=["topic", "rank", "word", "score"]).to_csv(
        out_dir / "topics_full.tsv", sep="\t", index=False
    )

    with (out_dir / "topics.json").open("w", encoding="utf-8") as f:
        json.dump(topics_json, f, ensure_ascii=False, indent=2)

    metrics = {"preprocess_stats": stats, "model_selection": meta}
    with (out_dir / "metrics.json").open("w", encoding="utf-8") as f:
        json.dump(metrics, f, ensure_ascii=False, indent=2)


# Plotting utilities #####

# --------------------------- Visualization helpers ---------------------------

def _compute_2d_embedding(dists: np.ndarray, method: str = "pca", random_state: int = 42) -> np.ndarray:
    """Return a 2D embedding for visualization. Default to PCA (no heavy deps).
    If umap-learn or sklearn are available, this function will try them too.
    """
    method = (method or "pca").lower()
    if method == "pca":
        # PCA via SVD (no sklearn dependency)
        X = dists - dists.mean(axis=0, keepdims=True)
        U, S, Vt = np.linalg.svd(X, full_matrices=False)
        X2d = U[:, :2] * S[:2]
        return X2d

    # Try UMAP if installed
    try:
        import umap  # type: ignore
        reducer = umap.UMAP(n_components=2, random_state=random_state)
        return reducer.fit_transform(dists)
    except Exception:
        pass

    # Try t-SNE if sklearn is installed
    try:
        from sklearn.manifold import TSNE  # type: ignore
        return TSNE(n_components=2, random_state=random_state, init="pca").fit_transform(dists)
    except Exception:
        pass

    # Fallback to PCA
    X = dists - dists.mean(axis=0, keepdims=True)
    U, S, Vt = np.linalg.svd(X, full_matrices=False)
    X2d = U[:, :2] * S[:2]
    return X2d


def _save_cluster_plot(out_dir: Path, coords: np.ndarray, labels: np.ndarray, method: str = "pca") -> None:
    if not _HAS_MPL:
        return  # silently skip if matplotlib missing
    fig = plt.figure(figsize=(8, 6))
    ax = fig.add_subplot(111)
    sc = ax.scatter(coords[:, 0], coords[:, 1], c=labels, s=28, alpha=0.85)
    ax.set_title(f"Document Clusters ({method.upper()})")
    cb = plt.colorbar(sc, ax=ax)
    cb.set_label("Topic")
    fig.tight_layout()
    fig.savefig(out_dir / f"clusters_{method}.png", dpi=150)
    plt.close(fig)


# --------------------------- Main ---------------------------

def main():
    ap = argparse.ArgumentParser(description="Short-text LDA/HDP-LDA with tomotopy (auto-K, keyword labels).")
    ap.add_argument("--input", required=True, help="Path to TSV / JSON / JSONL")
    ap.add_argument("--format", choices=["tsv", "json", "jsonl"], help="If omitted, inferred from extension")
    ap.add_argument("--id-col", help="TSV id column (optional)")
    ap.add_argument("--text-col", help="TSV text column")
    ap.add_argument("--json-id-field", help="JSON/JSONL id field (optional)")
    ap.add_argument("--json-text-field", help="JSON/JSONL text field")
    ap.add_argument("--n-topics", default="auto", help="'auto' or integer K")
    ap.add_argument("--auto-method", choices=["hdp", "sweep"], default="hdp",
                    help="Auto-K method: 'hdp' (nonparametric) or 'sweep' (choose K by log-likelihood)")
    ap.add_argument("--k-min", type=int, default=5)
    ap.add_argument("--k-max", type=int, default=30)
    ap.add_argument("--k-step", type=int, default=5)
    ap.add_argument("--iters", type=int, default=500, help="Training iterations per model")
    ap.add_argument("--seed", type=int, default=42)
    ap.add_argument("--topk", type=int, default=10, help="Top keywords per topic")
    ap.add_argument("--out-dir", required=True, help="Output directory")
    args = ap.parse_args()

    in_path = Path(args.input)
    fmt = args.format or infer_format(in_path)
    if fmt == "":
        raise SystemExit("Could not infer format from extension. Use --format tsv|json|jsonl.")

    # Load
    if fmt == "tsv":
        ids, texts = load_tsv(in_path, args.id_col, args.text_col)
    elif fmt == "jsonl":
        ids, texts = load_jsonl(in_path, args.json_text_field, args.json_id_field)
    else:
        ids, texts = load_json(in_path, args.json_text_field, args.json_id_field)

    if not texts:
        raise SystemExit("No texts loaded.")

    # Build corpus (tokenize, drop <3 tokens)
    ids_kept, docs_kept, toks_kept, stats = build_corpus(ids, texts)
    if len(ids_kept) == 0:
        raise SystemExit("All documents were dropped (<3 tokens).")

    # Train
    n_topics = args.n_topics
    meta: Dict[str, Any] = {}
    if isinstance(n_topics, str) and n_topics.lower() == "auto":
        if args.auto_method == "hdp":
            mdl, meta = train_hdp(toks_kept, iters=args.iters, seed=args.seed)
        else:
            mdl, meta = auto_k_sweep(toks_kept, args.k_min, args.k_max, args.k_step, args.iters, args.seed)
    else:
        try:
            k = int(n_topics)
        except Exception:
            raise SystemExit("--n-topics must be 'auto' or an integer.")
        mdl = train_lda_k(toks_kept, k, args.iters, args.seed)
        meta = {"method": "fixed", "chosen_k": k, "ll_per_word": float(mdl.ll_per_word)}

    # Topics
    tterms = topic_terms(mdl, topk=args.topk)

    # Save
    out_dir = Path(args.out_dir)
    save_outputs(out_dir, ids_kept, docs_kept, toks_kept, mdl, tterms, stats, meta)


# ----- Visualization (2D embedding from doc-topic distributions) -----
    try:
        n = min(len(ids_kept), len(getattr(mdl, "docs", [])))
        dists = [mdl.docs[i].get_topic_dist() for i in range(n)]
        D = np.array(dists)
        labels = np.argmax(D, axis=1)
        coords = _compute_2d_embedding(D, method="pca", random_state=args.seed)
    # Always save coordinates TSV
        pd.DataFrame({
            "id": ids_kept[:n],
            "topic": labels,
            "x": coords[:, 0],
            "y": coords[:, 1],
        }).to_csv(out_dir / "clusters_2d.tsv", sep="\t", index=False)
    # Save PNG if matplotlib present
        _save_cluster_plot(out_dir, coords, labels, method="pca")
    except Exception as _viz_err:
        with (out_dir / "viz_error.txt").open("w", encoding="utf-8") as _f:
            _f.write(str(_viz_err))

    print(f"Done. Outputs written to: {out_dir}")


if __name__ == "__main__":
    main()
