"""Chunk judgments + bare acts, embed with a local sentence encoder, build BM25 indexes.

Document-aware chunking: Supreme Court judgments put facts first and the ratio /
operative reasoning last, so for each case we keep the HEAD (first 2 windows) and the
TAIL (last 6 windows) and tag every chunk with its role, case id, year and outcome.
Statutes are chunked one section = one chunk (never split across sections).
"""
import json, sys, time
from pathlib import Path
import numpy as np, pandas as pd

ROOT = Path(__file__).resolve().parents[1]
P, IDX = ROOT / "data/processed", ROOT / "data/index"
IDX.mkdir(parents=True, exist_ok=True)
W, HEAD, TAIL = 300, 2, 6
MODEL = "BAAI/bge-small-en-v1.5"

def chunk_case(text):
    w = text.split()
    n = max(1, -(-len(w) // W))
    # windows aligned to the END of the document so the last window is always full
    starts = sorted({max(0, len(w) - (i + 1) * W) for i in range(n)})
    wins = [(s, " ".join(w[s:s + W])) for s in starts]
    keep = list(range(len(wins))) if len(wins) <= HEAD + TAIL else list(range(HEAD)) + list(range(len(wins) - TAIL, len(wins)))
    out = []
    for k in keep:
        from_end = len(wins) - 1 - k
        out.append((from_end, "tail" if from_end < TAIL else "head", wins[k][1]))
    return out

def build_chunks():
    rows = []
    for split in ("train", "test"):
        d = pd.read_parquet(P / f"{split}.parquet")
        for r in d.itertuples():
            for from_end, role, txt in chunk_case(r.text):
                rows.append((split, r.id, r.year, r.label, from_end, role, txt))
    c = pd.DataFrame(rows, columns=["split", "case_id", "year", "label", "from_end", "role", "text"])
    c.to_parquet(IDX / "case_chunks.parquet")
    return c

def embed(texts, name):
    import torch
    from sentence_transformers import SentenceTransformer
    m = SentenceTransformer(MODEL, device="mps" if torch.backends.mps.is_available() else "cpu")
    m.max_seq_length = 512
    m.half()
    out = np.lib.format.open_memmap(IDX / f"{name}.npy", mode="w+", dtype=np.float16, shape=(len(texts), 384))
    order = np.argsort([len(t) for t in texts])[::-1]
    B, t0 = 64, time.time()
    for i in range(0, len(order), B):
        ix = order[i:i + B]
        e = m.encode([texts[j] for j in ix], batch_size=B, normalize_embeddings=True, show_progress_bar=False)
        out[ix] = e.astype(np.float16)
        if (i // B) % 100 == 0:
            print(f"{name} {i}/{len(order)} {time.time()-t0:.0f}s", flush=True)
    out.flush()

def bm25(texts, name):
    import bm25s, Stemmer
    tok = bm25s.tokenize(texts, stopwords="en", stemmer=Stemmer.Stemmer("english"), show_progress=False)
    r = bm25s.BM25(); r.index(tok, show_progress=False); r.save(str(IDX / name))

if __name__ == "__main__":
    what = sys.argv[1:] or ["chunks", "acts", "bm25", "cases"]
    acts = pd.read_parquet(P / "acts.parquet")
    acts_text = acts.law.str.slice(0, 6000).tolist()
    c = build_chunks() if "chunks" in what else pd.read_parquet(IDX / "case_chunks.parquet")
    print("case chunks", len(c), c.split.value_counts().to_dict(), flush=True)
    if "bm25" in what or "bm25acts" in what:
        bm25(acts_text, "bm25_acts"); print("bm25 acts done", flush=True)
    if "bm25" in what:
        bm25(c[c.split == "train"].text.tolist(), "bm25_cases"); print("bm25 cases done", flush=True)
    if "acts" in what: embed(acts_text, "emb_acts")
    if "cases" in what: embed(c.text.tolist(), "emb_cases")
    print("DONE", flush=True)
