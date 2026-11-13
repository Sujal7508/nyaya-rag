"""Court Judgment Prediction and Explanation (CJPE) on ILDC, done the RAG way.

Three ingredients, combined:
  (1) supervised prior   - cheap local models trained on 31k past cases (TF-IDF LR, embedding MLP)
  (2) precedent retrieval - nearest past judgments with their known outcomes (kNN vote + shown to the LLM)
  (3) LLM reader          - reads the (long) case, sees precedents, returns decision + confidence +
                            verbatim key sentences (the explanation)
"""
import json, re, sys
from pathlib import Path
import numpy as np, pandas as pd

ROOT = Path(__file__).resolve().parents[1]
P, IDX = ROOT / "data/processed", ROOT / "data/index"
W, NT = 300, 6

def tail_words(t, n): return " ".join(t.split()[-n:])

def clip_case(text, head=600, tail=3400):
    w = text.split()
    if len(w) <= head + tail: return text
    return " ".join(w[:head]) + "\n[... middle of the judgment omitted ...]\n" + " ".join(w[-tail:])

# ------------------------------------------------------------------ local models
def load_dev():
    """IL-TUR 'single_dev' = ILDC dev + test; true dev = rows whose id is not in test."""
    test = pd.read_parquet(P / "test.parquet"); train = pd.read_parquet(P / "train.parquet")
    d = pd.read_parquet(ROOT / "data/raw/noleak_single_dev.parquet")[["id", "text", "label"]]
    d["text"] = d.text.str.replace(r"\s+", " ", regex=True).str.strip()
    d = d[~d.id.isin(set(test.id)) & ~d.id.isin(set(train.id))]
    key = lambda s: s.str[-1500:]
    d = d[~key(d.text).isin(set(key(test.text))) & ~key(d.text).isin(set(key(train.text)))]
    return d.reset_index(drop=True)

def tail_matrix(chunks, E, ids):
    """[n_docs, NT*384]: embeddings of the last NT windows, last window first; zero padded."""
    t = chunks[chunks.from_end < NT]
    pos = {c: i for i, c in enumerate(ids)}
    X = np.zeros((len(ids), NT, E.shape[1]), dtype=np.float32)
    di = t.case_id.map(pos).values; ok = ~np.isnan(di)
    X[di[ok].astype(int), t.from_end.values[ok]] = E[t.index.values[ok]]
    return X

def build_local(force=False):
    out = P / "local_probs.parquet"
    if out.exists() and not force: return pd.read_parquet(out)
    from sklearn.feature_extraction.text import TfidfVectorizer
    from sklearn.linear_model import LogisticRegression
    from sklearn.neural_network import MLPClassifier
    from sentence_transformers import SentenceTransformer
    tr, te, dv = pd.read_parquet(P / "train.parquet"), pd.read_parquet(P / "test.parquet"), load_dev()
    print("train/dev/test", len(tr), len(dv), len(te), flush=True)
    c = pd.read_parquet(IDX / "case_chunks.parquet"); E = np.load(IDX / "emb_cases.npy").astype(np.float32)
    ctr, cte = c[c.split == "train"], c[c.split == "test"]
    Xtr, Xte = tail_matrix(ctr, E, tr.id.tolist()), tail_matrix(cte, E, te.id.tolist())
    enc = SentenceTransformer("BAAI/bge-small-en-v1.5", device="mps"); enc.max_seq_length = 512
    Xdv = np.zeros((len(dv), NT, 384), dtype=np.float32)
    for i, t in enumerate(dv.text):
        w = t.split(); ch = [" ".join(w[max(0, len(w) - (k + 1) * W): len(w) - k * W]) for k in range(min(NT, -(-len(w) // W)))]
        Xdv[i, :len(ch)] = enc.encode(ch, normalize_embeddings=True, batch_size=16)
    res = {"dev": pd.DataFrame({"id": dv.id, "label": dv.label}), "test": pd.DataFrame({"id": te.id, "label": te.label})}
    # (a) TF-IDF + LR on the last 1024 words
    v = TfidfVectorizer(ngram_range=(1, 3), min_df=3, max_features=400000, sublinear_tf=True)
    A = v.fit_transform(tr.text.map(lambda t: tail_words(t, 1024)))
    m = LogisticRegression(C=10, max_iter=2000).fit(A, tr.label)
    for n, d in (("dev", dv), ("test", te)): res[n]["p_tfidf"] = m.predict_proba(v.transform(d.text.map(lambda t: tail_words(t, 1024))))[:, 1]
    print("tfidf done", flush=True)
    # (b) MLP on concatenated tail-chunk embeddings (a light, frozen-encoder cousin of the paper's hierarchical model)
    f = lambda X: X.reshape(len(X), -1)
    mlp = MLPClassifier((512,), alpha=1e-3, early_stopping=True, max_iter=60, random_state=0).fit(f(Xtr), tr.label)
    for n, X in (("dev", Xdv), ("test", Xte)): res[n]["p_emb"] = mlp.predict_proba(f(X))[:, 1]
    print("mlp done", flush=True)
    # (c) precedent vote: similarity-weighted outcome of the 25 nearest training cases
    def unit(X):
        cnt = np.maximum((np.abs(X).sum(-1) > 0).sum(1, keepdims=True), 1); V = X.sum(1) / cnt
        return V / np.maximum(np.linalg.norm(V, axis=1, keepdims=True), 1e-9)
    Vtr, ytr = unit(Xtr), tr.label.values
    for n, X in (("dev", Xdv), ("test", Xte)):
        S = unit(X) @ Vtr.T; ix = np.argpartition(-S, 25, axis=1)[:, :25]
        w = np.take_along_axis(S, ix, 1) ** 4
        res[n]["p_knn"] = (w * ytr[ix]).sum(1) / w.sum(1)
    # stack on dev
    cols = ["p_tfidf", "p_emb", "p_knn"]; lg = lambda p: np.log(np.clip(p, 1e-4, 1 - 1e-4) / (1 - np.clip(p, 1e-4, 1 - 1e-4)))
    st = LogisticRegression().fit(lg(res["dev"][cols].values), res["dev"].label)
    res["test"]["p_local"] = st.predict_proba(lg(res["test"][cols].values))[:, 1]
    from sklearn.model_selection import cross_val_predict
    res["dev"]["p_local"] = cross_val_predict(LogisticRegression(), lg(res["dev"][cols].values), res["dev"].label, cv=5, method="predict_proba")[:, 1]
    allr = pd.concat([res["dev"].assign(split="dev"), res["test"].assign(split="test")]); allr.to_parquet(out)
    for col in cols + ["p_local"]:
        print(col, {s: round(float(((g[col] > .5) == g.label).mean()), 4) for s, g in allr.groupby("split")})
    return allr

# ------------------------------------------------------------------ LLM predictor
SYS = ("You are a senior advocate analysing judgments of the Supreme Court of India. "
       "You will read a judgment whose final operative order has been removed and decide how the Court disposed of it. Reply with JSON only.")
PROMPT = """TASK: Predict the outcome of the CASE below.
Labels (ILDC convention):
- "ACCEPTED": the appeal / petition before the Supreme Court was allowed, fully OR partly (if several appeals are decided together and at least one succeeds, the label is ACCEPTED). The party who approached the Court gets some relief.
- "REJECTED": the appeal / petition was dismissed; the order under challenge stands.
Hints: read the last paragraphs carefully - the Court's own findings matter more than counsel's submissions. Decide who the appellant/petitioner is, then whether the Court agrees with them.
Corpus quirk: the text has systematic artifacts - "number" often stands for "no"/"not" (e.g. "numberjustification" = "no justification") and "company" replaces "co"/"cou" (e.g. "companyrt" = "court", "companynsel" = "counsel").
{extra}
CASE:
{case}

Return JSON with keys in this order:
{{"appellant": "who approached the Supreme Court and what they wanted (one line)",
 "reasoning": "2-3 sentences on what the Court concluded",
 "decision": "ACCEPTED" or "REJECTED",
 "confidence": number between 0.5 and 1.0,
 "key_sentences": ["up to 8 sentences copied VERBATIM from the case that best justify the decision, most important first"]}}"""

def precedent_block(r, sims):
    lines = ["RETRIEVED PRECEDENTS (most similar past Supreme Court cases with their known outcomes; use as weak evidence only):"]
    for i, (cid, s, lab) in enumerate(sims):
        lines.append(f"[P{i+1}] ILDC {cid} | similarity {s:.2f} | outcome: {'ACCEPTED' if lab else 'REJECTED'} | ending: ...{tail_words(r.case_tail(cid), 90)}")
    return "\n".join(lines) + "\n"

class Predictor:
    def __init__(self, llm, retriever=None): self.llm, self.r = llm, retriever

    def llm_predict(self, text, sims=None, prior=None, head=600, tail=3400):
        extra = ""
        if sims: extra += precedent_block(self.r, sims)
        if prior is not None: extra += f"STATISTICAL PRIOR: a classifier trained on 31,000 past cases estimates P(ACCEPTED) = {prior:.2f} for this case. Treat it as a weak prior; the text decides.\n"
        j = self.llm.json(PROMPT.format(extra=extra, case=clip_case(text, head, tail)), SYS, max_tokens=900)
        dec = str(j.get("decision", "")).upper()
        try: conf = min(1.0, max(0.5, float(j.get("confidence", 0.6))))
        except Exception: conf = 0.6
        ok = dec in ("ACCEPTED", "REJECTED")
        p = conf if dec == "ACCEPTED" else 1 - conf
        return {"ok": ok, "label": int(dec == "ACCEPTED"), "p": p if ok else 0.5, "reasoning": j.get("reasoning", ""),
                "appellant": j.get("appellant", ""), "key_sentences": [s for s in (j.get("key_sentences") or []) if isinstance(s, str)]}

    def predict(self, text):
        """Interactive use (demo UI)."""
        sims = self.r.similar_cases(text, k=5)
        out = self.llm_predict(text, sims)
        vote = float(np.mean([l for _, _, l in sims]))
        srcs = [{"type": "case", "key": f"prec:{cid}", "case_id": cid, "title": f"Precedent ILDC {cid} — similarity {s:.2f}; outcome: {'ALLOWED' if l else 'DISMISSED'}",
                 "text": self.r.case_tail(cid, 2), "score": s, "label": l} for cid, s, l in sims]
        ans = (f"**Predicted outcome: {'ACCEPTED (appeal/petition allowed)' if out['label'] else 'REJECTED (appeal/petition dismissed)'}** "
               f"(model confidence {max(out['p'], 1-out['p']):.0%}; {vote:.0%} of the 5 nearest precedents were allowed)\n\n"
               f"{out['reasoning']}\n\n**Sentences that drive the prediction:**\n" + "\n".join(f"- {s}" for s in out["key_sentences"][:6]))
        return {"answer": ans, "sources": srcs, **out}

if __name__ == "__main__":
    build_local(force="--force" in sys.argv)
