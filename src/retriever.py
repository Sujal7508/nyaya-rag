"""Hybrid retrieval (BM25 + dense, reciprocal-rank fusion) over statutes and judgments,
plus two document-aware tools: an exact statute-citation resolver and a precedent finder."""
import re
from pathlib import Path
import numpy as np, pandas as pd

ROOT = Path(__file__).resolve().parents[1]
P = ROOT / "data/processed"
# full index if it has been built locally, otherwise the small index shipped with the repository (hosted demo)
IDX = ROOT / "data/index" if (ROOT / "data/index/emb_cases.npy").exists() else ROOT / "data/index_lite"
MODEL = "BAAI/bge-small-en-v1.5"
QPREFIX = "Represent this sentence for searching relevant passages: "

ACT_ALIASES = {
    r"\bipc\b|indian penal code|penal code": "Indian Penal Code, 1860",
    r"\bcr\.?\s?p\.?c\b|code of criminal procedure|criminal procedure code": "Code of Criminal Procedure Act, 1973",
    r"\bc\.?p\.?c\b|code of civil procedure|civil procedure code": "Code of Civil Procedure, 1908",
    r"evidence act": "Indian Evidence Act, 1872",
    r"contract act": "Indian Contract Act, 1872",
    r"constitution": "Constitution of India, 1949",
    r"negotiable instruments|\bni act\b": "Negotiable Instruments Act, 1881",
    r"\bit act\b|information technology act": "Information Technology Act, 2000",
    r"hindu marriage": "Hindu Marriage Act, 1955",
    r"motor vehicles": "Motor Vehicles Act, 1988",
    r"transfer of property": "Transfer of Property Act, 1882",
    r"limitation act": "Limitation Act, 1963",
    r"specific relief": "Specific Relief Act, 1963",
    r"arbitration": "Arbitration and Conciliation Act, 1996",
    r"consumer protection": "Consumer Protection Act, 2019",
    r"companies act": "Companies Act, 2013",
    r"\bbns\b|nyaya sanhita": "Bharatiya Nyaya Sanhita, 2023",
    r"\bbnss\b|nagarik suraksha": "Bharatiya Nagarik Suraksha Sanhita, 2023",
    r"\bbsa\b|sakshya": "Bharatiya Sakshya Adhiniyam, 2023",
}
SEC_RE = re.compile(r"(?:section|sec\.|s\.|u/s|article|art\.)\s*(\d+[A-Za-z]{0,2})", re.I)

class Retriever:
    def __init__(self, load_cases=True):
        import bm25s, Stemmer
        from sentence_transformers import SentenceTransformer
        self.bm25s, self.stem = bm25s, Stemmer.Stemmer("english")
        self.enc = SentenceTransformer(MODEL, device="cpu"); self.enc.max_seq_length = 512
        self.acts = pd.read_parquet(P / "acts.parquet")
        self.acts["section"] = self.acts.section.astype(str)
        self.E = {"acts": np.load(IDX / "emb_acts.npy").astype(np.float32)}
        self.B = {"acts": bm25s.BM25.load(str(IDX / "bm25_acts"))}
        self.titles = set(self.acts.act_title)
        if load_cases:
            c = pd.read_parquet(IDX / "case_chunks.parquet")
            allE = np.load(IDX / "emb_cases.npy", mmap_mode="r")
            tr = (c.split == "train").values
            self.cases = c[tr].reset_index(drop=True)
            self.E["cases"] = np.asarray(allE[tr], dtype=np.float32)
            self.B["cases"] = bm25s.BM25.load(str(IDX / "bm25_cases"))
            # case-level vectors = mean of TAIL chunk embeddings (where the court's reasoning lives)
            t = self.cases[self.cases.role == "tail"]
            g = t.groupby("case_id", sort=False).indices
            self.case_ids = np.array(list(g.keys()))
            tailE = self.E["cases"][t.index.values]
            pos = {i: k for k, i in enumerate(t.index.values)}
            V = np.stack([tailE[v].mean(0) for v in g.values()])
            self.caseV = V / np.linalg.norm(V, axis=1, keepdims=True)
            meta = self.cases.drop_duplicates("case_id").set_index("case_id")
            self.case_label = meta.label.reindex(self.case_ids).values
            self.case_year = meta.year.reindex(self.case_ids).values

    # ---------- basic search ----------
    def embed(self, texts, query=True):
        return self.enc.encode([(QPREFIX + t) if query else t for t in texts], normalize_embeddings=True)

    def _dense(self, q, corpus, k):
        s = self.E[corpus] @ self.embed([q])[0]
        ix = np.argpartition(-s, k)[:k]; ix = ix[np.argsort(-s[ix])]
        return list(ix), list(s[ix])

    def _bm25(self, q, corpus, k):
        tok = self.bm25s.tokenize([q], stopwords="en", stemmer=self.stem, show_progress=False)
        try:
            ix, sc = self.B[corpus].retrieve(tok, k=k, show_progress=False)
        except Exception:
            return [], []
        return [int(i) for i, s in zip(ix[0], sc[0]) if s > 0], [float(s) for s in sc[0] if s > 0]

    def search(self, q, corpus="acts", k=6, mode="hybrid", pool=40, per_case=2):
        """mode: bm25 | dense | hybrid (RRF, k0=60). Returns list of hit dicts."""
        ranks = {}
        if mode in ("dense", "hybrid"):
            for r, i in enumerate(self._dense(q, corpus, pool)[0]): ranks[i] = ranks.get(i, 0) + 1 / (60 + r)
        if mode in ("bm25", "hybrid"):
            for r, i in enumerate(self._bm25(q, corpus, pool)[0]): ranks[i] = ranks.get(i, 0) + 1 / (60 + r)
        out, seen = [], {}
        for i, s in sorted(ranks.items(), key=lambda x: -x[1]):
            h = self.hit(corpus, int(i), s)
            if corpus == "cases":  # diversity: at most `per_case` chunks from one judgment
                seen[h["case_id"]] = seen.get(h["case_id"], 0) + 1
                if seen[h["case_id"]] > per_case: continue
            out.append(h)
            if len(out) == k: break
        return out

    def hit(self, corpus, i, score=1.0):
        if corpus == "acts":
            r = self.acts.iloc[i]
            return {"type": "statute", "key": f"act:{i}", "title": f"{r.act_title} — {'Article' if r.act_title.startswith('Constitution') else 'Section'} {r.section}",
                    "act": r.act_title, "section": r.section, "text": r.law[:3500], "score": float(score)}
        r = self.cases.iloc[i]
        outcome = "appeal/petition ALLOWED" if r.label == 1 else "appeal/petition DISMISSED"
        part = "concluding reasoning" if r.role == "tail" and r.from_end < 2 else ("reasoning" if r.role == "tail" else "facts / opening")
        yr = int(r.year) if r.year == r.year else "n.d."
        return {"type": "case", "key": f"case:{i}", "case_id": r.case_id, "title": f"Supreme Court of India, ILDC case {r.case_id} ({yr}) — {part}; outcome: {outcome}",
                "label": int(r.label), "text": r.text, "score": float(score)}

    # ---------- tool 1: exact statute citation resolver ----------
    def resolve_citations(self, q):
        """'Section 302 IPC', 'Article 21 of the Constitution' -> exact section rows (no embedding needed)."""
        acts = [(m.start(), t) for pat, t in ACT_ALIASES.items() for m in re.finditer(pat, q, re.I) if t in self.titles]
        hits, seen = [], set()
        for m in SEC_RE.finditer(q):
            if m.group(0).lower().startswith("art"): a = "Constitution of India, 1949"
            elif not acts: continue
            else: a = min(acts, key=lambda x: abs(x[0] - m.start()))[1]   # pair each section with the nearest act mention
            row = self.acts[(self.acts.act_title == a) & (self.acts.section.str.upper() == m.group(1).upper())]
            for i in row.index[:1]:
                if i not in seen: seen.add(i); hits.append(self.hit("acts", int(i), 9.0))
        return hits

    # ---------- tool 2: precedent finder (case-level) ----------
    def similar_cases(self, text, k=8, before_year=None):
        """Nearest past judgments to a new case description. Returns (case_id, sim, label)."""
        w = text.split(); W = 300
        chunks = [" ".join(w[max(0, len(w) - (i + 1) * W): len(w) - i * W]) for i in range(min(6, -(-len(w) // W)))]
        return self.similar_cases_vec(self.embed(chunks, query=False).mean(0), k, before_year)

    def similar_cases_vec(self, v, k=8, before_year=None):
        v = v / np.linalg.norm(v); s = self.caseV @ v
        if before_year is not None: s = np.where(self.case_year <= before_year, s, -1)
        ix = np.argsort(-s)[:k]
        return [(self.case_ids[i], float(s[i]), int(self.case_label[i])) for i in ix]

    def case_tail(self, case_id, n=1):
        c = self.cases[(self.cases.case_id == case_id) & (self.cases.from_end < n)].sort_values("from_end", ascending=False)
        return " ".join(c.text)
