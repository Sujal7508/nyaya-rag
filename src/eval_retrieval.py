"""Retrieval quality on the curated statute queries (no LLM involved): Hit@k and MRR for
BM25 vs dense vs hybrid-RRF. 'Lenient' also accepts the official BNS/BNSS/BSA counterpart of the gold section."""
import json, re
from pathlib import Path
import numpy as np
from retriever import Retriever

ROOT = Path(__file__).resolve().parents[1]

def main():
    r = Retriever(load_cases=False)
    qs = [q for q in json.load(open(ROOT / "data/eval/legal_queries.json")) if q["type"] == "statute"]
    xref = {(a, s): x for a, s, x in zip(r.acts.act_title, r.acts.section, r.acts.xref)}
    res = {}
    for mode in ("bm25", "dense", "hybrid"):
        ranks, ranks_len = [], []
        for q in qs:
            hits = r.search(q["query"], "acts", k=10, mode=mode)
            gold = {(q["gold_act"], s) for s in q["gold_sections"]}
            alt = set(gold)
            for a, s in gold:  # counterpart section in the 2023 codes / old codes
                m = re.search(r"corresponds to (.+?) Section (\w+)", xref.get((a, s), ""))
                if m: alt.add((m.group(1), m.group(2)))
            def rank(G): return next((i + 1 for i, h in enumerate(hits) if (h["act"], h["section"]) in G), None)
            ranks.append(rank(gold)); ranks_len.append(rank(alt))
        def summ(rk):
            return {"hit@1": np.mean([x == 1 for x in rk]), "hit@3": np.mean([x is not None and x <= 3 for x in rk]),
                    "hit@5": np.mean([x is not None and x <= 5 for x in rk]), "hit@10": np.mean([x is not None for x in rk]),
                    "mrr@10": np.mean([1 / x if x else 0 for x in rk])}
        res[mode] = {"strict": {k: round(float(v), 3) for k, v in summ(ranks).items()},
                     "lenient": {k: round(float(v), 3) for k, v in summ(ranks_len).items()},
                     "misses": [q["id"] for q, x in zip(qs, ranks_len) if x is None or x > 5]}
        print(mode, res[mode])
    res["n_queries"] = len(qs); res["n_sections_indexed"] = int(len(r.acts))
    json.dump(res, open(ROOT / "results/retrieval.json", "w"), indent=1)

if __name__ == "__main__":
    main()
