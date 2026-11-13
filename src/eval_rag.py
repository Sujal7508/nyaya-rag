"""End-to-end RAG evaluation on the curated query set, with RAGAS-style LLM-judged metrics.

Systems compared:
  vanilla : single dense retrieval -> generate              (classic RAG baseline)
  agentic : route + citation resolver + hybrid + generate + self-check + corrective loop (ours)
Metrics:
  faithfulness      mean claim-level support of the answer by the sources it cites (judge = verifier prompt)
  citation_validity share of claims that carry a valid [S#] tag
  answer_relevance  LLM judge 1-5 -> 0-1
  gold_cited        statute queries: the answer cites the gold section (context+citation correctness)
  routing_acc       router picks the expected route
"""
import json, sys, re
from pathlib import Path
import numpy as np
from llm import LLM
from retriever import Retriever
from agent import LegalAgent, TAG

ROOT = Path(__file__).resolve().parents[1]
REL = """Rate how well the ANSWER addresses the QUESTION on a 1-5 scale (5 = directly and completely answers; 3 = partially; 1 = irrelevant or refuses although the question is answerable). Return JSON {{"score": n, "why": "short"}}.

QUESTION: {q}
ANSWER: {a}"""
EXPECT = {"statute": {"statute", "hybrid"}, "statute_cite": {"statute", "hybrid"}, "case_law": {"case_law", "hybrid"}, "hybrid": {"hybrid"}, "out_of_scope": {"out_of_scope"}}

def main(limit=None):
    llm, r = LLM(), Retriever()
    ag = LegalAgent(llm, r)
    qs = json.load(open(ROOT / "data/eval/legal_queries.json"))
    if limit:  # quota-limited run: every other statute query + all case-law / hybrid / out-of-scope queries
        st_ = [q for q in qs if q["type"] == "statute"][::2]; qs = (st_ + [q for q in qs if q["type"] != "statute"])[:limit]
    outp = ROOT / "results/rag_eval_rows.json"
    rows = json.load(open(outp)) if outp.exists() else {}
    for q in qs:
        if q["id"] in rows: continue
        row = {"id": q["id"], "type": q["type"], "query": q["query"]}
        for name in ("vanilla", "agentic"):
            t = ag.answer(q["query"], mode="dense" if name == "vanilla" else "hybrid", agentic=name == "agentic")
            if q["type"] == "out_of_scope":
                row[name] = {"route": t["plan"]["route"], "answer": t["answer"]}; continue
            chk = t["check"] or ag.self_check(t["answer"], t["sources"])   # same judge for both systems
            rel = llm.json(REL.format(q=q["query"], a=t["answer"]), "You are a strict evaluator. JSON only.", max_tokens=150)
            cited = {int(x) for x in TAG.findall(t["answer"])}
            gold = None
            if q["type"] == "statute":
                gold = any(h.get("act") == q["gold_act"] and h.get("section") in q["gold_sections"] for i, h in enumerate(t["sources"]) if i + 1 in cited)
            row[name] = {"route": t["plan"]["route"], "answer": t["answer"], "faithfulness": chk["groundedness"], "citation_validity": chk["citation_validity"],
                         "emb_support": chk.get("emb_support"), "relevance": (float(rel.get("score", 1)) - 1) / 4, "gold_cited": gold,
                         "corrected": any(s.startswith("correct") for s in t["steps"]), "first_draft_g": (t.get("first_draft") or {}).get("check", {}).get("groundedness"),
                         "n_sources": len(t["sources"]), "sources": [h["title"] for h in t["sources"]]}
        rows[q["id"]] = row
        json.dump(rows, open(outp, "w"), indent=1)
        print(q["id"], {n: {k: v for k, v in row[n].items() if k in ("route", "faithfulness", "relevance", "gold_cited", "corrected")} for n in ("vanilla", "agentic")}, flush=True)
    R = list(rows.values()); summ = {"n": len(R)}
    for name in ("vanilla", "agentic"):
        A = [x[name] for x in R if x["type"] != "out_of_scope"]
        summ[name] = {m: round(float(np.mean([a[m] for a in A if a.get(m) is not None])), 3) for m in ("faithfulness", "citation_validity", "relevance", "gold_cited")}
    summ["agentic"]["routing_acc"] = round(float(np.mean([x["agentic"]["route"] in EXPECT[x["type"]] for x in R])), 3)
    summ["agentic"]["out_of_scope_refused"] = f"{sum(x['agentic']['route']=='out_of_scope' for x in R if x['type']=='out_of_scope')}/{sum(x['type']=='out_of_scope' for x in R)}"
    summ["agentic"]["corrective_loop_fired"] = int(sum(bool(x["agentic"].get("corrected")) for x in R))
    summ["llm"] = {"provider": llm.provider, "model": llm.model}
    json.dump(summ, open(ROOT / "results/rag_eval.json", "w"), indent=1); print(json.dumps(summ, indent=1))

if __name__ == "__main__":
    main(int(sys.argv[1]) if len(sys.argv) > 1 else None)
