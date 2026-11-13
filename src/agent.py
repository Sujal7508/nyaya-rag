"""Agentic RAG for Indian legal research.

Pipeline (each step is logged in a trace the UI shows):
  1. ROUTE      LLM classifies the query -> statute | case_law | hybrid | predict | out_of_scope,
                and decomposes it into focused search queries.
  2. RETRIEVE   exact citation resolver (Section N of Act X) + hybrid BM25/dense RRF on routed corpora.
  3. GENERATE   answer where every sentence carries [S#] source tags.
  4. SELF-CHECK claim-level verification with two independent signals:
                (a) LLM judge: is each claim entailed by the sources it cites?
                (b) embedding support: cosine of claim vs. best sentence of the cited source
                + a hard check that every cited tag exists.
  5. CORRECT    if groundedness is low: re-retrieve using the unsupported claims as queries,
                regenerate once; anything still unsupported is flagged to the user.
"""
import re, json
import numpy as np
from llm import LLM
from retriever import Retriever

ROUTE_SYS = "You are the query router of an Indian legal research assistant. Reply with JSON only."
ROUTE_PROMPT = """Classify the user query and plan retrieval.

Routes:
- "statute": asks what a law / section / article says (definitions, punishments, procedure, rights).
- "case_law": asks how the Supreme Court of India has reasoned or decided on an issue.
- "hybrid": needs both statute text and case law.
- "predict": the user pasted the facts/arguments of a case and wants the likely outcome.
- "out_of_scope": not about Indian law, or small talk.

Return JSON: {{"route": "...", "search_queries": ["1-3 short keyword-rich queries, expand abbreviations (IPC -> Indian Penal Code)"], "reason": "one line"}}

Query: {q}"""

GEN_SYS = ("You are a careful Indian legal research assistant. Answer ONLY from the numbered sources. "
           "Every sentence that states law or facts must end with the tags of the sources that support it, e.g. [S2] or [S1][S4]. "
           "Quote section numbers exactly as in the sources. If the sources do not contain the answer, say so plainly "
           "instead of guessing. Judgment sources contain transcription artifacts ('companyrt' = court, 'number' = no/not); read through them and write clean English. Be concise (max ~180 words). Do not give personal legal advice; this is research support.")

VERIFY_SYS = "You are a strict fact-checker for legal text. Reply with JSON only."
VERIFY_PROMPT = """For each numbered claim decide whether the SOURCES it cites fully support it.
Verdicts: "supported" (source states it), "partial" (source supports part / claim over-generalises),
"unsupported" (source does not state it or contradicts it). Check section numbers, punishments, dates and outcomes exactly.

SOURCES:
{sources}

CLAIMS:
{claims}

Return JSON: {{"claims": [{{"id": 1, "verdict": "...", "why": "short"}}, ...]}}"""

TAG = re.compile(r"\[S(\d+)\]")

def fmt_sources(hits, limit=1400):
    return "\n\n".join(f"[S{i+1}] {h['title']}\n{h['text'][:limit]}" for i, h in enumerate(hits))

def split_claims(answer):
    sents = re.split(r"(?<=[.!?\]])\s+(?=[A-Z(\"'])", answer.replace("\n", " ").strip())
    out = []
    for s in sents:
        s = s.strip(" -*")
        if len(re.sub(TAG, "", s).split()) < 4:
            if out and TAG.search(s): out[-1] += " " + s  # stray tags belong to previous sentence
            continue
        out.append(s)
    return out

class LegalAgent:
    def __init__(self, llm=None, retriever=None, k=6, tau=0.75):
        self.llm, self.r, self.k, self.tau = llm or LLM(), retriever or Retriever(), k, tau

    # 1 ---------------------------------------------------------------
    def route(self, q):
        if len(q.split()) > 250:
            return {"route": "predict", "search_queries": [], "reason": "long case description"}
        j = self.llm.json(ROUTE_PROMPT.format(q=q), ROUTE_SYS, max_tokens=300) or {}
        if j.get("route") not in ("statute", "case_law", "hybrid", "predict", "out_of_scope"): j["route"] = "hybrid"
        j["search_queries"] = [s for s in (j.get("search_queries") or []) if isinstance(s, str)][:3] or [q]
        return j

    # 2 ---------------------------------------------------------------
    def retrieve(self, q, plan, mode="hybrid"):
        corpora = {"statute": ["acts"], "case_law": ["cases"], "hybrid": ["acts", "cases"]}.get(plan["route"], ["acts", "cases"])
        pool, seen = [], set()
        def add(h, boost=0.0):
            if h["key"] not in seen:
                seen.add(h["key"]); h["score"] += boost; pool.append(h)
        for h in self.r.resolve_citations(q): add(h)
        qs = [q] + [s for s in plan["search_queries"] if s.lower() != q.lower()]
        for c in corpora:
            for j, sq in enumerate(qs):
                for h in self.r.search(sq, c, k=self.k, mode=mode): add(h, 0.002 if j == 0 else 0.0)
        pool.sort(key=lambda h: -h["score"])
        if len(corpora) == 2:  # keep both evidence types represented
            a = [h for h in pool if h["type"] == "statute"][: self.k // 2 + 1]; c = [h for h in pool if h["type"] == "case"][: self.k // 2]
            return a + c
        return pool[: self.k]

    # 3 ---------------------------------------------------------------
    def generate(self, q, hits, feedback=""):
        p = f"SOURCES:\n{fmt_sources(hits)}\n\nQUESTION: {q}\n"
        if feedback: p += f"\nA previous draft contained claims the sources do not support:\n{feedback}\nRewrite the answer without them unless a source now supports them.\n"
        return self.llm.complete(p + "\nANSWER (with [S#] tags):", GEN_SYS, max_tokens=700).strip()

    # 4 ---------------------------------------------------------------
    def self_check(self, answer, hits):
        claims = split_claims(answer)
        if not claims: return {"claims": [], "groundedness": 0.0, "citation_validity": 0.0}
        cited = [[int(t) for t in TAG.findall(c)] for c in claims]
        valid = [[t for t in ts if 1 <= t <= len(hits)] for ts in cited]
        used = sorted({t for ts in valid for t in ts})
        src = "\n\n".join(f"[S{t}] {hits[t-1]['title']}\n{hits[t-1]['text'][:2500]}" for t in used) or "(no valid sources cited)"
        j = self.llm.json(VERIFY_PROMPT.format(sources=src, claims="\n".join(f"{i+1}. {c}" for i, c in enumerate(claims))), VERIFY_SYS, max_tokens=900)
        verdicts = {int(x.get("id", 0)): x for x in (j.get("claims") or j.get("items") or []) if isinstance(x, dict) and str(x.get("id", "")).isdigit()}
        emb_c = self.r.embed([re.sub(TAG, "", c) for c in claims], query=False)
        rows = []
        for i, c in enumerate(claims):
            sim = 0.0
            if valid[i]:
                sents = [s for t in valid[i] for s in re.split(r"(?<=[.;])\s+", hits[t-1]["text"][:3500]) if len(s) > 25][:80]
                if sents: sim = float((self.r.embed(sents, query=False) @ emb_c[i]).max())
            v = verdicts.get(i + 1, {})
            verdict = v.get("verdict", "unsupported") if valid[i] else "uncited"
            if cited[i] and len(valid[i]) < len(cited[i]): verdict = "unsupported"  # cites a non-existent source
            rows.append({"claim": c, "cites": cited[i], "verdict": verdict, "why": v.get("why", ""), "emb_support": round(sim, 3)})
        score = {"supported": 1.0, "partial": 0.5}
        g = float(np.mean([score.get(r["verdict"], 0.0) for r in rows]))
        cv = float(np.mean([bool(r["cites"]) and all(1 <= t <= len(hits) for t in r["cites"]) for r in rows]))
        return {"claims": rows, "groundedness": round(g, 3), "citation_validity": round(cv, 3),
                "emb_support": round(float(np.mean([r["emb_support"] for r in rows])), 3)}

    # full loop ---------------------------------------------------------
    def answer(self, q, mode="hybrid", agentic=True):
        trace = {"query": q, "steps": []}
        plan = self.route(q) if agentic else {"route": "hybrid", "search_queries": [q], "reason": "router disabled"}
        trace["plan"] = plan; trace["steps"].append(f"route -> {plan['route']} ({plan.get('reason','')})")
        if plan["route"] == "out_of_scope":
            trace.update(answer="This assistant only covers Indian statutes and Supreme Court case law.", sources=[], check=None); return trace
        if plan["route"] == "predict":
            from predict import Predictor
            p = Predictor(self.llm, self.r).predict(q)
            trace.update(answer=p["answer"], sources=p["sources"], check=None, prediction=p)
            trace["steps"].append("predict -> precedent-augmented judgment prediction"); return trace
        hits = self.retrieve(q, plan, mode)
        trace["steps"].append(f"retrieve -> {len(hits)} sources ({sum(h['type']=='statute' for h in hits)} statute, {sum(h['type']=='case' for h in hits)} case)")
        ans = self.generate(q, hits)
        if not agentic:
            trace.update(answer=ans, sources=hits, check=None); return trace
        chk = self.self_check(ans, hits)
        trace["steps"].append(f"self-check -> groundedness {chk['groundedness']:.2f}, citation validity {chk['citation_validity']:.2f}")
        trace["first_draft"] = {"answer": ans, "check": chk}
        if chk["groundedness"] < self.tau:
            bad = [c for c in chk["claims"] if c["verdict"] in ("unsupported", "uncited", "partial")]
            keys = {h["key"] for h in hits}
            for c in bad[:3]:  # corrective retrieval: the unsupported claim itself becomes the query
                for h in self.retrieve(re.sub(TAG, "", c["claim"]), {"route": plan["route"], "search_queries": []}, mode)[:2]:
                    if h["key"] not in keys: keys.add(h["key"]); hits.append(h)
            fb = "\n".join(f"- {c['claim']} ({c['verdict']}: {c['why']})" for c in bad)
            ans2 = self.generate(q, hits, fb); chk2 = self.self_check(ans2, hits)
            trace["steps"].append(f"correct -> re-retrieved to {len(hits)} sources, regenerated; groundedness {chk2['groundedness']:.2f}")
            if chk2["groundedness"] >= chk["groundedness"]: ans, chk = ans2, chk2
        trace.update(answer=ans, sources=hits, check=chk)
        return trace
