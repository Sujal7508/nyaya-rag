"""CJPE evaluation on the ILDC test set (Malik et al., ACL 2021).

  python src/eval_cjpe.py predict  N     LLM prediction on a stratified random sample of N test cases (seed 0)
  python src/eval_cjpe.py explain        explanation quality on the expert-annotated cases
  python src/eval_cjpe.py report         aggregate everything into results/cjpe.json

Systems on each sampled case:
  llm_zero : LLM reads the case (head+tail clipped), no retrieval
  llm_rag  : + 5 retrieved precedents with outcomes + supervised prior          (ours)
  ensemble : average of llm_rag probability and the local stacked prior         (ours)
Local models (TF-IDF LR, embedding MLP, precedent kNN, stack) are scored on the FULL test set.
"""
import json, sys, re
from pathlib import Path
import numpy as np, pandas as pd
from sklearn.metrics import accuracy_score, precision_recall_fscore_support

ROOT = Path(__file__).resolve().parents[1]
P, RES = ROOT / "data/processed", ROOT / "results"
PAPER = {"XLNet + BiGRU (paper best, ILDC-multi)": {"acc": 77.78, "f1": 77.79}, "XLNet last-512 (paper)": {"acc": 70.01, "f1": 71.07},
         "RoBERTa last-512 (paper)": {"acc": 71.26, "f1": 71.77}, "Doc2Vec + LR (paper)": {"acc": 60.91, "f1": 62.00}, "Human experts (paper, avg of 5)": {"acc": 93.93}}

def metrics(y, p):
    y, p = np.asarray(y), np.asarray(p)
    pr, rc, f1, _ = precision_recall_fscore_support(y, p, average="macro", zero_division=0)
    acc = accuracy_score(y, p); n = len(y); se = (acc * (1 - acc) / n) ** .5
    return {"n": int(n), "acc": round(100 * acc, 2), "macro_p": round(100 * pr, 2), "macro_r": round(100 * rc, 2), "macro_f1": round(100 * f1, 2),
            "acc_ci95": [round(100 * (acc - 1.96 * se), 1), round(100 * (acc + 1.96 * se), 1)]}

def sample_ids(n, seed=0):
    te = pd.read_parquet(P / "test.parquet")
    if n >= len(te): return sample_ids(150, seed) + [i for i in te.id.tolist() if i not in set(sample_ids(150, seed))]
    k = n // 2
    s = pd.concat([te[te.label == 1].sample(k, random_state=seed), te[te.label == 0].sample(n - k, random_state=seed)])
    return s.sample(frac=1, random_state=seed).id.tolist()

def predict(n, workers=4):
    from concurrent.futures import ThreadPoolExecutor
    from llm import LLM
    from retriever import Retriever
    from predict import Predictor, build_local, tail_matrix
    llm, r = LLM(), Retriever(); pr = Predictor(llm, r)
    te = pd.read_parquet(P / "test.parquet").set_index("id"); loc = build_local(); loc = loc[loc.split == "test"].set_index("id")
    c = pd.read_parquet(ROOT / "data/index/case_chunks.parquet"); E = np.load(ROOT / "data/index/emb_cases.npy", mmap_mode="r")
    cte = c[(c.split == "test") & (c.from_end < 6)]
    outp = RES / "cjpe_rows.json"; rows = json.load(open(outp)) if outp.exists() else {}
    ids = [i for i in sample_ids(n) if i not in rows]
    def one(cid):
        text = te.loc[cid, "text"]; ix = cte.index[cte.case_id == cid].values
        v = np.asarray(E[ix], dtype=np.float32).mean(0)
        sims = r.similar_cases_vec(v, k=5)
        z = pr.llm_predict(text)
        g = pr.llm_predict(text, sims, float(loc.loc[cid, "p_local"]))
        return cid, {"label": int(te.loc[cid, "label"]), "zero": {k: z[k] for k in ("ok", "label", "p")}, "rag": {k: g[k] for k in ("ok", "label", "p", "reasoning", "key_sentences")},
                     "p_local": float(loc.loc[cid, "p_local"]), "knn_vote": float(np.mean([l for _, _, l in sims])), "precedents": [(a, round(b, 3), l) for a, b, l in sims]}
    with ThreadPoolExecutor(workers) as ex:
        for k, (cid, row) in enumerate(ex.map(one, ids)):
            rows[cid] = row
            if k % 50 == 0:
                json.dump(rows, open(outp, "w")); y = [x["label"] for x in rows.values()]
                print(len(rows), "zero", round(accuracy_score(y, [x["zero"]["label"] for x in rows.values()]), 3), "rag", round(accuracy_score(y, [x["rag"]["label"] for x in rows.values()]), 3),
                      f"| calls {llm.calls} cached {llm.cached} tok_in {llm.in_tok}", flush=True)
    json.dump(rows, open(outp, "w")); report()

# ---------------- explanation ----------------
EXP_PROMPT = """Below is a Supreme Court of India judgment (final order removed). The case was decided as: {dec}.
Select the sentences that best EXPLAIN why the Court reached this decision - the way a legal expert would highlight them:
the Court's own findings, the decisive legal reasoning and the facts it relied on. Copy each sentence VERBATIM (exact characters) from the case.
Select about {k} sentences, most important first.

CASE:
{case}

Return JSON: {{"sentences": ["..."]}}"""

def explain(k=12):
    from rouge_score import rouge_scorer
    from llm import LLM
    from predict import Predictor, clip_case
    llm = LLM(); pr = Predictor(llm)
    ex = pd.read_parquet(P / "expert.parquet"); sc = rouge_scorer.RougeScorer(["rouge1", "rouge2", "rougeL"], use_stemmer=True)
    tok = lambda s: set(re.findall(r"[a-z0-9]+", s.lower()))
    rows = []
    for r_ in ex.itertuples():
        try:
            z = pr.llm_predict(r_.text)
            dec = "ACCEPTED" if z["label"] else "REJECTED"   # explanation follows the MODEL's own prediction, as in the paper
            j = llm.json(EXP_PROMPT.format(dec=dec, k=k, case=clip_case(r_.text, 1200, 5000)), "You are an expert legal annotator. JSON only.", max_tokens=2500)
        except RuntimeError as e:
            print("stopped early:", str(e)[:80], flush=True); break
        sents = [s for s in (j.get("sentences") or j.get("items") or []) if isinstance(s, str)]
        pred = " ".join(sents); norm = lambda s: re.sub(r"\W+", " ", s.lower()).strip()
        verb = float(np.mean([norm(s)[:80] in norm(r_.text) for s in sents])) if sents else 0.0
        row = {"id": r_.id, "label": int(r_.label), "pred": z["label"], "n_sent": len(sents), "verbatim_rate": verb}
        for e in range(1, 6):
            g = getattr(r_, f"expert_{e}"); gold = " ".join(str(s) for rk in ("rank1", "rank2", "rank3", "rank4", "rank5") for s in (g.get(rk) if g.get(rk) is not None else []))
            if not gold.strip() or not pred.strip(): continue
            s = sc.score(gold, pred); a, b = tok(gold), tok(pred)
            row[f"e{e}"] = {"rouge1": s["rouge1"].fmeasure, "rouge2": s["rouge2"].fmeasure, "rougeL": s["rougeL"].fmeasure, "jaccard": len(a & b) / len(a | b),
                            "overlap_min": len(a & b) / min(len(a), len(b)), "overlap_max": len(a & b) / max(len(a), len(b)), "expert_label": int(g["label"])}
        rows.append(row); print(r_.id, row["pred"] == row["label"], row["n_sent"], round(verb, 2), flush=True)
    json.dump(rows, open(RES / "cjpe_explain_rows.json", "w")); report()

def ablate(n=40, cut=150):
    """Robustness: hide the last `cut` words of each case (where the Court usually states its conclusion)."""
    from llm import LLM
    from predict import Predictor
    llm = LLM(); pr = Predictor(llm)
    te = pd.read_parquet(P / "test.parquet").set_index("id"); rows = json.load(open(RES / "cjpe_rows.json"))
    out = {}
    for cid in sample_ids(150)[:n]:
        try: z = pr.llm_predict(" ".join(te.loc[cid, "text"].split()[:-cut]))
        except RuntimeError as e: print("stopped early:", str(e)[:80], flush=True); break
        out[cid] = {"label": rows[cid]["label"], "full": rows[cid]["zero"]["label"], "cut": z["label"]}
        print(cid, out[cid], flush=True)
    json.dump({"cut_words": cut, "rows": out}, open(RES / "cjpe_ablation_rows.json", "w")); report()

def report():
    out = {"paper": PAPER}
    lp = P / "local_probs.parquet"
    nl = set(open(P / "test_noleak_ids.txt").read().split())
    if lp.exists():
        loc = pd.read_parquet(lp); t = loc[loc.split == "test"]; out["local_full_test"] = {}
        for col, name in (("p_tfidf", "TF-IDF + LR (last 1024 words)"), ("p_emb", "Frozen bge-small tail-chunk embeddings + MLP"), ("p_knn", "Precedent kNN vote (no training)"), ("p_local", "Stacked local prior")):
            out["local_full_test"][name] = metrics(t.label, t[col] > .5)
            out["local_full_test"][name]["acc_noleak_subset"] = metrics(t.label[t.id.isin(nl)], t[col][t.id.isin(nl)] > .5)["acc"]
    rp = RES / "cjpe_rows.json"
    if rp.exists():
        rows = json.load(open(rp)); ids = list(rows); y = [rows[i]["label"] for i in ids]
        pz = np.array([rows[i]["zero"]["p"] for i in ids]); pg = np.array([rows[i]["rag"]["p"] for i in ids]); pl = np.array([rows[i]["p_local"] for i in ids])
        mk = np.array([i in nl for i in ids])
        S = {"LLM zero-shot (long context, no retrieval)": pz > .5, "LLM + retrieved precedents + prior (ours)": pg > .5,
             "Ensemble: LLM-RAG prob avg with local prior (ours)": (pg + pl) / 2 > .5, "Local stacked prior (same cases)": pl > .5}
        out["llm_sample"] = {k: {**metrics(y, v), "acc_noleak_subset": metrics(np.array(y)[mk], v[mk])["acc"], "n_noleak": int(mk.sum())} for k, v in S.items()}
        out["llm_sample"]["_parse_failures"] = {"zero": int(sum(not rows[i]["zero"]["ok"] for i in ids)), "rag": int(sum(not rows[i]["rag"]["ok"] for i in ids))}
    ep = RES / "cjpe_explain_rows.json"
    if ep.exists():
        rows = json.load(open(ep)); out["explanation"] = {"n_docs": len(rows), "pred_acc_on_expert_set": round(100 * float(np.mean([r["pred"] == r["label"] for r in rows])), 2),
            "verbatim_rate": round(float(np.mean([r["verbatim_rate"] for r in rows])), 3), "avg_sentences": round(float(np.mean([r["n_sent"] for r in rows])), 1), "n_empty": int(sum(r["n_sent"] == 0 for r in rows)), "vs_expert": {}}
        for e in range(1, 6):
            out["explanation"]["vs_expert"][f"expert_{e}"] = {m: round(float(np.mean([r[f"e{e}"][m] for r in rows if f"e{e}" in r])), 3) for m in ("jaccard", "overlap_min", "overlap_max", "rouge1", "rouge2", "rougeL")}
        out["explanation"]["paper_table5"] = {"jaccard": [0.333, 0.317, 0.328, 0.324, 0.318], "overlap_min": [0.744, 0.589, 0.81, 0.834, 0.617], "overlap_max": [0.39, 0.414, 0.36, 0.35, 0.401],
                                              "rouge1": [0.444, 0.517, 0.401, 0.391, 0.501], "rouge2": [0.303, 0.295, 0.296, 0.297, 0.294], "rougeL": [0.439, 0.407, 0.423, 0.444, 0.407]}
    ap = RES / "cjpe_ablation_rows.json"
    if ap.exists():
        a = json.load(open(ap)); R = list(a["rows"].values())
        out["ablation_hide_ending"] = {"n": len(R), "cut_words": a["cut_words"], "acc_full_text": round(100 * float(np.mean([x["full"] == x["label"] for x in R])), 2),
                                       "acc_ending_hidden": round(100 * float(np.mean([x["cut"] == x["label"] for x in R])), 2)}
    json.dump(out, open(RES / "cjpe.json", "w"), indent=1)
    print(json.dumps({k: v for k, v in out.items() if k != "paper"}, indent=1))

if __name__ == "__main__":
    cmd = sys.argv[1] if len(sys.argv) > 1 else "report"
    if cmd == "predict": predict(int(sys.argv[2]) if len(sys.argv) > 2 else 200, int(sys.argv[3]) if len(sys.argv) > 3 else 4)
    elif cmd == "explain": explain()
    elif cmd == "ablate": ablate()
    else: report()
