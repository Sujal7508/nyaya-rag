"""Does the self-evaluation module actually catch hallucinations?
Take claims the verifier accepted as supported, corrupt them programmatically (change a section/article
number, a number of years, or flip the outcome / polarity), and re-run the verifier.
  detection rate   = corrupted claims no longer judged 'supported'
  false-alarm rate = untouched supported claims that flip when re-checked alongside a corrupted one
"""
import json, random, re
from pathlib import Path
import numpy as np
from llm import LLM
from retriever import Retriever
from agent import LegalAgent, TAG

ROOT = Path(__file__).resolve().parents[1]
WORDNUM = ["two", "three", "five", "seven", "ten", "fourteen", "twenty"]
FLIPS = [("allowed", "dismissed"), ("dismissed", "allowed"), ("shall not", "shall"), ("is not", "is"), ("cannot", "can"), ("void", "valid"),
         ("death", "a fine only"), ("imprisonment for life", "imprisonment for six months"), ("bailable", "non-bailable"), ("may", "may not")]

def corrupt(claim, rng):
    body = re.sub(TAG, "", claim)
    opts = []
    m = re.search(r"(Section|Article)\s+(\d+)", body)
    if m: opts.append(("section_number", body[:m.start(2)] + str(int(m.group(2)) + rng.choice([7, 13, 21, 40])) + body[m.end(2):]))
    m = re.search(r"\b(" + "|".join(WORDNUM) + r")\b(?=\s+(years|months|days))", body)
    if m: opts.append(("duration", body[:m.start()] + rng.choice([w for w in WORDNUM if w != m.group(1)]) + body[m.end():]))
    m = re.search(r"\b(\d{1,3})\b(?=\s+(years|months|days|per cent|rupees))", body)
    if m: opts.append(("quantity", body[:m.start()] + str(int(m.group(1)) * 3 + 1) + body[m.end():]))
    for a, b in FLIPS:
        m = re.search(r"\b" + re.escape(a) + r"\b", body)
        if m: opts.append(("polarity", body[:m.start()] + b + body[m.end():])); break
    if not opts: return None
    kind, new = rng.choice(opts)
    return kind, new.strip() + " " + "".join(TAG.findall(claim) and [f"[S{t}]" for t in TAG.findall(claim)])

def main():
    rng = random.Random(0); llm, r = LLM(), Retriever(); ag = LegalAgent(llm, r)
    rows = json.load(open(ROOT / "results/rag_eval_rows.json"))
    out = []
    for qid, row in rows.items():
        if row["type"] == "out_of_scope": continue
        t = ag.answer(row["query"])                      # served from cache
        if not t.get("check"): continue
        sup = [c for c in t["check"]["claims"] if c["verdict"] == "supported"]
        if len(sup) < 2: continue
        tgt = None
        for c in rng.sample(sup, len(sup)):
            x = corrupt(c["claim"], rng)
            if x: tgt = (c, x); break
        if not tgt: continue
        c, (kind, new) = tgt
        ans2 = " ".join(new if cl["claim"] == c["claim"] else cl["claim"] for cl in t["check"]["claims"])
        chk2 = ag.self_check(ans2, t["sources"])
        v_new = next((cl["verdict"] for cl in chk2["claims"] if cl["claim"].startswith(new[:40])), "unsupported")
        emb_new = next((cl["emb_support"] for cl in chk2["claims"] if cl["claim"].startswith(new[:40])), None)
        others = [cl["verdict"] for cl in chk2["claims"] if any(cl["claim"] == s["claim"] for s in sup if s is not c)]
        out.append({"id": qid, "kind": kind, "original": c["claim"], "corrupted": new, "verdict_after": v_new, "detected": v_new != "supported",
                    "emb_before": c["emb_support"], "emb_after": emb_new, "others_still_supported": int(sum(o == "supported" for o in others)), "others": len(others)})
        print(qid, kind, v_new, flush=True)
    det = float(np.mean([o["detected"] for o in out])); fa = 1 - sum(o["others_still_supported"] for o in out) / max(1, sum(o["others"] for o in out))
    summ = {"n_corrupted_claims": len(out), "detection_rate": round(det, 3), "false_alarm_rate": round(fa, 3),
            "by_kind": {k: f"{sum(o['detected'] for o in out if o['kind']==k)}/{sum(o['kind']==k for o in out)}" for k in sorted({o['kind'] for o in out})},
            "emb_support_drop_mean": round(float(np.mean([o["emb_before"] - o["emb_after"] for o in out if o["emb_after"] is not None])), 4)}
    json.dump({"summary": summ, "rows": out}, open(ROOT / "results/hallucination.json", "w"), indent=1); print(json.dumps(summ, indent=1))

if __name__ == "__main__":
    main()
