"""Turn results/*.json into results/RESULTS.md and two figures."""
import json
from pathlib import Path
import matplotlib; matplotlib.use("Agg")
import matplotlib.pyplot as plt

ROOT = Path(__file__).resolve().parents[1]; RES = ROOT / "results"
load = lambda n: json.load(open(RES / n)) if (RES / n).exists() else None

def main():
    md = ["# Results\n", "All numbers below are produced by the scripts in `src/` and stored as JSON in this folder.\n"]
    cj, rt, rg, hl = load("cjpe.json"), load("retrieval.json"), load("rag_eval.json"), load("hallucination.json")
    if cj:
        md += ["## 1. Judgment prediction on the ILDC test set (paper: Malik et al., ACL 2021)\n", "| System | Test cases | Accuracy % | Macro-F1 % | 95% CI (acc) | Acc. on leak-filtered subset % |", "|---|---|---|---|---|---|"]
        bars = []
        for k, v in cj["paper"].items():
            md.append(f"| {k} | 1517 | {v['acc']} | {v.get('f1','-')} | - | - |"); bars.append((k, v["acc"], "#9aa0a6"))
        for k, v in (cj.get("local_full_test") or {}).items():
            md.append(f"| {k} | {v['n']} | {v['acc']} | {v['macro_f1']} | {v['acc_ci95'][0]}-{v['acc_ci95'][1]} | {v['acc_noleak_subset']} |"); bars.append((k, v["acc"], "#5b8def"))
        for k, v in (cj.get("llm_sample") or {}).items():
            if k.startswith("_"): continue
            md.append(f"| **{k}** | {v['n']}{'' if v['n'] >= 1517 else ' (stratified sample)'} | **{v['acc']}** | {v['macro_f1']} | {v['acc_ci95'][0]}-{v['acc_ci95'][1]} | {v['acc_noleak_subset']} (n={v['n_noleak']}) |"); bars.append((k, v["acc"], "#e8710a"))
        if cj.get("llm_sample"): md.append(f"\nJSON parse failures (counted as wrong/0.5): {cj['llm_sample']['_parse_failures']}\n")
        ab = cj.get("ablation_hide_ending")
        if ab: md.append(f"Robustness ({ab["n"]} of the sampled cases, zero-shot; hidden-ending run on gemini-3.1-flash-lite, full-text run on gemini-3.5-flash-lite): with the last {ab['cut_words']} words of each case hidden, accuracy goes from {ab['acc_full_text']}% to {ab['acc_ending_hidden']}%.\n")
        bars.sort(key=lambda b: b[1])
        fig, ax = plt.subplots(figsize=(9, 0.42 * len(bars) + 1))
        ax.barh([b[0] for b in bars], [b[1] for b in bars], color=[b[2] for b in bars])
        for i, b in enumerate(bars): ax.text(b[1] + 0.4, i, f"{b[1]:.1f}", va="center", fontsize=8)
        ax.set_xlim(50, 100); ax.set_xlabel("Accuracy on ILDC test (%)"); ax.axvline(77.78, ls="--", c="k", lw=0.8)
        ax.set_title("grey = reported in paper, blue = our local models, orange = our LLM systems", fontsize=9)
        plt.tight_layout(); plt.savefig(RES / "fig_cjpe.png", dpi=160); plt.close()
        md.append("![CJPE accuracy](fig_cjpe.png)\n")
        ex = cj.get("explanation")
        if ex:
            md += ["## 2. Explanation quality vs. legal experts (expert-annotated ILDC cases)\n",
                   f"{ex['n_docs']} cases; our explanations = sentences the LLM copies from the judgment (avg {ex['avg_sentences']} sentences, {ex['verbatim_rate']:.0%} found verbatim in the source). "
                   f"Prediction accuracy on these cases: {ex['pred_acc_on_expert_set']}%.\n", "| Metric | " + " | ".join(f"E{e} ours / paper" for e in range(1, 6)) + " | mean ours / paper |", "|---|" + "---|" * 6]
            for m in ("rouge1", "rouge2", "rougeL", "jaccard", "overlap_min", "overlap_max"):
                o = [ex["vs_expert"][f"expert_{e}"][m] for e in range(1, 6)]; p = ex["paper_table5"][m]
                md.append(f"| {m} | " + " | ".join(f"{a:.3f} / {b:.3f}" for a, b in zip(o, p)) + f" | **{sum(o)/5:.3f}** / {sum(p)/5:.3f} |")
            md.append("")
    if rt:
        md += ["## 3. Statute retrieval (curated queries with gold sections, no LLM)\n", f"{rt['n_queries']} plain-language questions over {rt['n_sections_indexed']} sections. Lenient = the official BNS/BNSS/BSA counterpart of the gold section also counts.\n",
               "| Retriever | Hit@1 | Hit@3 | Hit@5 | MRR@10 | Hit@1 (lenient) | Hit@5 (lenient) | MRR@10 (lenient) |", "|---|---|---|---|---|---|---|---|"]
        for m in ("bm25", "dense", "hybrid"):
            s, l = rt[m]["strict"], rt[m]["lenient"]
            md.append(f"| {m} | {s['hit@1']} | {s['hit@3']} | {s['hit@5']} | {s['mrr@10']} | {l['hit@1']} | {l['hit@5']} | {l['mrr@10']} |")
        fig, ax = plt.subplots(figsize=(6, 3)); ks = ["hit@1", "hit@3", "hit@5", "mrr@10"]
        for i, (m, c) in enumerate((("bm25", "#9aa0a6"), ("dense", "#5b8def"), ("hybrid", "#e8710a"))):
            ax.bar([x + i * 0.27 for x in range(4)], [rt[m]["strict"][k] for k in ks], 0.27, label=m, color=c)
        ax.set_xticks([x + 0.27 for x in range(4)]); ax.set_xticklabels(ks); ax.set_ylim(0, 1); ax.legend(); plt.tight_layout(); plt.savefig(RES / "fig_retrieval.png", dpi=160); plt.close()
        md.append("\n![retrieval](fig_retrieval.png)\n")
    if rg:
        md += ["## 4. End-to-end answer quality (RAGAS-style, LLM-judged)\n", f"{rg['n']} curated queries; judge and generator: `{rg['llm']['model']}`.\n",
               "| System | Faithfulness | Citation validity | Answer relevance | Gold section cited |", "|---|---|---|---|---|"]
        for n, lab in (("vanilla", "Vanilla RAG (dense top-k -> generate)"), ("agentic", "Agentic (ours)")):
            v = rg[n]; md.append(f"| {lab} | {v['faithfulness']} | {v['citation_validity']} | {v['relevance']} | {v['gold_cited']} |")
        a = rg["agentic"]; md.append(f"\nRouter accuracy {a['routing_acc']}; out-of-scope refused {a['out_of_scope_refused']}; corrective loop fired on {a['corrective_loop_fired']} queries.\n")
    if hl:
        s = hl["summary"]
        md += ["## 5. Hallucination detection (self-evaluation module)\n", f"{s['n_corrupted_claims']} supported claims were corrupted programmatically (wrong section number, wrong duration/quantity, flipped polarity). "
               f"Detection rate **{s['detection_rate']:.0%}**, false-alarm rate on untouched claims {s['false_alarm_rate']:.0%}. By kind: {s['by_kind']}. Mean drop in embedding-support score: {s['emb_support_drop_mean']}.\n"]
    (RES / "RESULTS.md").write_text("\n".join(md)); print("\n".join(md))

if __name__ == "__main__":
    main()
