# Nyaya-RAG: An Agentic RAG System for Indian Legal Document Analysis and Research

GenAI course project. Reference paper: Malik et al., *ILDC for CJPE: Indian Legal Documents Corpus for Court
Judgment Prediction and Explanation*, ACL 2021.

The system ingests **31,491 Supreme Court of India judgments** (ILDC training split) and **39,967 sections of
1,034 Indian bare acts** (incl. IPC, CrPC, Evidence Act, Constitution and the 2023 codes BNS / BNSS / BSA with their
official old-to-new section mappings) and supports:

1. **Legal question answering** with citation-grounded answers and a claim-level self-check.
2. **Court judgment prediction and explanation (CJPE)**, the task of the reference paper, done the RAG way
   (retrieved precedents + supervised prior + long-context LLM reader).

Full report: [REPORT.md](REPORT.md). All numbers: [results/RESULTS.md](results/RESULTS.md).

**Live demo:** _link to be added after deployment_

**Headline:** 94.79% accuracy on the full ILDC test set of 1,517 cases (95% CI 93.7-95.9) vs 77.78% for the paper's best model and ~94% for human experts. Hybrid retrieval ranks the correct statute section first for 70% of curated questions (BM25 50%, dense 59%).

## Architecture

```
                         ┌────────────── query ──────────────┐
                         ▼                                   │
                 1. ROUTER (LLM)                             │
   statute | case_law | hybrid | predict | out_of_scope      │
        + query decomposition / abbreviation expansion       │
                         ▼                                   │
     2. RETRIEVAL TOOLS                                      │
        a. citation resolver  "S. 302 IPC" -> exact section  │
        b. hybrid search      BM25 + bge-small dense, RRF    │
           over statutes (section = chunk) and judgments     │
           (head/tail role-tagged chunks, outcome metadata)  │
        c. precedent finder   case-level kNN on tail vectors │
                         ▼                                   │
     3. GENERATOR (LLM)  every sentence tagged [S#]          │
                         ▼                                   │
     4. SELF-CHECK  claim-level: LLM judge + embedding       │
        support + citation-existence check                   │
                         ▼                                   │
     5. CORRECTIVE LOOP  groundedness < 0.75 ->              │
        unsupported claims become new queries, regenerate ───┘
```

For `predict`, the agent calls the CJPE module: 5 nearest precedents with their known outcomes and a supervised
prior (stack of TF-IDF LR, embedding MLP and precedent kNN vote) are injected into the prompt of an LLM that
reads the judgment (final order removed), and returns decision, confidence and verbatim key sentences.

## What is new compared with the paper

| Paper (2021) | This project |
|---|---|
| Fine-tuned XLNet, 512-token windows + BiGRU | Long-context LLM reads head + tail of the case in one pass |
| Each case classified in isolation | **Precedent-augmented prediction**: retrieved similar past cases and their outcomes are evidence |
| Single model | **Prior-injected prompting + ensemble** of LLM confidence with a cheap supervised prior |
| Explanations by occlusion scores on chunks | LLM selects verbatim sentences; scored against the same expert annotations |
| No statutes | Statute corpus with **IPC↔BNS, CrPC↔BNSS, IEA↔BSA cross-references** attached to each section |
| – | **Claim-level hallucination detector** with two independent signals and a corrective loop, tested by injecting synthetic errors |
| Test set contains residual verdict phrases | **Leakage audit**: every accuracy is also reported on the leak-filtered subset |

## Layout

```
app.py                  Streamlit demo
src/prepare_data.py     rebuild ILDC splits from the public CSV, dedup, remove train/test overlap
src/prepare_acts.py     merge two bare-act collections + cross-references
src/build_index.py      document-aware chunking, embeddings, BM25
src/retriever.py        hybrid retrieval, citation resolver, precedent finder
src/llm.py              provider-agnostic client (Gemini API / Vertex AI / Groq / OpenAI / local Ollama), cache, rate limit, model fallback
src/agent.py            router -> retrieve -> generate -> self-check -> correct
src/predict.py          CJPE: local prior models + LLM predictor
src/eval_retrieval.py   Hit@k / MRR on curated statute queries (no LLM)
src/eval_rag.py         vanilla RAG vs agentic: faithfulness, citation validity, relevance, gold-cited
src/eval_hallucination.py  synthetic-error injection test of the self-check
src/eval_cjpe.py        prediction + explanation on ILDC test / expert set, compared with the paper
src/make_results.py     tables and figures
data/eval/legal_queries.json   74 curated queries (56 with gold sections)
data/index_lite/        small index shipped with the repo (all statutes + 5,000 recent cases) used by the hosted demo
results/                JSON outputs, RESULTS.md, figures
```

## Running

```bash
python3 -m venv .venv && .venv/bin/pip install -r requirements.txt
# data (public Hugging Face datasets) goes to data/raw/ - see "Data" below
.venv/bin/python src/prepare_data.py && .venv/bin/python src/prepare_acts.py
.venv/bin/python src/build_index.py            # ~1 h on an M4 laptop
printf "gemini\n<your key>\n" > llm_key.txt    # line 1 provider (gemini/groq/openai), line 2 key; git-ignored
# or: put a Google Cloud service-account JSON (project-*.json) in this folder, Desktop or Downloads to use Vertex AI
cd src
../.venv/bin/python predict.py                 # local prior models
../.venv/bin/python eval_retrieval.py
../.venv/bin/python eval_rag.py
../.venv/bin/python eval_hallucination.py
../.venv/bin/python eval_cjpe.py predict 1517 32
../.venv/bin/python eval_cjpe.py explain
../.venv/bin/python eval_cjpe.py ablate         # hide the last 150 words of each case
../.venv/bin/python make_results.py
cd .. && .venv/bin/streamlit run app.py        # demo
```

## Data

| File in `data/raw/` | Source (Hugging Face dataset) |
|---|---|
| `ildc_35k.csv` | `jayadityagandham9/ILDC_35k_COMPLETE` (ILDC multi-train + single-train + test) |
| `noleak_*.parquet` | `Praful932/il-tur-cjpe-noleak` (leak-filtered ids, dev split, expert annotations) |
| `bare_acts.parquet` | `mratanusarkar/Indian-Laws` |
| `gsms/*.json` | `GSMS-B/indian-legal-knowledge-base-json` (CC-BY-4.0) |

Test cases are never put in the retrieval index; training cases that share an id or a text tail with a test case
are removed (`prepare_data.py`).

## Limitations

- Research aid, not legal advice. The judgment corpus stops at 2020 and has only ILDC ids, not case names.
- Prediction is on the full test set; answer quality was run on 46 of the 74 curated queries.
- The LLM may have seen some public judgments during training; this is not ruled out.
- Models used: gemini-3.5-flash-lite via Vertex AI (prediction, explanation), gemini-3.1-flash-lite (answer quality, hallucination test, hidden-ending test). An earlier free-tier run on a 150-case sample is kept in results/freetier_run/.
- The faithfulness judge is the same LLM family as the generator.
