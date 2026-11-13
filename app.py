"""Streamlit demo:  .venv/bin/streamlit run app.py"""
import sys, re
from pathlib import Path
sys.path.insert(0, str(Path(__file__).parent / "src"))
import os, tempfile
import streamlit as st

# hosted demo: credentials come from Streamlit secrets instead of local files
def _gcp_secret():
    """Service-account key from Streamlit secrets: a JSON string (GCP_SERVICE_ACCOUNT_JSON) or a TOML table ([gcp_service_account])."""
    import json
    for name in ("GCP_SERVICE_ACCOUNT_JSON", "gcp_service_account"):
        if name not in st.secrets: continue
        v = st.secrets[name]
        if not isinstance(v, str): return dict(v)
        v = v.strip()
        try: return json.loads(v)
        except Exception: return json.loads(v, strict=False)   # tolerate line breaks pasted inside the private key
    return None

try:
    if not os.environ.get("GOOGLE_APPLICATION_CREDENTIALS"):
        _key = _gcp_secret()
        if _key:
            import json
            f = tempfile.NamedTemporaryFile("w", suffix=".json", delete=False); json.dump(_key, f); f.close()
            os.environ["GOOGLE_APPLICATION_CREDENTIALS"] = f.name
    for k in ("LLM_API_KEY", "LLM_PROVIDER", "LLM_MODEL"):
        if k in st.secrets: os.environ.setdefault(k, str(st.secrets[k]))
except Exception as _e:
    st.warning(f"Could not read the LLM credentials from secrets ({type(_e).__name__}). Check the Secrets box in the app settings.")

st.set_page_config(page_title="Nyaya-RAG: Indian Legal Research Agent", layout="wide")

@st.cache_resource(show_spinner="Loading indexes (first run takes ~20 s)...")
def load():
    from llm import LLM
    from retriever import Retriever
    from agent import LegalAgent
    return LegalAgent(LLM(), Retriever())

st.title("Nyaya-RAG")
st.caption("Agentic retrieval-augmented research over 40k sections of Indian bare acts and Supreme Court judgments (ILDC). "
           "Research aid only, not legal advice.")
ag = load()
with st.sidebar:
    st.subheader("Settings")
    mode = st.radio("Retrieval", ["hybrid", "bm25", "dense"], horizontal=True)
    agentic = st.toggle("Agentic pipeline (router + self-check + correction)", True)
    st.markdown(f"LLM: `{ag.llm.provider}` / `{ag.llm.model}`" + (f" (fallbacks: {', '.join(ag.llm.fallbacks)})" if ag.llm.fallbacks else ""))
    st.subheader("Try")
    EX = ["What is the punishment for murder and what is the equivalent section in the Bharatiya Nyaya Sanhita?",
          "How can a person who fears arrest get anticipatory bail, and what factors has the Supreme Court considered?",
          "What does Section 27 of the Evidence Act allow?",
          "When does the Supreme Court interfere with concurrent findings of fact?",
          "What is the best recipe for butter chicken?"]
    for e in EX:
        if st.button(e, use_container_width=True): st.session_state["q"] = e
    st.caption("Paste the full text of a judgment (without its final order) to get an outcome prediction with precedents.")

q = st.text_area("Ask a question about Indian law, or paste a case description", value=st.session_state.get("q", ""), height=110)
if st.button("Research", type="primary") and q.strip():
    with st.spinner("Routing, retrieving, generating, verifying..."):
        try:
            t = ag.answer(q.strip(), mode=mode, agentic=agentic)
        except Exception as e:
            msg = str(e)
            if "QUOTA" in msg or "429" in msg:
                st.error("The LLM provider's free daily quota is used up for every configured model. "
                         "Retrieval still works; try again after the quota resets, set LLM_MODEL to another model, "
                         "or run a local model with LLM_PROVIDER=ollama.")
            else:
                st.error(f"The LLM call failed: {msg[:300]}")
            st.stop()
    left, right = st.columns([3, 2])
    with left:
        st.subheader("Answer")
        st.markdown(t["answer"])
        chk = t.get("check")
        if chk:
            a, b, c = st.columns(3)
            a.metric("Groundedness (LLM judge)", f"{chk['groundedness']:.0%}")
            b.metric("Citation validity", f"{chk['citation_validity']:.0%}")
            c.metric("Embedding support", f"{chk.get('emb_support', 0):.2f}")
            st.subheader("Claim-level self-check")
            icon = {"supported": "✅", "partial": "🟡", "unsupported": "❌", "uncited": "⚠️"}
            for cl in chk["claims"]:
                st.markdown(f"{icon.get(cl['verdict'], '❔')} {cl['claim']}  \n<small>{cl['verdict']} · embedding support {cl['emb_support']:.2f}"
                            + (f" · {cl['why']}" if cl.get("why") else "") + "</small>", unsafe_allow_html=True)
        with st.expander("Agent trace"):
            for s in t["steps"]: st.write("•", s)
            st.json(t.get("plan", {}))
            if t.get("first_draft") and t["first_draft"]["answer"] != t["answer"]:
                st.markdown("**First draft (before correction):**"); st.write(t["first_draft"]["answer"])
    with right:
        st.subheader("Sources")
        for i, h in enumerate(t.get("sources") or []):
            with st.expander(f"[S{i+1}] {h['title']}"):
                st.write(h["text"][:3000])
