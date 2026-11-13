"""Provider-agnostic LLM client with disk cache, retry and rate-limit backoff.

Key lookup order: env LLM_API_KEY -> ./llm_key.txt -> ~/Desktop/llm_key.txt
The file may be just the key, or "provider\nkey". Provider is inferred from the key prefix.
"""
import hashlib, json, os, re, sqlite3, threading, time
from pathlib import Path
import requests

ROOT = Path(__file__).resolve().parents[1]
CACHE = ROOT / "data/llm_cache.sqlite"
_lock = threading.Lock()

def _find_gcp_key():
    """Service-account JSON for Vertex AI: env var, or gcp_key.json / project-*.json in the project folder, Desktop or Downloads."""
    import glob
    c = [os.environ.get("GOOGLE_APPLICATION_CREDENTIALS", "")]
    for d in (ROOT, Path.home() / "Desktop", Path.home() / "Downloads"):
        c += [str(d / "gcp_key.json")] + sorted(glob.glob(str(d / "project-*.json")))
    return next((x for x in c if x and Path(x).exists()), None)

def _load_key():
    if os.environ.get("LLM_PROVIDER", "") in ("", "vertex") and not os.environ.get("LLM_API_KEY") and _find_gcp_key():
        return "vertex", _find_gcp_key()
    if os.environ.get("LLM_PROVIDER", "") == "ollama":
        return "ollama", "ollama"
    if os.environ.get("LLM_API_KEY"):
        return os.environ.get("LLM_PROVIDER", ""), os.environ["LLM_API_KEY"].strip()
    for p in (ROOT / "llm_key.txt", Path.home() / "Desktop/llm_key.txt"):
        if p.exists():
            lines = [l.strip() for l in p.read_text().splitlines() if l.strip()]
            if lines:
                return (lines[0].lower() if len(lines) > 1 else ""), lines[-1]
    raise RuntimeError("No LLM key found. Put it in llm_key.txt (project folder or Desktop).")

def _infer(provider, key):
    if provider: return provider.split()[0]
    if key.startswith("AIza"): return "gemini"
    if key.startswith("gsk_"): return "groq"
    if key.startswith("sk-or-"): return "openrouter"
    return "openai"

OPENAI_STYLE = {
    "ollama": os.environ.get("OLLAMA_URL", "http://localhost:11434/v1"),
    "groq": "https://api.groq.com/openai/v1",
    "openai": "https://api.openai.com/v1",
    "openrouter": "https://openrouter.ai/api/v1",
}

class LLM:
    def __init__(self, model=None, min_interval=None):
        prov, self.key = _load_key()
        self.provider = _infer(prov, self.key)
        self.model = model or os.environ.get("LLM_MODEL") or self._default_model()
        self.min_interval = min_interval if min_interval is not None else float(os.environ.get("LLM_MIN_INTERVAL", "4.4" if self.provider == "gemini" else "0"))  # free tier: 15 req/min
        if self.provider == "vertex":
            import json as _j
            self.gcp_project = os.environ.get("GCP_PROJECT") or _j.load(open(self.key))["project_id"]
            self.gcp_location = os.environ.get("GCP_LOCATION", "global")
        dflt = "gemini-3.1-flash-lite,gemini-3-flash-preview" if self.provider == "gemini" else ""
        if self.provider == "vertex" and not os.environ.get("LLM_API_KEY"):   # optional: free-tier Gemini key is not used when Vertex is configured
            pass
        self.fallbacks = [m for m in os.environ.get("LLM_FALLBACKS", dflt).split(",") if m and m != self.model]; self.by_model = {}
        self._last = 0.0
        self.calls = self.cached = self.in_tok = self.out_tok = 0
        self.db = sqlite3.connect(CACHE, check_same_thread=False, timeout=60)
        self.db.execute("create table if not exists c (k text primary key, v text)")

    # ---------- model discovery ----------
    def list_models(self):
        if self.provider == "gemini":
            r = requests.get("https://generativelanguage.googleapis.com/v1beta/models?pageSize=200",
                             headers={"x-goog-api-key": self.key}, timeout=30)
            r.raise_for_status()
            return [m["name"].split("/")[-1] for m in r.json().get("models", [])
                    if "generateContent" in m.get("supportedGenerationMethods", [])]
        r = requests.get(OPENAI_STYLE[self.provider] + "/models", headers={"Authorization": f"Bearer {self.key}"}, timeout=30)
        r.raise_for_status(); return [m["id"] for m in r.json()["data"]]

    def _default_model(self):
        return {"vertex": "gemini-3.5-flash-lite", "ollama": "qwen3:8b", "gemini": "gemini-3.5-flash-lite", "groq": "llama-3.3-70b-versatile", "openai": "gpt-4o-mini", "openrouter": "google/gemini-2.5-flash"}[self.provider]

    def _gcp_token(self):
        from google.oauth2 import service_account
        import google.auth.transport.requests as gr
        if not getattr(self, "_cred", None):
            self._cred = service_account.Credentials.from_service_account_file(self.key, scopes=["https://www.googleapis.com/auth/cloud-platform"])
        if not self._cred.valid: self._cred.refresh(gr.Request())
        return self._cred.token

    # ---------- raw call ----------
    def _call(self, system, prompt, json_mode, max_tokens, temperature):
        if self.provider in ("gemini", "vertex"):
            body = {"contents": [{"role": "user", "parts": [{"text": prompt}]}],
                    "generationConfig": {"temperature": temperature, "maxOutputTokens": max_tokens + 3000}}  # head-room for hidden thinking tokens
            if system: body["systemInstruction"] = {"parts": [{"text": system}]}
            if json_mode: body["generationConfig"]["responseMimeType"] = "application/json"
            if self.provider == "vertex":
                host = "aiplatform.googleapis.com" if self.gcp_location == "global" else f"{self.gcp_location}-aiplatform.googleapis.com"
                url = f"https://{host}/v1/projects/{self.gcp_project}/locations/{self.gcp_location}/publishers/google/models/{self.model}:generateContent"
                hdr = {"Authorization": "Bearer " + self._gcp_token()}
            else:
                url, hdr = f"https://generativelanguage.googleapis.com/v1beta/models/{self.model}:generateContent", {"x-goog-api-key": self.key}
            r = requests.post(url, headers=hdr, json=body, timeout=180)
            if r.status_code == 400 and "thinking" in r.text.lower():
                body["generationConfig"].pop("thinkingConfig", None)
                r = requests.post(r.request.url, headers={"x-goog-api-key": self.key}, json=body, timeout=180)
            if r.status_code != 200: raise _HTTP(r)
            j = r.json(); u = j.get("usageMetadata", {})
            self.in_tok += u.get("promptTokenCount", 0); self.out_tok += u.get("candidatesTokenCount", 0)
            parts = (j.get("candidates") or [{}])[0].get("content", {}).get("parts", [])
            return "".join(p.get("text", "") for p in parts if not p.get("thought"))
        msgs = ([{"role": "system", "content": system}] if system else []) + [{"role": "user", "content": prompt}]
        body = {"model": self.model, "messages": msgs, "temperature": temperature, "max_tokens": max_tokens}
        if json_mode: body["response_format"] = {"type": "json_object"}
        r = requests.post(OPENAI_STYLE[self.provider] + "/chat/completions", json=body, timeout=180,
                          headers={"Authorization": f"Bearer {self.key}"})
        if r.status_code != 200: raise _HTTP(r)
        j = r.json(); u = j.get("usage", {})
        self.in_tok += u.get("prompt_tokens", 0); self.out_tok += u.get("completion_tokens", 0)
        return j["choices"][0]["message"]["content"] or ""

    # ---------- public ----------
    def complete(self, prompt, system="", json_mode=False, max_tokens=1024, temperature=0.0, use_cache=True):
        k = hashlib.sha256(json.dumps([self.provider, self.model, system, prompt, json_mode, max_tokens, temperature]).encode()).hexdigest()
        if use_cache:
            with _lock:
                row = self.db.execute("select v from c where k=?", (k,)).fetchone()
            if row: self.cached += 1; return row[0]
        delay = 4.0
        for attempt in range(8):
            with _lock:
                wait = self._last + self.min_interval - time.time()
                self._last = max(time.time(), self._last + self.min_interval)
            if wait > 0: time.sleep(wait)
            try:
                out = self._call(system, prompt, json_mode, max_tokens, temperature)
                self.calls += 1; self.by_model[self.model] = self.by_model.get(self.model, 0) + 1
                if out.strip():
                    with _lock:
                        self.db.execute("insert or replace into c values (?,?)", (k, out)); self.db.commit()
                return out
            except _HTTP as e:
                if e.status in (429, 500, 502, 503, 529) and attempt < 7:
                    if e.status == 429 and e.daily:
                        if self.fallbacks:
                            print(f"[llm] daily quota hit on {self.model}; switching to {self.fallbacks[0]}", flush=True)
                            self.model = self.fallbacks.pop(0); continue
                        raise RuntimeError("DAILY_QUOTA_EXHAUSTED: " + e.text[:300])
                    time.sleep(e.retry_after or delay); delay = min(delay * 2, 60); continue
                raise
            except requests.RequestException:
                if attempt == 7: raise
                time.sleep(delay); delay = min(delay * 2, 60)

    def json(self, prompt, system="", **kw):
        raw = self.complete(prompt, system, json_mode=True, **kw)
        j = parse_json(raw)
        return j if isinstance(j, dict) else {"items": j if isinstance(j, list) else []}

class _HTTP(Exception):
    def __init__(self, r):
        self.status, self.text = r.status_code, r.text
        m = re.search(r'"retryDelay":\s*"(\d+)', r.text) or re.search(r"try again in ([\d.]+)s", r.text)
        self.retry_after = float(m.group(1)) + 1 if m else (float(r.headers["retry-after"]) + 1 if r.headers.get("retry-after", "").replace(".", "").isdigit() else None)
        self.daily = "PerDay" in r.text or "per day" in r.text.lower() or "(TPD)" in r.text or "(RPD)" in r.text
        m2 = re.search(r"retry in (?:(\d+)h)?(?:(\d+)m)?([\d.]+)s", r.text)
        if m2 and not m: self.retry_after = int(m2.group(1) or 0) * 3600 + int(m2.group(2) or 0) * 60 + float(m2.group(3)) + 1
        super().__init__(f"HTTP {r.status_code}: {r.text[:400]}")

def parse_json(raw):
    raw = raw.strip()
    raw = re.sub(r"^```(?:json)?|```$", "", raw, flags=re.M).strip()
    try: return json.loads(raw)
    except Exception:
        m = re.search(r"[\[{].*[\]}]", raw, re.S)
        if m:
            try: return json.loads(m.group(0))
            except Exception: pass
    return {}

if __name__ == "__main__":
    l = LLM(); print(l.provider, l.model); print(l.list_models()[:60]); print(l.complete("Say OK."))
