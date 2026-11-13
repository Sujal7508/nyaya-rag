"""Merge two public bare-act collections into one section-level table, and attach the
official IPC<->BNS, CrPC<->BNSS, Evidence Act<->BSA cross-references to each section."""
import json, re, glob
from pathlib import Path
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
RAW, OUT = ROOT / "data/raw", ROOT / "data/processed"
G = RAW / "gsms"
LIST_TITLES = {
    "indian_penal_code_1860": "Indian Penal Code, 1860", "code_of_criminal_procedure_1973": "Code of Criminal Procedure Act, 1973",
    "indian_evidence_act_1872": "Indian Evidence Act, 1872", "negotiable_instruments_act_1881": "Negotiable Instruments Act, 1881",
    "code_of_civil_procedure_1908": "Code of Civil Procedure, 1908", "motor_vehicles_act_1988": "Motor Vehicles Act, 1988",
    "bharatiya_nyaya_sanhita_2023": "Bharatiya Nyaya Sanhita, 2023", "bharatiya_nagarik_suraksha_sanhita_2023": "Bharatiya Nagarik Suraksha Sanhita, 2023",
    "bharatiya_sakshya_adhiniyam_2023": "Bharatiya Sakshya Adhiniyam, 2023"}
XREF = {"ipc_to_bns": "Indian Penal Code, 1860", "crpc_to_bnss": "Code of Criminal Procedure Act, 1973", "iea_to_bsa": "Indian Evidence Act, 1872",
        "bns_to_ipc": "Bharatiya Nyaya Sanhita, 2023", "bnss_to_crpc": "Bharatiya Nagarik Suraksha Sanhita, 2023", "bsa_to_iea": "Bharatiya Sakshya Adhiniyam, 2023"}
key = lambda t: " ".join(w for w in re.sub(r"[^a-z0-9 ]", " ", t.lower()).split() if w not in ("the", "act"))

def main():
    rows = []
    for f in sorted(glob.glob(str(G / "*.json"))):
        n = Path(f).stem; o = json.load(open(f))
        if n in XREF: continue
        if n == "constitution_of_india":
            for a in o["articles"]:
                rows.append(("Constitution of India, 1949", str(a["article_number"]), a.get("article_title", ""), f"Part {a.get('part_number','')} {a.get('part_title','')}", a["text"]))
        elif isinstance(o, dict):
            t = o["act_name"] if str(o["year"]) in o["act_name"] else f"{o['act_name']}, {o['year']}"
            for s in o["sections"]:
                rows.append((t, str(s["section_number"]), s.get("section_title") or "", s.get("chapter_title") or "", s["text"]))
        else:
            t = LIST_TITLES[n]
            for s in o:
                sec = s.get("Section", s.get("section", s.get("section_number")))
                body = s.get("section_desc") or s.get("description") or s.get("text") or ""
                body = re.sub(r"^\[Context:.*?\]\s*", "", body, flags=re.S)
                rows.append((t, str(sec), s.get("section_title") or s.get("title") or "", str(s.get("chapter_title") or s.get("chapter") or ""), body))
    g = pd.DataFrame(rows, columns=["act_title", "section", "sec_title", "chapter", "body"])
    g["section"] = g.section.str.strip().str.rstrip(".")
    g = g[g.body.str.len() > 15].drop_duplicates(["act_title", "section"])
    # cross-references between the colonial-era codes and the 2023 codes
    xr = {}
    for n, act in XREF.items():
        for m in json.load(open(G / f"{n}.json")):
            xr.setdefault((act, str(m["source"]["section"])), []).append(m["mapping_statement"])
    g["xref"] = [" ".join(dict.fromkeys(xr.get((a, s), []))[:3] if False else list(dict.fromkeys(xr.get((a, s), [])))[:3]) for a, s in zip(g.act_title, g.section)]
    g["law"] = [f"{a}\n{('Chapter: ' + c + chr(10)) if c else ''}{'Article' if a.startswith('Constitution') else 'Section'} {s}. {t}\n{b}" + (f"\nCross-reference: {x}" if x else "")
                for a, c, s, t, b, x in zip(g.act_title, g.chapter, g.section, g.sec_title, g.body, g.xref)]
    old = pd.read_parquet(RAW / "bare_acts.parquet"); old["law"] = old.law.astype(str).str.strip(); old["section"] = old.section.astype(str)
    have = set(g.act_title.map(key))
    old = old[(old.law.str.len() > 40) & ~old.act_title.map(key).isin(have)].assign(xref="")
    acts = pd.concat([g[["act_title", "section", "law", "xref"]], old[["act_title", "section", "law", "xref"]]]).reset_index(drop=True)
    acts.to_parquet(OUT / "acts.parquet")
    print("acts", acts.act_title.nunique(), "sections", len(acts), "| curated:", g.act_title.nunique(), len(g), "| with xref:", int((acts.xref != "").sum()))
    for t in ("Indian Penal Code, 1860", "Constitution of India, 1949", "Bharatiya Nyaya Sanhita, 2023", "Indian Evidence Act, 1872"):
        print(" ", t, int((acts.act_title == t).sum()))

if __name__ == "__main__":
    main()
