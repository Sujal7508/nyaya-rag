"""Rebuild the ILDC splits (paper: Malik et al., ACL 2021) from the public CSV and clean them.

CSV row layout (verified against row counts in the paper, Table 1):
  rows 0..32304      ILDC_multi  train
  rows 32305..37386  ILDC_single train
  rows 37387..38903  test (1517 docs, shared by multi and single)
"""
import hashlib, re, sys
from pathlib import Path
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
RAW, OUT = ROOT / "data/raw", ROOT / "data/processed"
OUT.mkdir(parents=True, exist_ok=True)

def norm_hash(t, n=2000):
    t = re.sub(r"\W+", " ", t.lower())
    return hashlib.md5(t[-n:].encode()).hexdigest()

def main():
    d = pd.read_csv(RAW / "ildc_35k.csv", usecols=["id", "text", "label"])
    d["text"] = d.text.astype(str).str.replace(r"\s+", " ", regex=True).str.strip()
    d["h"] = d.text.map(norm_hash)
    d["year"] = d.id.str.extract(r"^(\d{4})_")[0].astype(float)
    multi, single, test = d.iloc[:32305], d.iloc[32305:37387], d.iloc[37387:]
    assert len(test) == 1517
    train = pd.concat([multi.assign(src="multi"), single.assign(src="single")])
    n0 = len(train)
    train = train.drop_duplicates("h")
    # anti-leak: no train doc may share id or text tail with a test doc
    train = train[~train.h.isin(set(test.h)) & ~train.id.isin(set(test.id))]
    print(f"train {n0} -> {len(train)} after dedup / test-overlap removal; test {len(test)}")
    print("train label dist", train.label.value_counts().to_dict(), "test", test.label.value_counts().to_dict())
    train.drop(columns="h").reset_index(drop=True).to_parquet(OUT / "train.parquet")
    test.drop(columns="h").reset_index(drop=True).to_parquet(OUT / "test.parquet")
    ex = pd.read_parquet(RAW / "noleak_expert.parquet")
    ex.to_parquet(OUT / "expert.parquet")
    nl = set(pd.read_parquet(RAW / "noleak_test.parquet").id)
    pd.Series(sorted(nl)).to_csv(OUT / "test_noleak_ids.txt", index=False, header=False)
    acts = pd.read_parquet(RAW / "bare_acts.parquet")
    acts["law"] = acts.law.astype(str).str.strip()
    acts = acts[acts.law.str.len() > 40].reset_index(drop=True)
    acts.to_parquet(OUT / "acts.parquet")
    print("acts", acts.act_title.nunique(), "sections", len(acts))

if __name__ == "__main__":
    main()
