"""Build disjoint pools from public corpora and compose drifting email streams.

Sources
  seven  : puyang2025/seven-phishing-email-datasets (Enron, Ling, SpamAssassin, TREC-05/06/07, CEAS-08)
  gptgen : Dizzzy0x00/LLMGen-Phishing-Email-Dataset (GPT-generated phishing and legitimate, English file)
  spear  : kxm1k4m1/generate_phishing_email_final (OpenAI-generated personalised spear-phishing)
"""
import re
import numpy as np
import pandas as pd
from .config import RAW, CACHE, POOL_SEED

WS = re.compile(r"\s+")
FROM_HDR = re.compile(r"\bFrom:\s*\S+@\S+", re.I)


def norm(t):
    return WS.sub(" ", str(t)).strip()


def load_seven():
    d = pd.concat([pd.read_parquet(RAW / f"seven_{s}.parquet") for s in ("train", "eval", "test")])
    subj = d["subject"].fillna("").map(norm)
    body = d["text"].fillna("").map(norm)
    d = pd.DataFrame({"text": ("Subject: " + subj + " " + body).str.strip(),
                      "label": d["label"].astype(int).values,
                      "source": d["dataset_name"].values})
    d["gen"] = "human"
    return d


def load_gptgen():
    g = pd.read_csv(RAW / "gpt_phish.csv")
    # inline sender addresses exist only in this corpus; remove to avoid a trivial artefact
    t = g["content"].map(lambda s: norm(FROM_HDR.sub(" ", s)))
    return pd.DataFrame({"text": t, "label": g["label"].astype(int), "source": "GPTGen", "gen": "llm"})


def load_spear():
    k = pd.read_json(RAW / "kx_openai.jsonl", lines=True)
    t = k["output"].map(norm)
    t = t[t.str.len() > 80]
    return pd.DataFrame({"text": t.values, "label": 1, "source": "SpearGen", "gen": "llm"})


def build_pools():
    """Disjoint pools; every text appears in at most one pool."""
    out = CACHE / "pools.parquet"
    if out.exists():
        return pd.read_parquet(out)
    rng = np.random.default_rng(POOL_SEED)
    s = load_seven()
    s = s[s.text.str.len() > 40]
    allx = pd.concat([s, load_gptgen(), load_spear()], ignore_index=True)
    allx = allx.drop_duplicates("text").reset_index(drop=True)

    def take(mask, n, tag):
        idx = np.flatnonzero(mask & (allx.pool.values == ""))
        sel = rng.choice(idx, size=min(n, len(idx)), replace=False)
        allx.loc[sel, "pool"] = tag

    allx["pool"] = ""
    src, lab = allx.source.values, allx.label.values
    early = np.isin(src, ["Enron", "Ling", "Assassin"])
    trec = np.isin(src, ["TREC-05", "TREC-06", "TREC-07"])
    ceas = src == "CEAS-08"
    take(early, 4000, "WARM")
    take(early, 8000, "A")
    take(trec, 8000, "B")
    take(ceas, 8000, "C")
    # segment D: LLM-generated campaigns mixed with human and LLM legitimate traffic
    take(src == "GPTGen", 10**9, "_gpt")  # temporarily reserve all GPTGen
    gpt = allx.pool.values == "_gpt"
    allx.loc[gpt, "pool"] = ""
    take(gpt & (lab == 1), 1500, "D")
    take(gpt & (lab == 0), 900, "D")
    take(src == "SpearGen", 1800, "D")
    take((trec | ceas) & (lab == 0), 3500, "D")
    # segment E: paraphrase-evasion campaign (texts are paraphrased later by attacks.py)
    take(gpt & (lab == 1), 500, "E_para")
    take(src == "SpearGen", 500, "E_para")
    take((trec | ceas) & (lab == 1), 500, "E_para")
    take(gpt & (lab == 0), 300, "E_para")          # paraphrased legitimate control
    take((trec | ceas) & (lab == 0), 2700, "E")
    # held-out stress-test pool (never streamed)
    take(gpt & (lab == 1), 200, "STRESS")
    take(src == "SpearGen", 200, "STRESS")
    take((trec | ceas) & (lab == 1), 200, "STRESS")
    allx = allx[allx.pool != ""].reset_index(drop=True)
    allx["uid"] = np.arange(len(allx))
    allx.to_parquet(out)
    return allx


SEGMENTS = [  # (segment, pool, n per seed, description)
    ("A", ["A"], 6000, "human 1999-2004 (Enron/Ling/SpamAssassin)"),
    ("B", ["B"], 6000, "human 2005-2007 (TREC)"),
    ("C", ["C"], 6000, "human 2008 (CEAS)"),
    ("D", ["D"], 6000, "LLM campaigns + mixed legitimate"),
    ("E", ["E", "E_para"], 4000, "LLM paraphrase evasion + legitimate"),
]


def compose_stream(pools, seed):
    """Return the uid order of one stream realisation; segments in fixed order, contents shuffled."""
    rng = np.random.default_rng(seed)
    order, seg = [], []
    for name, pl, n, _ in SEGMENTS:
        cand = pools.index[pools.pool.isin(pl)].values
        if name == "D":  # enforce ~30% prevalence inside D
            pos = cand[pools.label.values[cand] == 1]
            neg = cand[pools.label.values[cand] == 0]
            npos = int(0.3 * n)
            sel = np.concatenate([rng.choice(pos, npos, replace=False), rng.choice(neg, n - npos, replace=False)])
        elif name == "E":
            para = cand[pools.pool.values[cand] == "E_para"]
            rest = cand[pools.pool.values[cand] == "E"]
            npar = min(int(0.85 * len(para)), 1500)
            sel = np.concatenate([rng.choice(para, npar, replace=False),
                                  rng.choice(rest, min(n - npar, int(0.95 * len(rest))), replace=False)])
        else:
            sel = rng.choice(cand, min(n, len(cand)), replace=False)
        rng.shuffle(sel)
        order.append(sel)
        seg += [name] * len(sel)
    return np.concatenate(order), np.array(seg)
