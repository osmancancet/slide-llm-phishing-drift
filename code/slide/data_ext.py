"""Revision data: two further shift sources, extended stress attacks and alternative stream designs.

New sources (both public, fetched by download_data.py)
  phishfuzzer : simlab-vs/PhishFuzzer_emails_entity_rephrased_v1, LLM-written phishing and valid e-mail
                (6 length/entity variants per seed; we keep ONE variant per seed so no seed appears twice)
  spanish     : Darito/spanish_spear_phishing, Spanish translations (gpt-4o-mini) of LLM spear-phishing,
                real phishing, Enron/SpamAssassin ham and hard ham; every class is machine-translated

Pools added on top of build_pools() (uids continue after the original pools, nothing is re-drawn):
  F        : PhishFuzzer phishing (1 variant per seed) + PhishFuzzer valid + unused human TREC/CEAS ham
  G        : Spanish phishing (spear + traditional) + Spanish ham (Enron + hard ham)
  STRESS2  : 200 held-out PhishFuzzer phishing seeds and 200 held-out Spanish phishing (never streamed)
  NULLH    : unused human ham/malicious from TREC/CEAS for the real-drift probe
"""
import html
import json
import re
from pathlib import Path
import numpy as np
import pandas as pd
from .config import RAW, CACHE, POOL_SEED
from .data import build_pools, norm, SEGMENTS

TAG = re.compile(r"<[^>]+>")
EXTRA = RAW / "extra"


def strip_html(t):
    return norm(html.unescape(TAG.sub(" ", str(t))))


def load_phishfuzzer(rng):
    d = pd.read_parquet(EXTRA / "phishfuzzer.parquet")
    d = d[d.Type.isin(["Phishing", "Valid"])]
    # one variant per seed (uniformly chosen), so a seed never appears twice in the stream or stress pool
    d = d.sample(frac=1.0, random_state=int(rng.integers(1 << 30))).drop_duplicates("Original_ID")
    t = ("Subject: " + d.Subject.fillna("").map(norm) + " " + d.Body.fillna("").map(norm)).str.strip()
    return pd.DataFrame({"text": t.values, "label": (d.Type == "Phishing").astype(int).values,
                         "source": "PhishFuzzer", "gen": "llm", "seed_id": d.Original_ID.values})


def load_spanish():
    rows = []
    for sub, lab in (("spear_phishing", 1), ("traditional_phishing", 1), ("enron_ham", 0), ("hard_ham", 0)):
        for p in sorted((EXTRA / "spanish" / sub).glob("*.json")):
            j = json.load(open(p))
            body = strip_html(j.get("translated_email_body", ""))
            if len(body) < 40:
                continue
            subj = norm(j.get("email_subject", "") or "")
            rows.append((f"Subject: {subj} {body}".strip(), lab, f"ES-{sub}"))
    d = pd.DataFrame(rows, columns=["text", "label", "source"])
    d["gen"] = "llm-translated"
    return d


def build_pools_ext():
    out = CACHE / "pools_ext.parquet"
    if out.exists():
        return pd.read_parquet(out)
    base = build_pools()
    rng = np.random.default_rng(POOL_SEED + 1)
    from .data import load_seven
    s = load_seven()
    s = s[s.text.str.len() > 40]
    s = s[~s.text.isin(set(base.text))].drop_duplicates("text")
    pf = load_phishfuzzer(rng)
    es = load_spanish().drop_duplicates("text")
    new = []
    # PhishFuzzer: 200 phishing seeds held out for stress, the rest to F
    ph = pf[pf.label == 1].sample(frac=1.0, random_state=int(rng.integers(1 << 30)))
    new.append(ph.iloc[:200].assign(pool="STRESS2"))
    new.append(ph.iloc[200:].assign(pool="F"))
    new.append(pf[pf.label == 0].assign(pool="F"))
    hum_ham = s[s.source.isin(["TREC-05", "TREC-06", "TREC-07", "CEAS-08"]) & (s.label == 0)]
    hum_ham = hum_ham.sample(frac=1.0, random_state=int(rng.integers(1 << 30)))
    new.append(hum_ham.iloc[:3000].assign(pool="F_hum"))
    # Spanish: 200 phishing held out for stress
    esp = es[es.label == 1].sample(frac=1.0, random_state=int(rng.integers(1 << 30)))
    new.append(esp.iloc[:200].assign(pool="STRESS2"))
    new.append(esp.iloc[200:].assign(pool="G"))
    new.append(es[es.label == 0].assign(pool="G"))
    # unused human messages for the real-drift probe (TREC, both classes)
    trec = s[s.source.isin(["TREC-05", "TREC-06", "TREC-07"])]
    trec = trec[~trec.text.isin(set(hum_ham.iloc[:3000].text))]
    for lab, n in ((0, 6000), (1, 6000)):
        tt = trec[trec.label == lab].sample(n=n, random_state=int(rng.integers(1 << 30)))
        new.append(tt.assign(pool="NULLH"))
    ext = pd.concat(new, ignore_index=True)
    ext = ext.drop_duplicates("text")
    ext = ext[~ext.text.isin(set(base.text))]
    ext["uid"] = base.uid.max() + 1 + np.arange(len(ext))
    keep = ["uid", "text", "label", "pool", "source", "gen"]
    ext = ext[keep].reset_index(drop=True)
    ext.to_parquet(out)
    return ext


SEGMENTS_EXT = SEGMENTS + [
    ("F", ["F", "F_hum"], 4000, "PhishFuzzer LLM phishing + LLM and human legitimate (new generator family)"),
    ("G", ["G"], 4000, "Spanish machine-translated phishing and legitimate (language shift)"),
]


def _draw(pools, name, pl, n, rng):
    cand = pools.index[pools.pool.isin(pl)].values
    lab = pools.label.values
    if name == "D":
        pos, neg = cand[lab[cand] == 1], cand[lab[cand] == 0]
        npos = int(0.3 * n)
        return np.concatenate([rng.choice(pos, npos, replace=False), rng.choice(neg, n - npos, replace=False)])
    if name == "E":
        para = cand[pools.pool.values[cand] == "E_para"]
        rest = cand[pools.pool.values[cand] == "E"]
        npar = min(int(0.85 * len(para)), 1500)
        return np.concatenate([rng.choice(para, npar, replace=False),
                               rng.choice(rest, min(n - npar, int(0.95 * len(rest))), replace=False)])
    if name == "F":   # 30% LLM phishing, 30% LLM valid, 40% human ham
        pf = cand[(pools.pool.values[cand] == "F")]
        pos, llmneg = pf[lab[pf] == 1], pf[lab[pf] == 0]
        hum = cand[pools.pool.values[cand] == "F_hum"]
        n1, n2 = int(0.3 * n), int(0.3 * n)
        n1, n2 = min(n1, int(0.95 * len(pos))), min(n2, int(0.95 * len(llmneg)))
        return np.concatenate([rng.choice(pos, n1, replace=False), rng.choice(llmneg, n2, replace=False),
                               rng.choice(hum, n - n1 - n2, replace=False)])
    if name == "G":   # 30% phishing
        pos, neg = cand[lab[cand] == 1], cand[lab[cand] == 0]
        npos = int(0.3 * n)
        return np.concatenate([rng.choice(pos, npos, replace=False), rng.choice(neg, n - npos, replace=False)])
    return rng.choice(cand, min(n, len(cand)), replace=False)


def compose_stream_ext(pools, seed, design="abrupt", segments=None):
    """Stream over segments A-G.

    design
      abrupt   : fixed order A..G, abrupt boundaries (main protocol)
      gradual  : as abrupt, but each boundary is a 2,000-message linear mixing zone in which the probability
                 of drawing from the next segment rises from 0 to 1; segment label = majority source
      varlen   : abrupt, each segment length drawn uniformly in [0.5, 1.0] x nominal (pool-bounded)
      permuted : A first, the order of B..G randomly permuted per realisation
    """
    rng = np.random.default_rng(seed + (0 if design == "abrupt" else 1000 * (["gradual", "varlen", "permuted"].index(design) + 1)))
    segs = list(segments or SEGMENTS_EXT)
    if design == "permuted":
        rest = segs[1:]
        segs = [segs[0]] + [rest[i] for i in rng.permutation(len(rest))]
    parts = []
    for name, pl, n, _ in segs:
        if design == "varlen":
            n = int(n * rng.uniform(0.5, 1.0))
        sel = _draw(pools, name, pl, n, rng)
        rng.shuffle(sel)
        parts.append((name, sel))
    if design != "gradual":
        order = np.concatenate([p for _, p in parts])
        seg = np.concatenate([[nm] * len(p) for nm, p in parts])
        return order, seg
    # gradual: interleave the last 1,000 of segment k with the first 1,000 of segment k+1
    Z = 1000
    out_o, out_s = [], []
    cur, cur_name = list(parts[0][1]), parts[0][0]
    for nxt_name, nxt in parts[1:]:
        nxt = list(nxt)
        a_tail, b_head = cur[-Z:], nxt[:Z]
        out_o += cur[:-Z]
        out_s += [cur_name] * (len(cur) - Z)
        ia = ib = 0
        for i in range(2 * Z):
            pb = (i + 0.5) / (2 * Z)
            take_b = (rng.random() < pb and ib < len(b_head)) or ia >= len(a_tail)
            if take_b:
                out_o.append(b_head[ib]); out_s.append(nxt_name); ib += 1
            else:
                out_o.append(a_tail[ia]); out_s.append(cur_name); ia += 1
        cur, cur_name = nxt[Z:], nxt_name
    out_o += cur
    out_s += [cur_name] * len(cur)
    return np.array(out_o), np.array(out_s)


def compose_null_stream(pools, seed, n=20000):
    """stationary stream: i.i.d. draws (without replacement) from the union of pools A, B and C."""
    rng = np.random.default_rng(50000 + seed)
    cand = pools.index[pools.pool.isin(["A", "B", "C"])].values
    sel = rng.choice(cand, n, replace=False)
    return sel, np.array(["N"] * n)


def compose_realdrift_stream(pools, seed, n=4000):
    """real-drift probe on unused human TREC mail (pool NULLH): a labelled warm-up set and two halves R0, R1 are
    drawn i.i.d. from the same pool, so P(X) is unchanged; the caller flips the labels of a fixed region of
    legitimate mail inside R1 only (posterior change). Returns stream order, segment labels and warm-up rows."""
    rng = np.random.default_rng(60000 + seed)
    cand = pools.index[pools.pool == "NULLH"].values
    sel = rng.choice(cand, 3 * n, replace=False)
    return sel[n:], np.array(["R0"] * n + ["R1"] * n), sel[:n]
