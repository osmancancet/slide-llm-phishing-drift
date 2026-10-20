"""Stage 2b: artefact / leakage audit of the corpora (protocol after Cetlenbik & Suzen, Appl. Sci. 2026).

Per source x class: (i) template collapse after entity masking, (ii) near-duplicate structure
(max and 99th-percentile pairwise TF-IDF cosine), (iii) length distribution, and corpus-level
(iv) zero-learning shortcut rules (opening n-gram lookup) and (v) a human-vs-LLM authorship probe.
"""
import json
import re
import numpy as np
import pandas as pd
from scipy.stats import mannwhitneyu
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import cross_val_score, StratifiedKFold
from slide.data import build_pools
from slide.config import RESULTS

MASKS = [(re.compile(r"https?://\S+|www\.\S+"), " <URL> "), (re.compile(r"\S+@\S+\.\S+"), " <EMAIL> "),
         (re.compile(r"\b\d[\d,.:/-]*\b"), " <NUM> "), (re.compile(r"\b[A-Z][a-z]+(?:\s[A-Z][a-z]+)*\b"), " <CAP> ")]


def mask(t):
    for r, s in MASKS:
        t = r.sub(s, t)
    return re.sub(r"\s+", " ", t).strip().lower()


def near_dup(texts, n=1500, seed=0):
    rng = np.random.default_rng(seed)
    t = list(rng.choice(texts, size=min(n, len(texts)), replace=False))
    V = TfidfVectorizer(ngram_range=(1, 2), min_df=1, sublinear_tf=True).fit_transform(t)
    S = (V @ V.T).toarray()
    iu = np.triu_indices(len(t), 1)
    s = S[iu]
    return float(s.max()), float(np.quantile(s, 0.99)), float((s > 0.9).mean())


def main():
    p = build_pools()
    p = p[p.pool.isin(["WARM", "A", "B", "C", "D", "E", "E_para", "STRESS"])]
    rows = []
    for (src, lab), g in p.groupby(["source", "label"]):
        tx = g.text.values
        m = pd.Series([mask(t) for t in tx])
        mx, q99, frac = near_dup(tx)
        L = g.text.str.len()
        rows.append(dict(source=src, label=int(lab), n=len(g), unique_masked=int(m.nunique()),
                         masked_ratio=m.nunique() / len(g), unique_opening5=int(m.map(lambda s: " ".join(s.split()[:5])).nunique()),
                         max_cos=mx, q99_cos=q99, frac_cos_gt_0_9=frac, len_median=float(L.median()),
                         len_iqr=float(L.quantile(.75) - L.quantile(.25))))
    tab = pd.DataFrame(rows)
    out = {"per_source": tab.to_dict("records")}
    # length balance inside GPTGen (phishing vs legitimate)
    g = p[p.source == "GPTGen"]
    out["gptgen_length_mwu_p"] = float(mannwhitneyu(g[g.label == 1].text.str.len(), g[g.label == 0].text.str.len()).pvalue)
    # zero-learning opening-lookup rule: label = majority label of the first 3 masked tokens seen in a train split
    def lookup_rule(df, k=3, seed=0):
        df = df.sample(frac=1, random_state=seed)
        key = df.text.map(lambda s: " ".join(mask(s).split()[:k]))
        n = len(df) // 2
        tr = pd.DataFrame({"k": key[:n].values, "y": df.label.values[:n]})
        maj = tr.groupby("k").y.mean()
        prior = tr.y.mean()
        pred = key[n:].map(lambda z: maj.get(z, prior)).values >= 0.5
        yt = df.label.values[n:]
        from sklearn.metrics import f1_score
        return float(f1_score(yt, pred, average="macro")), float(key[n:].isin(maj.index).mean())
    out["lookup_rule_gptgen"] = lookup_rule(g)
    d = p[p.pool == "D"]
    out["lookup_rule_segmentD"] = lookup_rule(d)
    out["lookup_rule_segmentC"] = lookup_rule(p[p.pool == "C"])
    # authorship probe: can a linear TF-IDF model tell LLM from human text in legitimate mail only?
    leg = p[(p.label == 0) & p.pool.isin(["D", "E"])]
    leg = pd.concat([leg[leg.gen == "llm"], leg[leg.gen == "human"].sample(len(leg[leg.gen == "llm"]) * 2, random_state=0)])
    X = TfidfVectorizer(ngram_range=(1, 2), min_df=2, sublinear_tf=True).fit_transform(leg.text)
    yy = (leg.gen == "llm").astype(int).values
    cv = StratifiedKFold(5, shuffle=True, random_state=0)
    out["authorship_probe_f1macro"] = float(cross_val_score(LogisticRegression(max_iter=2000), X, yy, cv=cv, scoring="f1_macro").mean())
    tab.to_csv(RESULTS / "leakage_audit.csv", index=False)
    json.dump(out, open(RESULTS / "leakage_audit.json", "w"), indent=1)
    print(tab.round(3).to_string())
    print({k: v for k, v in out.items() if k != "per_source"})


if __name__ == "__main__":
    main()
