"""Stage 4b: representation-space displacement of each attack and certified fractions (Prop. 1)."""
import json
import numpy as np
import pandas as pd
import torch
from slide.config import RESULTS
from slide.models import train_head, certified_fraction, predict_proba
from run_experiments import load_features
from slide.stream import DEF

torch.set_num_threads(4)
texts, X, Xnp, pft = load_features()
warm = np.flatnonzero((texts.pool == "WARM").values)
y = texts.label.values
plain = train_head(X[warm], y[warm], seed=0)
robust = train_head(X[warm], y[warm], seed=0, lam_r=0.0, lam_a=DEF["lam_a"], eps=DEF["eps"])
st = texts[texts.pool.isin(["STRESS", "STRESS_ATT"])]
orig = st[st.variant == "orig"].set_index("uid").row
out = {"displacement": {}, "cert": {}}
for v, g in st[st.variant != "orig"].groupby("variant"):
    r_att = g.row.values
    r_org = orig.loc[g.uid.values].values
    dz = np.linalg.norm(X[r_att] - X[r_org], axis=1)
    out["displacement"][v] = dict(median=float(np.median(dz)), q10=float(np.quantile(dz, .1)), q90=float(np.quantile(dz, .9)),
                                  rel_median=float(np.median(dz / np.linalg.norm(X[r_org], axis=1))))
# T1 paraphrase constraint statistics actually achieved by the attacker
para = pd.read_parquet(__import__("slide.config", fromlist=["CACHE"]).CACHE / "paraphrased.parquet")
out["paraphrase"] = dict(n=len(para), sim_median=float(para.sim.median()), sim_q10=float(para.sim.quantile(.1)),
                         frac_sim_ge_0_7=float((para.sim >= 0.7).mean()), word_edit_median=float(para.edit.median()))
ro = orig.values
for name, h in (("plain", plain), ("robust", robust)):
    for eps in (0.25, 0.5, 1.0, 2.0):
        fr, L = certified_fraction(h, X[ro], y[ro], eps)
        out["cert"][f"{name}|eps={eps}"] = fr
    out["cert"][f"{name}|L_m"] = L
    out["cert"][f"{name}|recall_orig"] = float((predict_proba(h, X[ro]) >= .5).mean())
json.dump(out, open(RESULTS / "cert_displacement.json", "w"), indent=1)
print(json.dumps(out, indent=1))
