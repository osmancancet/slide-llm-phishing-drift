"""Pool composition table (Table 3) from the final text table."""
import pandas as pd
from slide.config import CACHE, RESULTS

t = pd.read_parquet(CACHE / "texts.parquet")
t = t[~t.pool.str.endswith("_unused") & (t.pool != "STRESS_ATT")]
g = t.groupby(["pool", "source", "label"]).size().unstack(fill_value=0)
order = ["WARM", "A", "B", "C", "D", "E", "E_para", "STRESS"]
desc = {"WARM": "warm-up", "A": "segment A", "B": "segment B", "C": "segment C", "D": "segment D", "E": "segment E (orig.)",
        "E_para": "segment E (T1)", "STRESS": "stress pool"}
lines = []
for p in order:
    sub = g.loc[p] if p in g.index.get_level_values(0) else None
    if sub is None:
        continue
    srcs = "; ".join(f"{s} {int(r[0])}/{int(r[1])}" for s, r in sub.iterrows())
    lines.append(f"{desc[p]} & {int(sub[0].sum())} & {int(sub[1].sum())} & {srcs} \\\\")
with open(RESULTS.parent / "manuscript" / "tables" / "pools.tex", "w") as f:
    f.write("\\begin{table}[t]\n\\centering\n\\caption{Disjoint pools (legitimate/malicious counts). Each realisation samples its segments from these pools.}\n\\label{tab:pools}\n\\scriptsize\n\\setlength{\\tabcolsep}{2.5pt}\n\\begin{tabular}{@{}lrr>{\\raggedright\\arraybackslash}p{9cm}@{}}\n\\toprule\nPool & Legit. & Malic. & Sources (legit./malic.) \\\\\n\\midrule\n" + "\n".join(lines) + "\n\\bottomrule\n\\end{tabular}\n\\end{table}\n")
print("\n".join(lines))
