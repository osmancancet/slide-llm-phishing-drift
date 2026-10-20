"""Leakage-audit table (Sec. 6.1) from results/leakage_audit.{csv,json}."""
import json
import pandas as pd
from slide.config import RESULTS

t = pd.read_csv(RESULTS / "leakage_audit.csv")
j = json.load(open(RESULTS / "leakage_audit.json"))
lines = []
for _, r in t.iterrows():
    lines.append(f"{r.source} & {'mal.' if r.label == 1 else 'leg.'} & {r.n} & {r.masked_ratio:.3f} & {r.unique_opening5} & {r.max_cos:.3f} & {r.q99_cos:.3f} & {r.len_median:.0f} & {r.len_iqr:.0f} \\\\")
with open(RESULTS.parent / "manuscript" / "tables" / "audit.tex", "w") as f:
    f.write("\\begin{table}[t]\n\\centering\n\\caption{Leakage and artefact audit per source and class. Masked ratio: unique entity-masked texts / $n$; Open.$_5$: distinct masked 5-token openings; cos: pairwise TF-IDF cosine (max and 99th percentile, up to 1{,}500 sampled texts); length in characters (median, IQR).}\n\\label{tab:audit}\n\\scriptsize\n\\setlength{\\tabcolsep}{2.2pt}\n\\begin{tabular}{@{}llrrrrrrr@{}}\n\\toprule\nSource & Cl. & $n$ & Masked & Open.$_5$ & cos$_{\\max}$ & cos$_{.99}$ & Len. & IQR \\\\\n\\midrule\n" + "\n".join(lines) + "\n\\bottomrule\n\\end{tabular}\n\\end{table}\n")
print(j["gptgen_length_mwu_p"], j["lookup_rule_gptgen"], j["lookup_rule_segmentD"], j["lookup_rule_segmentC"], j["authorship_probe_f1macro"])
