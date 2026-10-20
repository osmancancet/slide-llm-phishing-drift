"""Stage 6: latency/memory and certificate tables, label-cost Pareto figure (reads results of stages 4, 4b and 5)."""
import json
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from slide.config import RESULTS
from run_experiments import MAIN

TAB = RESULTS.parent / "manuscript" / "tables"
FIG = RESULTS.parent / "manuscript" / "figures"
L = json.load(open(RESULTS / "latency.json"))
C = json.load(open(RESULTS / "cert_displacement.json"))
N = json.load(open(RESULTS / "numbers.json"))
S = pd.read_csv(RESULTS / "raw" / "summary.csv")
EV = pd.DataFrame(json.load(open(RESULTS / "raw" / "events.json")))

# ---------------- latency and memory
ev = EV[EV.method == "SLIDE"]
upd_s = float(np.mean([e["t_update"] / max(len(e["events"]), 1) for _, e in ev.iterrows()]))
rows = [
    ("DistilRoBERTa pass $\\phi$ (1 / 4 threads)", f"{L['encoder_ms_1t']:.1f} / {L['encoder_ms_4t']:.1f} ms per e-mail"),
    ("distilgpt2 surprisal $\\psi$ (1 / 4 threads)", f"{L['surprisal_lm_ms_1t']:.1f} / {L['surprisal_lm_ms_4t']:.1f} ms per e-mail"),
    ("Head $h_\\theta$ scoring", f"{L['head_ms'] * 1000:.1f} $\\mu$s per e-mail"),
    ("Drift test, isolated ($n_R{=}1000$, $n_W{=}500$, $K{=}64$)", f"{L['drift_test_ms_per_batch']:.1f} ms per batch ({L['drift_test_ms_per_email']:.2f} ms per e-mail)"),
    ("Drift test, inside the stream", f"{N['t_detect_ms_per_test']:.1f} ms per batch"),
    ("Reference rebuild and calibration ($B{=}100$)", f"{L['reference_build_s']:.2f} s per alarm"),
    ("Head update, plain CE / $\\mathcal{L}_{\\mathrm{total}}$", f"{L['retrain_plain_s']:.2f} / {L['retrain_slide_s']:.2f} s per update"),
    ("Head update inside the stream (mean)", f"{upd_s:.2f} s per update"),
    ("TF-IDF + LR scoring (reference)", f"{L['tfidf_lr_ms']:.2f} ms per e-mail"),
    ("Parameters: encoder / LM / head", f"{L['encoder_params'] / 1e6:.1f}M / {L['lm_params'] / 1e6:.1f}M / {L['head_params'] / 1e3:.0f}k"),
    ("State at capacity: reservoir (2 classes $\\times$ 1{,}000 $\\times$ 772 float32) / reference / projections", f"{2 * 1000 * 772 * 4 / 2 ** 20:.2f} / {L['reference_MB']:.2f} / {L['projection_MB']:.2f} MB"),
]
with open(TAB / "latency.tex", "w") as f:
    f.write("\\begin{table}[t]\n\\centering\n\\caption{Measured cost of each SLIDE component on one " + L["cpu"].split(":")[-1].strip().replace("(R)", "") +
            " (4 cores, no GPU). Transformer timings use batches of 32 e-mails of at most 128 tokens.}\n\\label{tab:latency}\n\\scriptsize\n\\setlength{\\tabcolsep}{3pt}\n\\begin{tabular}{@{}>{\\raggedright\\arraybackslash}p{7cm}>{\\raggedright\\arraybackslash}p{6cm}@{}}\n\\toprule\nComponent & Cost \\\\\n\\midrule\n")
    for a, b in rows:
        f.write(f"{a} & {b} \\\\\n")
    f.write("\\bottomrule\n\\end{tabular}\n\\end{table}\n")

# ---------------- certificate and attack displacement
lab = {"T1_para": "T1 paraphrase", "T2_char_0.05": "T2, $\\epsilon_{\\mathrm{char}}{=}0.05$", "T2_char_0.10": "T2, $\\epsilon_{\\mathrm{char}}{=}0.10$",
       "T2_char_0.20": "T2, $\\epsilon_{\\mathrm{char}}{=}0.20$", "T3_pad": "T3 padding", "T1+T2": "T1$\\circ$T2 (0.10)"}
with open(TAB / "cert.tex", "w") as f:
    f.write("\\begin{table}[t]\n\\centering\n\\caption{Left: displacement $\\lVert\\Phi(x')-\\Phi(x)\\rVert_2$ caused by each attack on the 600 stress e-mails (median, 10th--90th percentile; rel.: median ratio to $\\lVert\\Phi(x)\\rVert_2$). Right: fraction of the stress e-mails that the warm-up head classifies correctly \\emph{and} certifies by Proposition~\\ref{prop:cert} at radius $\\epsilon$, for a plain head and for a head trained with the adversarial term ($\\lambda_a{=}0.5$, $\\epsilon{=}1$).}\n\\label{tab:cert}\n\\scriptsize\n\\setlength{\\tabcolsep}{2.5pt}\n\\begin{tabular}{@{}lrr@{\\hspace{8pt}}lrr@{}}\n\\toprule\nAttack & median (q10--q90) & rel. & radius $\\epsilon$ & plain & adv. \\\\\n\\midrule\n")
    eps = [0.25, 0.5, 1.0, 2.0]
    keys = list(lab)
    for i in range(max(len(keys), len(eps) + 1)):
        left = " & & "
        if i < len(keys):
            d = C["displacement"][keys[i]]
            left = f"{lab[keys[i]]} & {d['median']:.1f} ({d['q10']:.1f}--{d['q90']:.1f}) & {d['rel_median']:.2f}"
        if i < len(eps):
            right = f"{eps[i]:g} & {C['cert'][f'plain|eps={eps[i]}']:.3f} & {C['cert'][f'robust|eps={eps[i]}']:.3f}"
        elif i == len(eps):
            right = f"$L_m$ & {C['cert']['plain|L_m']:.2f} & {C['cert']['robust|L_m']:.2f}"
        elif i == len(eps) + 1:
            right = f"recall & {C['cert']['plain|recall_orig']:.3f} & {C['cert']['robust|recall_orig']:.3f}"
        else:
            right = " & & "
        f.write(f"{left} & {right} \\\\\n")
    f.write("\\bottomrule\n\\end{tabular}\n\\end{table}\n")

# ---------------- label cost versus MCC
fig, ax = plt.subplots(figsize=(3.5, 2.5))
ty = {"Online-TFIDF-SGD": 0.916, "ADWIN-Retrain": 0.903, "PH-Retrain": 0.890, "Online-MLP": 0.877, "Periodic-Retrain": 0.862}
for m in MAIN:
    d = S[(S.method == m) & (S.segment == "ALL")]
    x = max(N[f"{m}|labels"], 1)
    y = d.mcc.values
    ax.errorbar(x, y.mean(), yerr=1.96 * y.std(ddof=1) / np.sqrt(len(y)), fmt="o" if m != "SLIDE" else "*",
                ms=4 if m != "SLIDE" else 9, color="black" if m == "SLIDE" else "tab:blue", capsize=2, lw=0.8)
    if m in ty:
        ax.annotate(m, (x, y.mean()), xytext=(1.3e5, ty[m]), fontsize=5.5, va="center", arrowprops=dict(arrowstyle="-", lw=0.4, color="grey"))
    elif m == "Static-MLP":
        continue
    else:
        name = "Static-FT / Static-MLP" if m == "Static-FT" else m
        off = {"KSWIN-Score": (-52, -3), "SLIDE": (-8, 7), "Static-FT": (6, -2), "TFIDF-LR": (6, -2)}[m]
        ax.annotate(name, (x, y.mean()), fontsize=5.5, xytext=off, textcoords="offset points")
ax.set_xscale("log"); ax.set_xlim(0.6, 3e7); ax.set_ylim(0.675, 0.925); ax.set_xlabel("stream labels consumed (log; 1 = none)", fontsize=7); ax.set_ylabel("stream MCC", fontsize=7)
ax.tick_params(labelsize=6); fig.tight_layout(); fig.savefig(FIG / "pareto.pdf"); fig.savefig(FIG / "pareto.png", dpi=200)
print(open(TAB / "latency.tex").read()); print(open(TAB / "cert.tex").read())
