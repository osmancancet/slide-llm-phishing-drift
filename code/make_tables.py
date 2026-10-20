"""Stage 5: LaTeX tables, figures and numbers.json from the raw experiment outputs."""
import json
import sys
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from slide.config import RESULTS, CACHE, BATCH
from slide.stats import holm, wilcoxon_rb, friedman_nemenyi, mean_ci
from run_experiments import MAIN, ABL, SENS, CTRL

TAG = sys.argv[1] if len(sys.argv) > 1 else ""
RAW = RESULTS / f"raw{TAG}"
TAB = RESULTS.parent / "manuscript" / "tables"
FIG = RESULTS.parent / "manuscript" / "figures"
TAB.mkdir(exist_ok=True, parents=True)
FIG.mkdir(exist_ok=True, parents=True)
S = pd.read_csv(RAW / "summary.csv")
B = pd.read_csv(RAW / "batch.csv")
ST = pd.read_csv(RAW / "stress.csv")
EV = pd.DataFrame(json.load(open(RAW / "events.json")))
N = {}


def fmt(m, lo, hi, d=3):
    return f"{m:.{d}f}\\,$\\pm$\\,{(hi - lo) / 2:.{d}f}"


def ci(x, d=3):
    m, lo, hi = mean_ci(x)
    return fmt(m, lo, hi, d)


def tex_name(n):
    return n.replace("label_win=", "LABELWIN=").replace("_", "\\_").replace("w/o", "w/o").replace("psi", "$\\psi$").replace("eps=", "$\\epsilon$=").replace("alpha=", "$\\alpha$=").replace("LABELWIN=", "$n_\\ell$=").replace("K=", "$K$=")


def seed_metric(method, metric, seg="ALL"):
    d = S[(S.method == method) & (S.segment == seg)].sort_values("seed")
    return d[metric].values


ev = EV.groupby("method").agg(labels=("stream_labels", "mean"), ms=("ms_score", "mean"),
                               updates=("events", lambda e: np.mean([len(x) for x in e])),
                               t_update=("t_update", "mean"), t_detect=("t_detect", "mean"))

# ---------------- main table
rows = []
for m in MAIN:
    r = {"Method": m}
    for k in ("precision", "recall", "f1", "mcc", "pr_auc", "fpr", "brier"):
        r[k] = ci(seed_metric(m, k)) if k in S.columns else "--"
        N[f"{m}|{k}"] = float(np.mean(seed_metric(m, k))) if k in S.columns else None
    r["labels"] = f"{ev.loc[m, 'labels']:.0f}"
    r["upd"] = "every batch" if m in ("Online-TFIDF-SGD", "Online-MLP") else f"{ev.loc[m, 'updates']:.1f}"
    N[f"{m}|labels"] = float(ev.loc[m, "labels"])
    N[f"{m}|updates"] = float(ev.loc[m, "updates"])
    rows.append(r)
with open(TAB / "main.tex", "w") as f:
    f.write("\\begin{table*}[t]\n\\centering\n\\caption{Prequential stream-level results (mean\\,$\\pm$\\,95\\% CI half-width over 10 realisations, 28{,}000 messages each). Labels: stream labels consumed (warm-up excluded); Upd.: head updates per stream (online methods update after every delayed batch).}\n\\label{tab:main}\n\\scriptsize\n\\resizebox{\\textwidth}{!}{\\begin{tabular}{@{}lccccccccr@{}}\n\\toprule\nMethod & Precision & Recall & F1 & MCC & PR-AUC & FPR & Brier & Labels & Upd. \\\\\n\\midrule\n")
    for r in rows:
        name = "\\textbf{SLIDE}" if r["Method"] == "SLIDE" else r["Method"]
        f.write(f"{name} & {r['precision']} & {r['recall']} & {r['f1']} & {r['mcc']} & {r['pr_auc']} & {r['fpr']} & {r['brier']} & {r['labels']} & {r['upd']} \\\\\n")
    f.write("\\bottomrule\n\\end{tabular}}\n\\end{table*}\n")

# ---------------- per segment MCC
with open(TAB / "segments.tex", "w") as f:
    f.write("\\begin{table}[t]\n\\centering\n\\caption{MCC per stream segment (mean over 10 realisations). Best value per column in bold.}\n\\label{tab:segments}\n\\scriptsize\n\\setlength{\\tabcolsep}{3.5pt}\n\\begin{tabular}{@{}lccccc@{}}\n\\toprule\nMethod & A & B & C & D & E \\\\\n\\midrule\n")
    M = {m: [np.mean(seed_metric(m, "mcc", g)) for g in "ABCDE"] for m in MAIN}
    best = np.max(np.array(list(M.values())), 0)
    for m in MAIN:
        cells = [("\\textbf{%.3f}" if abs(v - b) < 5e-4 else "%.3f") % v for v, b in zip(M[m], best)]
        f.write(f"{m} & " + " & ".join(cells) + " \\\\\n")
        for g, v in zip("ABCDE", M[m]):
            N[f"{m}|mcc|{g}"] = float(v)
    f.write("\\bottomrule\n\\end{tabular}\n\\end{table}\n")

# ---------------- slices inside D and E
sl = S[S.segment.str.contains(":")]
with open(TAB / "slices.tex", "w") as f:
    f.write("\\begin{table}[t]\n\\centering\n\\caption{Recall on malicious and FPR on legitimate messages by authorship inside segments D and E (mean over realisations; segment D contains no human-written malicious mail). ``llm'' = LLM-written, ``para'' = T1-paraphrased (malicious and the legitimate control).}\n\\label{tab:slices}\n\\scriptsize\n\\setlength{\\tabcolsep}{2.5pt}\n\\begin{tabular}{@{}lccccc@{}}\n\\toprule\n & \\multicolumn{2}{c}{Recall} & \\multicolumn{3}{c}{FPR} \\\\\n\\cmidrule(lr){2-3}\\cmidrule(lr){4-6}\nMethod & D:llm & E:para & D:llm & D:human & E:para \\\\\n\\midrule\n")
    for m in MAIN:
        g = sl[sl.method == m].groupby("segment")[["recall", "fpr"]].mean()
        get = lambda s, k: g.loc[s, k] if s in g.index else np.nan
        vals = [get("D:llm", "recall"), get("D:human", "recall"), get("E:para", "recall"),
                get("D:llm", "fpr"), get("D:human", "fpr"), get("E:para", "fpr")]
        for nm, v in zip(["rec_Dllm", "rec_Dhuman", "rec_Epara", "fpr_Dllm", "fpr_Dhuman", "fpr_Epara"], vals):
            N[f"{m}|{nm}"] = float(v)
        f.write(f"{m} & " + " & ".join("--" if np.isnan(v) else f"{v:.3f}" for i, v in enumerate(vals) if i != 1) + " \\\\\n")
    f.write("\\bottomrule\n\\end{tabular}\n\\end{table}\n")

# ---------------- statistics: Friedman + Nemenyi + Wilcoxon (seed level, batch level)
Mseed = np.column_stack([seed_metric(m, "mcc") for m in MAIN])
chi, pf, ranks, cd = friedman_nemenyi(Mseed)
N.update(friedman_chi2=chi, friedman_p=pf, nemenyi_cd=cd)
for m, r in zip(MAIN, ranks):
    N[f"{m}|rank"] = float(r)
comp = [m for m in MAIN if m != "SLIDE"]
res = []
for m in comp:
    p, rb, n = wilcoxon_rb(seed_metric("SLIDE", "mcc"), seed_metric(m, "mcc"))
    bs = B[B.method == "SLIDE"].sort_values(["seed", "batch"]).mcc.values
    bm = B[B.method == m].sort_values(["seed", "batch"]).mcc.values
    pb, rbb, nb = wilcoxon_rb(bs, bm)
    pr_s = seed_metric("SLIDE", "pr_auc"); pr_m = seed_metric(m, "pr_auc")
    pp, rbp, _ = wilcoxon_rb(pr_s, pr_m)
    res.append(dict(m=m, d=np.mean(seed_metric("SLIDE", "mcc") - seed_metric(m, "mcc")), p=p, rb=rb, pb=pb, rbb=rbb, pp=pp, rbp=rbp,
                    wins=int(np.sum(seed_metric("SLIDE", "mcc") > seed_metric(m, "mcc")))))
res = pd.DataFrame(res)
res["p_holm"] = holm(res.p.values)
res["pp_holm"] = holm(res.pp.values)
res["pb_holm"] = holm(res.pb.values)
with open(TAB / "stats.tex", "w") as f:
    f.write("\\begin{table}[t]\n\\centering\n\\caption{SLIDE versus each competitor. $\\Delta$MCC: mean paired difference; wins out of 10 realisations; two-sided Wilcoxon signed-rank tests with Holm correction on realisation-level MCC and PR-AUC ($n=10$, primary) and on batch-level MCC ($n=1120$, descriptive); $r_{rb}$: rank-biserial effect size. Friedman $\\chi^2_F=%.2f$, $p=%s$; Nemenyi CD$_{0.05}=%.2f$.}\n\\label{tab:stats}\n\\scriptsize\n\\setlength{\\tabcolsep}{2.5pt}\n\\begin{tabular}{@{}lrrrrrrrr@{}}\n\\toprule\nCompetitor & $\\Delta$MCC & wins & $p_{\\mathrm{Holm}}$ & $r_{rb}$ & $p^{\\mathrm{PR}}_{\\mathrm{Holm}}$ & $p^{\\mathrm{batch}}_{\\mathrm{Holm}}$ & $r^{\\mathrm{batch}}_{rb}$ & rank \\\\\n\\midrule\n" % (chi, ('%.1e' % pf).replace('e-', r'\times10^{-') + '}', cd))
    for _, r in res.iterrows():
        f.write(f"{r.m} & {r.d:+.3f} & {r.wins} & {r.p_holm:.3g} & {r.rb:+.2f} & {r.pp_holm:.3g} & {r.pb_holm:.2g} & {r.rbb:+.2f} & {N[r.m + '|rank']:.1f} \\\\\n")
        for k in ("d", "p", "p_holm", "rb", "pb_holm", "rbb", "pp_holm", "wins"):
            N[f"vs|{r.m}|{k}"] = float(r[k])
    f.write(f"\\midrule\nSLIDE (rank) & & & & & & & & {N['SLIDE|rank']:.1f} \\\\\n\\bottomrule\n\\end{{tabular}}\n\\end{{table}}\n")

# ---------------- ablation
abl = ["SLIDE"] + list(ABL)
with open(TAB / "ablation.tex", "w") as f:
    f.write("\\begin{table}[t]\n\\centering\n\\caption{Ablation over 10 realisations. $\\Delta$: change of mean MCC relative to full SLIDE; $p$: Holm-corrected two-sided Wilcoxon test against full SLIDE on realisation-level MCC.}\n\\label{tab:ablation}\n\\scriptsize\n\\setlength{\\tabcolsep}{2.5pt}\n\\begin{tabular}{@{}lccccrr@{}}\n\\toprule\nVariant & MCC & PR-AUC & MCC (E) & Labels & $\\Delta$ & $p_{\\mathrm{Holm}}$ \\\\\n\\midrule\n")
    base = seed_metric("SLIDE", "mcc")
    ps = holm([wilcoxon_rb(base, seed_metric(a, "mcc"))[0] for a in abl[1:]])
    for a, p in zip(abl, [np.nan] + list(ps)):
        x = seed_metric(a, "mcc")
        N[f"abl|{a}|mcc"] = float(x.mean()); N[f"abl|{a}|p"] = float(p) if not np.isnan(p) else None
        N[f"abl|{a}|mccE"] = float(np.mean(seed_metric(a, "mcc", "E"))); N[f"abl|{a}|labels"] = float(ev.loc[a, "labels"])
        N[f"abl|{a}|pr"] = float(np.mean(seed_metric(a, "pr_auc")))
        f.write(f"{tex_name(a)} & {ci(x)} & {np.mean(seed_metric(a, 'pr_auc')):.3f} & {np.mean(seed_metric(a, 'mcc', 'E')):.3f} & {ev.loc[a, 'labels']:.0f} & "
                + ("--" if np.isnan(p) else f"{x.mean() - base.mean():+.3f}") + " & " + ("--" if np.isnan(p) else f"{p:.3g}") + " \\\\\n")
    f.write("\\bottomrule\n\\end{tabular}\n\\end{table}\n")

# ---------------- equal-budget and update-rule controls
ctrl = ["SLIDE", "KSWIN-Score"] + CTRL
with open(TAB / "controls.tex", "w") as f:
    f.write("\\begin{table}[t]\n\\centering\n\\caption{Controls that separate the timing of label purchases from the update rule (10 realisations). All rows except KSWIN-Score use the SLIDE update; Budget-Random buys four 500-label windows at random batches, Budget-Oracle buys them as soon as 500 messages of each new segment have arrived. $p_{\\mathrm{Holm}}$: two-sided Wilcoxon test against SLIDE on realisation-level MCC, Holm-corrected over the four rows.}\n\\label{tab:controls}\n\\scriptsize\n\\setlength{\\tabcolsep}{2.5pt}\n\\begin{tabular}{@{}lcccccr@{}}\n\\toprule\nMethod & MCC & MCC (D) & MCC (E) & Labels & $\\Delta$ & $p_{\\mathrm{Holm}}$ \\\\\n\\midrule\n")
    base = seed_metric("SLIDE", "mcc")
    ps = holm([wilcoxon_rb(base, seed_metric(a, "mcc"))[0] for a in ctrl[1:]])
    for a, p in zip(ctrl, [np.nan] + list(ps)):
        x = seed_metric(a, "mcc")
        N[f"ctrl|{a}|mcc"] = float(x.mean()); N[f"ctrl|{a}|p"] = None if np.isnan(p) else float(p)
        N[f"ctrl|{a}|mccD"] = float(np.mean(seed_metric(a, "mcc", "D"))); N[f"ctrl|{a}|mccE"] = float(np.mean(seed_metric(a, "mcc", "E")))
        N[f"ctrl|{a}|labels"] = float(ev.loc[a, "labels"]); N[f"ctrl|{a}|wins"] = float(np.sum(base > x))
        f.write(f"{a} & {ci(x)} & {N[f'ctrl|{a}|mccD']:.3f} & {N[f'ctrl|{a}|mccE']:.3f} & {ev.loc[a, 'labels']:.0f} & "
                + ("--" if np.isnan(p) else f"{x.mean() - base.mean():+.3f}") + " & " + ("--" if np.isnan(p) else f"{p:.3g}") + " \\\\\n")
    f.write("\\bottomrule\n\\end{tabular}\n\\end{table}\n")

# ---------------- sensitivity
sens = [s for s in SENS if s in set(S.method)]
with open(TAB / "sensitivity.tex", "w") as f:
    f.write("\\begin{table}[t]\n\\centering\n\\caption{Sensitivity of SLIDE (5 realisations; default: $\\epsilon=1$, $\\alpha=0.01$, $n_\\ell=500$, $K=64$; first row shows the default on the same 5 realisations).}\n\\label{tab:sens}\n\\scriptsize\n\\begin{tabular}{@{}lcccr@{}}\n\\toprule\nSetting & MCC & MCC (E) & FPR & Labels \\\\\n\\midrule\n")
    d5 = S[(S.method == "SLIDE") & (S.segment == "ALL") & (S.seed < 5)]
    d5e = S[(S.method == "SLIDE") & (S.segment == "E") & (S.seed < 5)]
    l5 = EV[(EV.method == "SLIDE") & (EV.seed < 5)].stream_labels.mean()
    f.write(f"default & {d5.mcc.mean():.3f} & {d5e.mcc.mean():.3f} & {d5.fpr.mean():.3f} & {l5:.0f} \\\\\n")
    N["sens|default5|mcc"] = float(d5.mcc.mean())
    for s in sens:
        a = S[(S.method == s) & (S.segment == "ALL")]
        e = S[(S.method == s) & (S.segment == "E")]
        l = EV[EV.method == s].stream_labels.mean()
        N[f"sens|{s}|mcc"] = float(a.mcc.mean()); N[f"sens|{s}|labels"] = float(l); N[f"sens|{s}|mccE"] = float(e.mcc.mean())
        f.write(f"{tex_name(s).replace('SLIDE ', '')} & {a.mcc.mean():.3f} & {e.mcc.mean():.3f} & {a.fpr.mean():.3f} & {l:.0f} \\\\\n")
    f.write("\\bottomrule\n\\end{tabular}\n\\end{table}\n")

# ---------------- stress test
VAR = ["orig", "T1_para", "T2_char_0.05", "T2_char_0.10", "T2_char_0.20", "T3_pad", "T1+T2"]
VN = ["orig.", "T1", "T2 .05", "T2 .10", "T2 .20", "T3", "T1$\\circ$T2"]
STm = ST[ST.method.isin(MAIN)]
with open(TAB / "stress.tex", "w") as f:
    f.write("\\begin{table*}[t]\n\\centering\n\\caption{Stress test on the held-out phishing pool: detection rate (recall at 0.5) before the stream ($t_0$) and at the D$\\to$E boundary (pre-E), mean over 10 realisations.}\n\\label{tab:stress}\n\\scriptsize\n\\setlength{\\tabcolsep}{2.5pt}\n\\resizebox{\\textwidth}{!}{\\begin{tabular}{@{}l" + "c" * (2 * len(VAR)) + "@{}}\n\\toprule\n & \\multicolumn{%d}{c}{$t_0$ (warm-up model)} & \\multicolumn{%d}{c}{pre-E (after segments A--D)} \\\\\n" % (len(VAR), len(VAR)))
    f.write("\\cmidrule(lr){2-%d}\\cmidrule(lr){%d-%d}\nMethod & " % (len(VAR) + 1, len(VAR) + 2, 2 * len(VAR) + 1) + " & ".join(VN + VN) + " \\\\\n\\midrule\n")
    for m in MAIN:
        cells = []
        for w in ("t0", "preE"):
            g = STm[(STm.method == m) & (STm.when == w)].groupby("variant").detect.mean()
            for v in VAR:
                val = g.get(v, np.nan)
                N[f"stress|{m}|{w}|{v}"] = float(val)
                cells.append("--" if np.isnan(val) else f"{val:.3f}")
        f.write(f"{m} & " + " & ".join(cells) + " \\\\\n")
    f.write("\\bottomrule\n\\end{tabular}}\n\\end{table*}\n")

# ---------------- figures: batch MCC trajectory + SLIDE alarms
show = ["Static-FT", "Static-MLP", "Periodic-Retrain", "ADWIN-Retrain", "KSWIN-Score", "SLIDE"]
fig, ax = plt.subplots(figsize=(7.2, 2.7))
segs = B[(B.method == "SLIDE") & (B.seed == 0)].sort_values("batch").segment.values
for m in show:
    g = B[B.method == m].groupby("batch").mcc.mean()
    ax.plot(g.index, g.values, lw=1.6 if m == "SLIDE" else 0.9, label=m, color="black" if m == "SLIDE" else None)
for i in range(1, len(segs)):
    if segs[i] != segs[i - 1]:
        ax.axvline(i - 0.5, color="grey", ls=":", lw=0.8)
        ax.text(i + 0.5, 1.0, segs[i], fontsize=7, color="grey")
ax.text(0.5, 1.0, segs[0], fontsize=7, color="grey")
ev_s = EV[EV.method == "SLIDE"]
alarms = np.concatenate([np.array([d[0] for d in dl if d[4]]) for dl in ev_s.dlog if len(dl)] or [np.array([])])
if len(alarms):
    h, e = np.histogram(alarms, bins=np.arange(0, len(segs) + 1))
    ax2 = ax.twinx(); ax2.bar(e[:-1], h / len(ev_s), width=1.0, color="tab:red", alpha=0.25); ax2.set_ylim(0, 4); ax2.set_ylabel("SLIDE alarms / stream", fontsize=7); ax2.tick_params(labelsize=7)
ax.set_xlabel("batch (250 e-mails)", fontsize=8); ax.set_ylabel("batch MCC (mean of 10)", fontsize=8)
ax.set_ylim(-0.1, 1.08); ax.tick_params(labelsize=7); ax.legend(fontsize=6, ncol=6, loc="lower center", bbox_to_anchor=(0.5, 1.0), frameon=False)
fig.tight_layout(); fig.savefig(FIG / "trajectory.pdf"); fig.savefig(FIG / "trajectory.png", dpi=200)

fig, ax = plt.subplots(figsize=(3.5, 2.4))
dl = ev_s[ev_s.seed == 0].dlog.values[0]
if len(dl):
    a = np.array(dl, dtype=float)
    # break the lines where no test ran (reference rebuilt after an update)
    gap = np.where(np.diff(a[:, 0]) > 1)[0] + 1
    g = np.insert(a, gap, np.nan, axis=0)
    ax.plot(g[:, 0], g[:, 1], ".-", ms=3, lw=0.8, label="$\\widehat{SW}_1(R,W)$")
    ax.step(g[:, 0], g[:, 2], where="post", lw=0.8, color="tab:orange", label="$\\tau_\\alpha$")
    al = a[a[:, 4] > 0]
    ax.plot(al[:, 0], al[:, 1], "v", color="tab:red", ms=5, label="alarm")
for i in range(1, len(segs)):
    if segs[i] != segs[i - 1]:
        ax.axvline(i - 0.5, color="grey", ls=":", lw=0.8)
ax.set_xlabel("batch", fontsize=8); ax.set_ylabel("sliced $W_1$", fontsize=8); ax.tick_params(labelsize=7); ax.legend(fontsize=6)
fig.tight_layout(); fig.savefig(FIG / "swtrace.pdf"); fig.savefig(FIG / "swtrace.png", dpi=200)

# alarm timing: delay from each boundary to first alarm, false alarms inside A
bounds = [i for i in range(1, len(segs)) if segs[i] != segs[i - 1]]
delays = {b: [] for b in bounds}
falseA = []
for dlx in ev_s.dlog:
    al = [d[0] for d in dlx if d[4]]
    falseA.append(sum(1 for t in al if t < bounds[0]))
    for j, b in enumerate(bounds):
        nxt = bounds[j + 1] if j + 1 < len(bounds) else len(segs)
        hit = [t for t in al if b <= t < nxt]
        delays[b].append(hit[0] - b if hit else np.nan)
for b in bounds:
    N[f"delay|{segs[b - 1]}->{segs[b]}"] = float(np.nanmean(delays[b])) if np.any(~np.isnan(delays[b])) else None
    N[f"detected|{segs[b - 1]}->{segs[b]}"] = float(np.mean(~np.isnan(delays[b])))
N["alarms_in_A_mean"] = float(np.mean(falseA))
N["alarms_per_stream"] = float(np.mean([sum(1 for d in dlx if d[4]) for dlx in ev_s.dlog]))
N["t_detect_ms_per_test"] = float(np.mean([e["t_detect"] / max(len(e["dlog"]), 1) * 1000 for _, e in ev_s.iterrows()]))
json.dump(N, open(RESULTS / f"numbers{TAG}.json", "w"), indent=1, default=float)
print("tables written to", TAB)
