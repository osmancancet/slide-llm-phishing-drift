"""Revision tables, figures and numbers from results/revision (written by run_revision.py --collect)."""
import json
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from slide.config import RESULTS, BATCH
from slide.stats import holm, wilcoxon_rb, friedman_nemenyi, mean_ci

R = RESULTS / "revision"
TAB = RESULTS.parent / "manuscript" / "tables"
FIG = RESULTS.parent / "manuscript" / "figures"
S = pd.read_csv(R / "summary.csv", keep_default_na=False, na_values=[""])   # the group name "null" must not parse as NaN
B = pd.read_csv(R / "batch.csv.gz", keep_default_na=False, na_values=[""])
ST = pd.read_csv(R / "stress.csv")
EV = pd.DataFrame(json.load(open(R / "events.json")))
EV["group"] = EV.key.str[0]
EV["name"] = EV.key.str[1]
# the ablation reuses the main-group run of the KBS configuration
KBS = "full objective + psi (KBS version)"
_s = S[(S.group == "main") & (S.name == "SLIDE (KBS version)")].assign(group="abl", name=KBS)
_e = EV[(EV.group == "main") & (EV.name == "SLIDE (KBS version)")].assign(group="abl", name=KBS)
S = pd.concat([_s, S], ignore_index=True)
EV = pd.concat([_e, EV], ignore_index=True)
N = {}
SEGS = list("ABCDEFG")


def ci(x, d=3):
    m, lo, hi = mean_ci(x)
    return f"{m:.{d}f}\\,$\\pm$\\,{(hi - lo) / 2:.{d}f}"


def sm(g, n, metric="mcc", seg="ALL"):
    d = S[(S.group == g) & (S.name == n) & (S.segment == seg)].sort_values("seed")
    return d[metric].values.astype(float)


def ev(g, n):
    return EV[(EV.group == g) & (EV.name == n)].sort_values("seed")


def tex(s):
    return (s.replace("_", "\\_").replace("n_l=", "$n_\\ell$=").replace("K=", "$K$=").replace("psi", "$\\psi$")
            .replace("alpha", "$\\alpha$").replace("D=", "$D$="))


def alarm_batches(e):
    """alarm batches of one run: drift-test alarms for SLIDE-type runs, update events otherwise."""
    if e["info"]:
        return [d["t"] for d in e["info"] if d["alarm"]]
    return list(e["events"])


def detection(e):
    """per boundary: delay (batches) to the first alarm in the new segment, and alarms not attributable."""
    bounds = [b // BATCH for b in e["seg_bounds"]]
    al = sorted(alarm_batches(e))
    edges = bounds + [e["n_batches"]]
    delays, extra = [], 0
    pre = [a for a in al if a < (bounds[0] if bounds else e["n_batches"])]
    extra += len(pre)
    for b, nx in zip(bounds, edges[1:]):
        inside = [a for a in al if b <= a < nx]
        delays.append(inside[0] - b if inside else np.nan)
        extra += max(len(inside) - 1, 0)
    return delays, extra, len(al)


def write(name, s):
    s = s.replace("(KBS version)", "(earlier design)")
    (TAB / name).write_text(s)


def table(caption, label, cols, rows, star=False, align=None):
    env = "table*" if star else "table"
    al = align or ("l" + "c" * (len(cols) - 1))
    s = (f"\\begin{{{env}}}[t]\n\\centering\n\\caption{{{caption}}}\n\\label{{{label}}}\n\\scriptsize\n"
         f"\\setlength{{\\tabcolsep}}{{3pt}}\n" + ("\\resizebox{\\textwidth}{!}{" if star else "")
         + f"\\begin{{tabular}}{{@{{}}{al}@{{}}}}\n\\toprule\n" + " & ".join(cols) + " \\\\\n\\midrule\n")
    for r in rows:
        s += ("\\midrule\n" if r == "MID" else " & ".join(str(c) for c in r) + " \\\\\n")
    s += "\\bottomrule\n\\end{tabular}" + ("}" if star else "") + f"\n\\end{{{env}}}\n"
    return s


# ------------------------------------------------------------------ main comparison (A-G, abrupt)
MAIN = list(dict.fromkeys(S[S.group == "main"].name))
ORDER = ["TFIDF-LR", "Online-TFIDF-SGD", "Static-FT", "Static-MLP", "Online-MLP", "ADWIN-Retrain", "KSWIN-Score",
         "Budget-Uniform-MLP", "Budget-Uncertainty-MLP", "Budget-Uniform-TFIDF", "Budget-Uncertainty-TFIDF",
         "Budget-Random", "Budget-Oracle", "Budget-Random-TFIDF", "Budget-Oracle-TFIDF",
         "SLIDE (KBS version)", "SLIDE", "SLIDE+replay", "SLIDE-TFIDF"]
MAIN = [m for m in ORDER if m in MAIN]
rows = []
for m in MAIN:
    e = ev("main", m)
    lab, upd = e.stream_labels.mean(), e.events.apply(len).mean()
    online = m in ("Online-TFIDF-SGD", "Online-MLP")
    r = [m.replace("SLIDE", "\\textbf{SLIDE}") if m in ("SLIDE", "SLIDE+replay", "SLIDE-TFIDF") else m]
    for k in ("precision", "recall", "mcc", "pr_auc", "fpr"):
        x = sm("main", m, k)
        r.append(ci(x))
        N[f"main|{m}|{k}"] = float(x.mean())
    r += [f"{lab:.0f}", "every batch" if online else f"{upd:.1f}"]
    N[f"main|{m}|labels"] = float(lab)
    N[f"main|{m}|updates"] = float(upd)
    N[f"main|{m}|ms_score"] = float(e.ms_score.mean())
    N[f"main|{m}|t_update"] = float(e.t_update.mean())
    rows.append(r)
    if m in ("Budget-Uniform-MLP", "Budget-Oracle-TFIDF", "Static-FT"):
        rows.append("MID")
write("rev_main.tex", table(
    "Prequential results on the seven-segment stream A--G (36{,}000 messages, mean\\,$\\pm$\\,95\\% CI half-width over 10 "
    "realisations). Labels: stream labels consumed after warm-up; Upd.: model updates per stream. The Budget-* rows spend "
    "the same number of labels as SLIDE, either spread over the stream (uniform, uncertainty) or as four windows at "
    "random batches or right after each true change (oracle).", "tab:rev-main",
    ["Method", "Precision", "Recall", "MCC", "PR-AUC", "FPR", "Labels", "Upd."], rows, star=True))

# per segment
rows = []
M = {m: [sm("main", m, "mcc", g).mean() for g in SEGS] for m in MAIN}
best = np.nanmax(np.array(list(M.values())), 0)
for m in MAIN:
    rows.append([m] + [("\\textbf{%.3f}" if abs(v - b) < 5e-4 else "%.3f") % v for v, b in zip(M[m], best)])
    for g, v in zip(SEGS, M[m]):
        N[f"main|{m}|mcc|{g}"] = float(v)
write("rev_segments.tex", table("MCC per segment of the A--G stream (mean over 10 realisations; best per column in bold).",
                                "tab:rev-segments", ["Method"] + SEGS, rows, star=True))

# slices
SL = [("D:llm", "LLM, D"), ("D:human", "human, D"), ("E:para", "para., E"), ("F:llm", "LLM, F"),
      ("F:human", "human, F"), ("G:translated", "transl., G")]
sl = S[(S.group == "main") & S.segment.str.contains(":")]
rows = []
for m in MAIN:
    g = sl[sl.name == m].groupby("segment")[["recall", "fpr"]].mean()
    cells = []
    for kind in ("recall", "fpr"):
        for s_, _ in SL:
            v = g.loc[s_, kind] if s_ in g.index and not (kind == "recall" and "human" in s_) else np.nan
            N[f"slice|{m}|{s_.replace(':', '_')}|{kind}"] = None if np.isnan(v) else float(v)
            cells.append("--" if np.isnan(v) else f"{v:.3f}")
    rows.append([m] + cells)
hdr = ["Method"] + [f"Rec. {b}" for _, b in SL] + [f"FPR {b}" for _, b in SL]
write("rev_slices.tex", table("Recall on malicious and false-positive rate on legitimate mail by authorship (mean over 10 "
                              "realisations). LLM: LLM-written; human: human-written; para.: T1-paraphrased; transl.: "
                              "machine-translated Spanish mail. D contains no human-written malicious mail, so its "
                              "human recall is undefined.", "tab:rev-slices", hdr, rows, star=True))

# statistics against SLIDE
ref = "SLIDE"
comp = [m for m in MAIN if m != ref]
Mseed = np.column_stack([sm("main", m) for m in MAIN])
chi, pf, ranks, cd = friedman_nemenyi(Mseed)
N.update(friedman_chi2=chi, friedman_p=pf, nemenyi_cd=cd, n_methods=len(MAIN), n_comp=len(comp),
         p_min_holm=len(comp) * 2 / 1024)
res = []
for m in comp:
    a, b = sm("main", ref), sm("main", m)
    p, rb, _ = wilcoxon_rb(a, b)
    res.append(dict(m=m, d=float(np.mean(a - b)), p=p, rb=rb, wins=int(np.sum(a > b))))
res = pd.DataFrame(res)
res["ph"] = holm(res.p.values)
rows = []
for _, r in res.iterrows():
    rows.append([r.m, f"{r.d:+.3f}", f"{r.wins}/10", f"{r.ph:.3g}", f"{r.rb:+.2f}", f"{ranks[MAIN.index(r.m)]:.1f}"])
    for k in ("d", "ph", "rb", "wins"):
        N[f"vs|{r.m}|{k}"] = float(r[k])
for m, rk in zip(MAIN, ranks):
    N[f"rank|{m}"] = float(rk)
rows.append("MID")
rows.append(["SLIDE (mean rank)", "", "", "", "", f"{ranks[MAIN.index(ref)]:.1f}"])
write("rev_stats.tex", table(
    "Revised SLIDE against each method on the A--G stream: mean paired MCC difference, wins out of 10 realisations, "
    "two-sided Wilcoxon signed-rank test with Holm correction over %d comparisons, rank-biserial effect size $r_{rb}$, "
    "and Friedman mean rank (Friedman $\\chi^2_F=%.1f$, $p=%.1e$; Nemenyi CD$_{0.05}=%.2f$)." % (len(comp), chi, pf, cd),
    "tab:rev-stats", ["Competitor", "$\\Delta$MCC", "wins", "$p_{\\mathrm{Holm}}$", "$r_{rb}$", "rank"], rows))

# ------------------------------------------------------------------ ablation
AB = list(dict.fromkeys(S[S.group == "abl"].name))
base = sm("abl", "revised + alpha-spending")
ps = holm([wilcoxon_rb(base, sm("abl", a))[0] for a in AB if a != "revised + alpha-spending"])
it = iter(ps)
rows = []
for a in AB:
    x = sm("abl", a)
    e = ev("abl", a)
    p = np.nan if a == "revised + alpha-spending" else next(it)
    N[f"abl|{a}|mcc"] = float(x.mean()); N[f"abl|{a}|p"] = None if np.isnan(p) else float(p)
    N[f"abl|{a}|labels"] = float(e.stream_labels.mean()); N[f"abl|{a}|t_update"] = float(e.t_update.mean())
    N[f"abl|{a}|d"] = float((x - base).mean())
    rows.append([tex(a), ci(x), f"{sm('abl', a, 'pr_auc').mean():.3f}", f"{sm('abl', a, 'mcc', 'F').mean():.3f}",
                 f"{sm('abl', a, 'mcc', 'G').mean():.3f}", f"{e.stream_labels.mean():.0f}", f"{e.t_update.mean():.1f}",
                 "--" if np.isnan(p) else f"{(x - base).mean():+.3f}", "--" if np.isnan(p) else f"{p:.3g}"])
write("rev_ablation.tex", table(
    "Component ablation on the A--G stream (10 realisations). $\\Delta$ and $p_{\\mathrm{Holm}}$ compare each variant with "
    "the revised configuration (warm-start update, no $\\psi$, $\\alpha$-spending) by a two-sided Wilcoxon test on "
    "realisation-level MCC, Holm-corrected. Upd.\\,s: total update time per stream in seconds.", "tab:rev-ablation",
    ["Variant", "MCC", "PR-AUC", "MCC (F)", "MCC (G)", "Labels", "Upd.\\,s", "$\\Delta$", "$p_{\\mathrm{Holm}}$"], rows,
    star=True))


# effect of alpha-spending on detection delay (same update, per-test alpha vs harmonic schedule)
_d = {}
for nm, key in (("revised + alpha-spending", "harm"), ("warm (revised update)", "none")):
    _d[key] = np.array([detection(r)[0] for _, r in ev("abl", nm).iterrows()], float)
N["spend|delay|harm"] = float(np.nanmean(_d["harm"])); N["spend|delay|none"] = float(np.nanmean(_d["none"]))
_diff = _d["harm"] - _d["none"]
N["spend|n_later"] = float(np.sum(_diff > 0)); N["spend|n_earlier"] = float(np.sum(_diff < 0))
N["spend|n_bound"] = float(_diff.size); N["spend|max_later"] = float(np.nanmax(_diff))
N["spend|miss"] = float(np.isnan(_d["harm"]).sum())

# ------------------------------------------------------------------ K, budget, delay
def det_stats(g, n):
    e = ev(g, n)
    D, X, A = [], [], []
    fs, cv = [], []
    for _, r in e.iterrows():
        d, x, a = detection(r)
        D += d; X.append(x); A.append(a)
        inf = [i for i in r["info"]]
        if inf:
            fs.append(np.mean([i.get("frac_sig", np.nan) for i in inf if i["alarm"]] or [np.nan]))
            cv.append(np.mean([i.get("w1_cv", np.nan) for i in inf]))
    D = np.array(D, float)
    return dict(miss=float(np.mean(np.isnan(D))), delay=float(np.nanmean(D)) if np.any(~np.isnan(D)) else np.nan,
                extra=float(np.mean(X)), alarms=float(np.mean(A)), frac_sig=float(np.nanmean(fs)) if fs else np.nan,
                w1_cv=float(np.nanmean(cv)) if cv else np.nan, labels=float(e.stream_labels.mean()),
                t_detect=float(e.t_detect.mean()))


rows = []
for K in (2, 4, 8, 16, 32, 64, 128, 256, 512):
    n = f"K={K}"
    if not len(sm("K", n)):
        continue
    d = det_stats("K", n)
    x = sm("K", n)
    for k, v in d.items():
        N[f"K|{K}|{k}"] = v
    N[f"K|{K}|mcc"] = float(x.mean())
    rows.append([K, ci(x), f"{d['labels']:.0f}", f"{d['alarms']:.1f}", f"{d['delay']:.2f}", f"{d['miss']:.2f}",
                 f"{d['extra']:.1f}", f"{d['frac_sig']:.2f}", f"{d['w1_cv']:.2f}", f"{d['t_detect']:.1f}"])
write("rev_K.tex", table(
    "Number of random projections $K$ (revised SLIDE, 5 realisations, six boundaries per stream). Delay: batches from a "
    "boundary to the first alarm in the new segment; miss: share of boundaries without an alarm before the next boundary; "
    "extra: further alarms per stream; sig.: share of projections individually significant at the Holm level in alarming "
    "tests; CV$_{W_1}$: coefficient of variation of the per-projection $W_1$ distances; $t$: total test time (s).",
    "tab:rev-K", ["$K$", "MCC", "Labels", "Alarms", "Delay", "Miss", "Extra", "Sig.", "CV$_{W_1}$", "$t$"], rows))

rows = []
for nl in (50, 100, 250, 500, 1000):
    cells = [nl]
    for m, suf in (("SLIDE", ""), ("SLIDE-TFIDF", " TFIDF")):
        n = f"n_l={nl}{suf}"
        x = sm("budget", n)
        if not len(x):
            cells += ["--"] * 3
            continue
        e = ev("budget", n)
        N[f"budget|{m}|{nl}|mcc"] = float(x.mean()); N[f"budget|{m}|{nl}|labels"] = float(e.stream_labels.mean())
        N[f"budget|{m}|{nl}|mccG"] = float(sm("budget", n, "mcc", "G").mean())
        cells += [ci(x), f"{sm('budget', n, 'mcc', 'G').mean():.3f}", f"{e.stream_labels.mean():.0f}"]
    rows.append(cells)
write("rev_budget.tex", table(
    "Label window $n_\\ell$ bought per alarm (5 realisations). SLIDE updates the transformer head; SLIDE-TFIDF uses the "
    "same alarms to update an online TF-IDF learner. For $n_\\ell<250$ the window is a random subset of the alarm batch; "
    "MCC (G): MCC on segment G.", "tab:rev-budget",
    ["$n_\\ell$", "SLIDE MCC", "MCC (G)", "Labels", "SLIDE-TFIDF MCC", "MCC (G)", "Labels"], rows))

DL = ["1", "2", "4", "8", "16", "rand1-8", "rand1-16"]
DM = ["SLIDE", "SLIDE-TFIDF", "Online-TFIDF-SGD", "Budget-Uniform-TFIDF"]
rows = []
for m in DM:
    cells = [m]
    for d in DL:
        x = sm("main", m)[:5] if d == "2" else sm("delay", f"{m} D={d}")
        N[f"delay|{m}|{d}"] = float(x.mean()) if len(x) else None
        cells.append(f"{x.mean():.3f}" if len(x) else "--")
    rows.append(cells)
write("rev_delay.tex", table(
    "MCC under label delay (5 realisations, $r=0,\\dots,4$). $D$ is the delay in batches of 250 messages; rand$a$--$b$ draws each "
    "batch's delay uniformly from $\\{a,\\dots,b\\}$. $D=2$ is the default of Table~\\ref{tab:rev-main}.",
    "tab:rev-delay", ["Method"] + [("$D$=" + d) if not d.startswith("rand") else d.replace("rand", "rand ") for d in DL],
    rows))

# designs
DS = ["abrupt", "gradual", "varlen", "permuted"]
DSM = ["SLIDE", "SLIDE-TFIDF", "ADWIN-Retrain", "Online-TFIDF-SGD", "Budget-Uniform-TFIDF",
       "Budget-Uniform-MLP", "Budget-Random", "Budget-Oracle", "Static-MLP"]
rows = []
for m in DSM:
    cells = [m]
    for d in DS:
        if d == "abrupt":
            x, lab = sm("main", m)[:5], ev("main", m).stream_labels.values[:5].mean()
        else:
            x, lab = sm("design", f"{d}|{m}"), ev("design", f"{d}|{m}").stream_labels.mean()
        N[f"design|{m}|{d}"] = float(x.mean()) if len(x) else None
        N[f"design|{m}|{d}|labels"] = float(lab) if len(x) else None
        cells += [f"{x.mean():.3f}" if len(x) else "--", f"{lab:.0f}" if len(x) else "--"]
    rows.append(cells)
write("rev_designs.tex", table(
    "MCC and labels under four stream designs (5 realisations each, $r=0,\\dots,4$): abrupt (default), gradual (2{,}000-message mixing "
    "zones), variable segment lengths, and permuted segment order.", "tab:rev-designs",
    ["Method"] + [c for d in DS for c in (f"{d} MCC", "Lab.")], rows, star=True))

# ------------------------------------------------------------------ null stream
rows = []
for n in list(dict.fromkeys(S[S.group == "null"].name)):
    e = ev("null", n)
    A = np.array([len(alarm_batches(r)) for _, r in e.iterrows()])
    tests = np.array([len(r["info"]) for _, r in e.iterrows()])
    N[f"null|{n}|alarms"] = float(A.mean()); N[f"null|{n}|p_any"] = float((A > 0).mean())
    N[f"null|{n}|tests"] = float(tests.mean()); N[f"null|{n}|labels"] = float(e.stream_labels.mean())
    rows.append([tex({"SW gate only": "alpha-spending, SW only"}.get(n, n)), f"{tests.mean():.0f}" if tests.any() else "--", f"{A.mean():.2f}", f"{(A > 0).mean():.2f}",
                 f"{e.stream_labels.mean():.0f}"])
write("rev_null.tex", table(
    "Stationary null stream (20{,}000 messages drawn from A$\\cup$B$\\cup$C without segments, 20 realisations). Tests: drift "
    "tests run per stream; alarms: false alarms per stream; $P(\\geq1)$: share of streams with at least one false alarm.",
    "tab:rev-null", ["Configuration", "Tests", "Alarms", "$P(\\geq 1)$", "Labels"], rows))

# ------------------------------------------------------------------ real drift probe
rows = []
for n in list(dict.fromkeys(S[S.group == "real"].name)):
    e = ev("real", n)
    x0, x1 = sm("real", n, "mcc", "R0"), sm("real", n, "mcc", "R1")
    r1 = sm("real", n, "recall", "R1")
    N[f"real|{n}|R0"] = float(x0.mean()); N[f"real|{n}|R1"] = float(x1.mean()); N[f"real|{n}|recR1"] = float(r1.mean())
    N[f"real|{n}|labels"] = float(e.stream_labels.mean())
    N[f"real|{n}|audit_alarms"] = float(e.audit_alarms.mean())
    N[f"real|{n}|updates"] = float(e.events.apply(len).mean())
    first = np.array([min(x) if len(x) else np.nan for x in e.events], float)
    N[f"real|{n}|p_upd"] = float(np.isfinite(first).mean())
    N[f"real|{n}|lag"] = float(np.nanmean(first) - 16) if np.isfinite(first).any() else None   # R1 starts at batch 16
    rows.append([n.replace("n_a=", "$n_a$="), f"{x0.mean():.3f}", ci(x1), f"{r1.mean():.3f}", f"{e.stream_labels.mean():.0f}",
                 f"{e.events.apply(len).mean():.1f}"])
write("rev_real.tex", table(
    "Real-drift probe (10 realisations): in R1 a fixed region of legitimate mail (about one fifth of it, one $k$-means "
    "cell of $\\phi$) becomes malicious while $P(x)$ is unchanged. SLIDE+Audit adds $n_a$ random audit labels per batch "
    "($n_a=5$ unless stated) and an ADWIN test on their error stream; an audit alarm buys the same 500-label window.", "tab:rev-real",
    ["Method", "MCC R0", "MCC R1", "Recall R1", "Labels", "Upd."], rows))

# ------------------------------------------------------------------ stress (prefix attacks, chunked scoring)
VAR = ["orig", "T1_para", "T2_char_0.10", "T3_pad_200", "T3_pad", "T3_pad_800", "T4_suffix", "T5_sandwich"]
VN = ["orig.", "T1", "T2", "T3-200", "T3-400", "T3-800", "T4", "T5"]
STS = ST[ST.sgrp == "S1"]
SHOW = [("main", m) for m in ("TFIDF-LR", "Online-TFIDF-SGD", "Static-MLP", "ADWIN-Retrain", "SLIDE", "SLIDE+replay", "SLIDE-TFIDF",
                              "SLIDE (KBS version)")] + [("chunk", c) for c in ("Static-MLP chunked",
                                                                                  "ADWIN-Retrain chunked", "SLIDE chunked")]
rows = []
for g, m in SHOW:
    d = STS[(STS.group == g) & (STS.name == m) & (STS.when == "end")].groupby("variant").detect.mean()
    cells = []
    for v in VAR:
        val = d.get(v, np.nan)
        N[f"stress|{m}|{v}"] = None if np.isnan(val) else float(val)
        cells.append("--" if np.isnan(val) else f"{val:.3f}")
    rows.append([m] + cells)
    if m == "SLIDE (KBS version)":
        rows.append("MID")
write("rev_stress.tex", table(
    "Detection rate on the 600 held-out malicious messages of Table~\\ref{tab:rev-retention} at the end of the stream (mean over 10 "
    "realisations). T1: paraphrase; T2: a homoglyph, transposition or zero-width character in 10\\% of the words of four or more letters; T3-$n$: "
    "$n$-character benign prefix; T4: 400-character benign suffix; T5: 400-character prefix and suffix. Chunked rows "
    "score up to five 126-token windows and take the maximum.", "tab:rev-stress", ["Method"] + VN, rows, star=True))

S2 = ST[ST.sgrp.str.startswith("S2") & (ST.when == "end")]
rows = []
for g, m in SHOW:
    d = S2[(S2.group == g) & (S2.name == m)].groupby(["sgrp", "variant"]).detect.mean()
    cells = []
    for sg in sorted(S2.sgrp.unique()):
        for v in ("orig", "T2_char_0.10", "T3_pad"):
            val = d.get((sg, v), np.nan)
            N[f"stress2|{m}|{sg.replace(':', '_')}|{v}"] = None if np.isnan(val) else float(val)
            cells.append("--" if np.isnan(val) else f"{val:.3f}")
    rows.append([m] + cells)
SG = sorted(S2.sgrp.unique())
write("rev_stress2.tex", table(
    "Detection rate on a second held-out pool never streamed: 200 PhishFuzzer LLM phishing and 200 machine-translated "
    "Spanish phishing messages (end of stream, mean over 10 realisations).", "tab:rev-stress2",
    ["Method"] + [f"{sg.split(':')[1][:10]} {v}" for sg in SG for v in ("orig.", "T2", "T3")], rows, star=True))

# ------------------------------------------------------------------ retention: held-out pool before and after the stream
rows = []
for m in MAIN:
    d = ST[(ST.group == "main") & (ST.name == m) & (ST.variant == "orig")]
    cells = [m]
    for sg, w in (("S1", "t0"), ("S1", "preE"), ("S1", "end")):
        v = d[(d.sgrp == sg) & (d.when == w)].detect.mean()
        N[f"ret|{m}|{w}"] = float(v)
        cells.append(f"{v:.3f}")
    v0 = N[f"ret|{m}|preE"]
    N[f"ret|{m}|drop"] = float(v0 - N[f"ret|{m}|end"])
    cells.append(f"{N[f'ret|{m}|end'] - v0:+.3f}")
    rows.append(cells)
    if m in ("Budget-Uniform-MLP", "Budget-Oracle-TFIDF", "Static-FT"):
        rows.append("MID")
write("rev_retention.tex", table(
    "Retention: detection rate on 600 held-out malicious messages from the sources of segments B--D (400 GPTGen and "
    "SpearGen, 200 TREC and CEAS-08), never streamed, scored by "
    "each detector as it stands before the stream ($t_0$), just before segment E and at the end of the stream "
    "(mean over 10 realisations). $\\Delta$: end minus pre-E.", "tab:rev-retention",
    ["Method", "$t_0$", "pre-E", "end", "$\\Delta$"], rows))

# extra constants from the original audit and latency runs
_a = json.load(open(RESULTS / "leakage_audit.json"))
N["auth_probe"] = float(_a["authorship_probe_f1macro"])
_l = json.load(open(RESULTS / "latency.json"))
N["lat_enc_ms"] = float(_l["encoder_ms_4t"]); N["lat_lm_ms"] = float(_l["surprisal_lm_ms_4t"])
N["lat_test_ms_batch"] = float(_l["drift_test_ms_per_batch"]); N["lat_head_ms"] = float(_l["head_ms"])
N["lat_ref_s"] = float(_l["reference_build_s"]); N["lat_tfidf_ms"] = float(_l["tfidf_lr_ms"])
for n in list(dict.fromkeys(S[S.group == "chunk"].name)):          # chunked scoring on the clean A-G stream
    for met in ("mcc", "fpr", "recall"):
        N[f"chunk|{n}|{met}"] = float(sm("chunk", n, met).mean())
    N[f"chunk|{n}|wall"] = float(ev("chunk", n).wall.mean())
for m in ("SLIDE", "Static-MLP", "ADWIN-Retrain", "Online-TFIDF-SGD", "SLIDE-TFIDF", "SLIDE+replay"):
    e = ev("main", m)
    N[f"main|{m}|wall"] = float(e.wall.mean()); N[f"main|{m}|t_detect"] = float(e.t_detect.mean())

# ------------------------------------------------------------------ figure: batch MCC
fig, ax = plt.subplots(figsize=(7.2, 2.7))
Bm = B[B.group == "main"]
segs = Bm[(Bm.name == "SLIDE") & (Bm.seed == 0)].sort_values("batch").segment.values
for m in ("Static-MLP", "Online-TFIDF-SGD", "ADWIN-Retrain", "Budget-Uniform-MLP", "SLIDE-TFIDF", "SLIDE"):
    g = Bm[Bm.name == m].groupby("batch").mcc.mean()
    ax.plot(g.index, g.values, lw=1.6 if m == "SLIDE" else 0.9, label=m, color="black" if m == "SLIDE" else None)
for i in range(1, len(segs)):
    if segs[i] != segs[i - 1]:
        ax.axvline(i - 0.5, color="grey", ls=":", lw=0.8)
        ax.text(i + 0.5, 1.01, segs[i], fontsize=7, color="grey")
ax.text(0.5, 1.01, segs[0], fontsize=7, color="grey")
ax.set_xlabel("batch (250 messages)")
ax.set_ylabel("MCC (mean of 10)")
ax.set_ylim(-0.05, 1.08)
ax.legend(fontsize=6.5, ncol=3, loc="lower left")
fig.tight_layout()
fig.savefig(FIG / "rev_trajectory.pdf")
fig.savefig(FIG / "rev_trajectory.png", dpi=200)

json.dump(N, open(R / "numbers.json", "w"), indent=1, default=float)
print(len(N), "numbers")

# ------------------------------------------------------------------ pools added in the revision
from slide.config import CACHE
tx = pd.read_parquet(CACHE / "texts_v2.parquet")
tx = tx[tx.variant == "orig"]
rows = []
for pool, nm in (("F", "segment F (LLM)"), ("F_hum", "segment F (human)"), ("G", "segment G"),
                 ("STRESS2", "stress pool 2"), ("NULLH", "real-drift probe")):
    d = tx[tx.pool == pool]
    src = d.groupby(["source", "label"]).size().unstack(fill_value=0)
    desc = "; ".join(f"{s} {int(r.get(0, 0))}/{int(r.get(1, 0))}" for s, r in src.iterrows())
    nl, nm1 = int((d.label == 0).sum()), int((d.label == 1).sum())
    N[f"pool|{pool}|legit"], N[f"pool|{pool}|mal"] = nl, nm1
    rows.append([nm, nl, nm1, desc.replace("_", "\\_")])
write("rev_pools.tex", table("Pools added in this revision (legitimate/malicious counts), disjoint from each other and from "
                             "the pools of Table~\\ref{tab:pools}. One PhishFuzzer variant is kept per seed message.",
                             "tab:rev-pools", ["Pool", "Legit.", "Malic.", "Sources (legit./malic.)"], rows,
                             align="lrrp{8.5cm}"))
json.dump(N, open(R / "numbers.json", "w"), indent=1, default=float)
