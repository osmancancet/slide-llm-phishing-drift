"""Revision experiments on the seven-segment stream (A-G) and the probe streams.

usage: python run_revision.py --group main --shard i/n      (run)      |  --collect   (assemble)
groups: main, abl, K, budget, delay, design, null, real, chunk
"""
import argparse
import json
import os
import pickle
import time
import numpy as np
import pandas as pd
import torch
from slide.config import CACHE, RESULTS, BATCH
from slide.data_ext import compose_stream_ext, compose_null_stream, compose_realdrift_stream
from slide import stream as S0
from slide import stream_ext as S
from slide.metrics import metrics

G = {}
REV = dict(update="warm", spend="harmonic")      # revised SLIDE: no psi, plain warm-start update, alpha-spending
NB = 21                                         # equal-budget streaming baselines: 21 labels per batch (= SLIDE mean 3000 / 144 batches)


def load(fake=False):
    from slide.features import embed, surprisal
    from slide.features_ext import embed_windows, h
    texts = pd.read_parquet(CACHE / "texts_v2.parquet")
    if fake:     # smoke test only: random features with a weak label signal
        rng = np.random.default_rng(0)
        yv = texts.label.values
        phi = (rng.standard_normal((len(texts), 768)) + 0.3 * yv[:, None]).astype(np.float32)
        psi = rng.standard_normal((len(texts), 4)).astype(np.float32)
        pft = 1 / (1 + np.exp(-2 * phi[:, 0]))
        ln = rng.integers(0, 3, len(texts))
        G["win_end"] = np.cumsum(ln); G["win_start"] = G["win_end"] - ln
        G["WX"] = rng.standard_normal((ln.sum(), 768)).astype(np.float32)
    else:
        phi, pft = embed(texts.text.tolist())
        psi = surprisal(texts.text.tolist())
    warm = (texts.pool == "WARM").values
    mu, sd = phi[warm].mean(0), phi[warm].std(0) + 1e-6
    mu2, sd2 = psi[warm].mean(0), psi[warm].std(0) + 1e-6
    Zphi, Zpsi = (phi - mu) / sd, (psi - mu2) / sd2
    G.update(texts=texts, X=np.hstack([Zphi, Zpsi]).astype(np.float32), Xnp=Zphi.astype(np.float32), pft=pft)
    if not fake and os.path.exists(CACHE / "phi_win.pkl"):
        W = embed_windows(texts.text.tolist())
        keys = [h(t) for t in texts.text.tolist()]
        lens = np.array([len(W[k]) for k in keys])
        G["win_end"] = np.cumsum(lens)
        G["win_start"] = G["win_end"] - lens
        WX = np.concatenate([W[k] for k in keys if len(W[k])]).astype(np.float32)
        G["WX"] = ((WX - mu) / sd).astype(np.float32)


# ---------------------------------------------------------------- job registry
def methods_main():
    return [("TFIDF-LR", {}), ("Online-TFIDF-SGD", {}), ("Static-FT", {}), ("Static-MLP", {}),
            ("Online-MLP", {}), ("ADWIN-Retrain", {}), ("KSWIN-Score", {}),
            ("Budget-Uniform-MLP", {}), ("Budget-Uncertainty-MLP", {}),
            ("Budget-Uniform-TFIDF", {}), ("Budget-Uncertainty-TFIDF", {}),
            ("Budget-Random", REV), ("Budget-Oracle", REV), ("Budget-Random-TFIDF", {}), ("Budget-Oracle-TFIDF", {}),
            ("SLIDE", REV), ("SLIDE-TFIDF", dict(spend="harmonic")),
            ("SLIDE+replay", dict(REV, update="warm+replay")),
            ("SLIDE (KBS version)", dict(update="full", spend="none", psi=True))]


def registry():
    J = {}
    seeds = range(10)
    s5 = range(5)            # sensitivity grids (K, n_l, delay, design): 5 realisations, as in the original paper
    for m, c in methods_main():
        J[("main", m)] = (m, c, "abrupt", seeds)
    # ablation of the original components on the A-G stream (psi included where stated)
    # the KBS configuration itself is the main-group row "SLIDE (KBS version)"; the ablation table reuses it
    abl = {"full objective, no psi": dict(update="full", spend="none"),
           "warm+replay": dict(update="warm+replay", spend="none"),
           "warm (revised update)": dict(update="warm", spend="none"),
           "scratch": dict(update="scratch", spend="none"),
           "warm + psi": dict(update="warm", spend="none", psi=True),
           "revised + alpha-spending": REV,
           "revised, KS gate only": dict(REV, use_sw=False),
           "revised, SW gate only": dict(REV, use_ks=False),
           "revised, score-only test": dict(REV, use_proj=False),
           "revised, projections-only": dict(REV, use_score=False),
           "revised, no adaptation": dict(REV, adapt=False)}
    for n, c in abl.items():
        J[("abl", n)] = ("SLIDE", c, "abrupt", seeds)
    for K in (2, 4, 8, 16, 32, 64, 128, 256, 512):
        J[("K", f"K={K}")] = ("SLIDE", dict(REV, K=K), "abrupt", s5)
    for n in (50, 100, 250, 500, 1000):
        J[("budget", f"n_l={n}")] = ("SLIDE", dict(REV, label_win=n), "abrupt", s5)
        J[("budget", f"n_l={n} TFIDF")] = ("SLIDE-TFIDF", dict(spend="harmonic", label_win=n), "abrupt", s5)
    for d in ("1", "4", "8", "16", "rand1-8", "rand1-16"):
        for m, c in (("SLIDE", REV), ("SLIDE-TFIDF", dict(spend="harmonic")),
                     ("Online-TFIDF-SGD", {}), ("Budget-Uniform-TFIDF", {})):
            J[("delay", f"{m} D={d}")] = (m, dict(c, delay=d), "abrupt", s5)
    for design in ("gradual", "varlen", "permuted"):
        for m, c in (("SLIDE", REV), ("SLIDE-TFIDF", dict(spend="harmonic")), ("ADWIN-Retrain", {}),
                     ("Online-TFIDF-SGD", {}), ("Budget-Uniform-TFIDF", {}),
                     ("Budget-Uniform-MLP", {}), ("Budget-Random", REV), ("Budget-Oracle", REV), ("Static-MLP", {})):
            J[("design", f"{design}|{m}")] = (m, c, design, s5)
    for m, c in (("per-test alpha", dict(update="warm", spend="none")), ("alpha-spending", REV),
                 ("per-test alpha, KS only", dict(update="warm", spend="none", use_sw=False)),
                 ("alpha-spending, KS only", dict(REV, use_sw=False)), ("SW gate only", dict(REV, use_ks=False)),
                 ("KSWIN-Score", None)):
        J[("null", m)] = ("KSWIN-Score" if c is None else "SLIDE", c or {}, "null", range(20))
    for m, c in (("SLIDE", REV), ("SLIDE+Audit", REV), ("ADWIN-Retrain", {}), ("Online-TFIDF-SGD", {}),
                 ("Static-MLP", {}), ("Budget-Uniform-MLP", {})):
        J[("real", m)] = (m, c, "real", seeds)
    for na in (10, 25):     # audit-size sensitivity (5 labels per batch is the default above)
        J[("real", f"SLIDE+Audit n_a={na}")] = ("SLIDE+Audit", dict(REV, n_a=na), "real", seeds)
    for m, c in (("SLIDE chunked", dict(REV, chunk=True)), ("Static-MLP chunked", dict(chunk=True)),
                 ("ADWIN-Retrain chunked", dict(chunk=True))):
        J[("chunk", m)] = (m.replace(" chunked", ""), c, "abrupt", seeds)
    return J


def make(name, ctx, cfg):
    c = {k: v for k, v in cfg.items() if k not in ("psi", "delay")}
    if name == "TFIDF-LR":
        return S0.TfidfStatic(ctx)
    if name == "Online-TFIDF-SGD":
        return S.OnlineHashSGD(ctx)
    if name == "Static-FT":
        return S0.StaticFT(ctx)
    if name == "Static-MLP":
        return S.StaticHead(ctx, cfg=c)
    if name == "Online-MLP":
        return S.OnlineHead(ctx)
    if name == "ADWIN-Retrain":
        return S.ErrorDetectorHead(ctx, "ADWIN", cfg=c)
    if name == "KSWIN-Score":
        m = S0.ScoreKSWIN(ctx)
        m.purchase = S.LabelOnAlarmExt.purchase.__get__(m)
        m.ctx = ctx
        return m
    if name.startswith("Budget-Uniform") or name.startswith("Budget-Uncertainty"):
        mode = "uniform" if "Uniform" in name else "uncertainty"
        return S.BudgetStream(ctx, NB, mode, "tfidf" if name.endswith("TFIDF") else "head")
    if name.startswith("Budget-Random") or name.startswith("Budget-Oracle"):
        return S.BudgetSchedule(ctx, "random" if "Random" in name else "oracle", cfg=c,
                                learner="tfidf" if name.endswith("TFIDF") else "head")
    if name == "SLIDE-TFIDF":
        return S.SLIDETfidf(ctx, cfg=c)
    if name == "SLIDE+Audit":
        na = c.pop("n_a", 5)
        return S.SLIDEAudit(ctx, cfg=c, n_a=na)
    return S.SLIDE(ctx, cfg=c, name=name)


def flip_region(texts, Xnp):
    """legitimate NULLH messages in one k-means region (k=8, on warm-standardised phi) whose label flips."""
    from sklearn.cluster import KMeans
    p = CACHE / "flip_rows.npy"
    if p.exists():
        return np.load(p)
    rows = np.flatnonzero(((texts.pool == "NULLH") & (texts.label == 0)).values)
    km = KMeans(8, n_init=10, random_state=0).fit(Xnp[rows])
    share = np.bincount(km.labels_, minlength=8) / len(rows)
    c = int(np.argmin(np.abs(share - 0.2)))       # the region closest to 20% of legitimate mail
    out = rows[km.labels_ == c]
    np.save(p, out)
    return out


def job(key, spec, seed):
    name, cfg, design, _ = spec
    texts = G["texts"]
    X = G["X"] if cfg.get("psi") else G["Xnp"]
    y = texts.label.values.copy()
    if design == "null":
        order, seg = compose_null_stream(texts, seed)
    elif design == "real":
        order, seg, warm_rows = compose_realdrift_stream(texts, seed)
        flip = np.isin(order, flip_region(texts, G["Xnp"])) & (seg == "R1")
        y[order[flip]] = 1        # each row is drawn once, so the flipped rows occur only in R1
    else:
        order, seg = compose_stream_ext(texts, seed, design)
    texts_arr = texts.text.values
    warm_idx = warm_rows if design == "real" else np.flatnonzero((texts.pool == "WARM").values)
    T = int(np.ceil(len(order) / BATCH))
    dspec = cfg.get("delay", "2")
    if dspec.startswith("rand"):
        lo, hi = map(int, dspec[4:].split("-"))
        delays = S.Delays(T, mode="rand", lo=lo, hi=hi, seed=seed)
    else:
        delays = S.Delays(T, D=int(dspec))
    ctx = dict(seed=seed, Xw=X[warm_idx], yw=y[warm_idx], p_ft=G["pft"],
               texts=texts_arr, y=y, warm_idx=warm_idx, D=delays[0], delays=delays, seg=seg, nbatch=BATCH)
    if "WX" in G:
        ctx.update(WX=G["WX"], win_start=G["win_start"], win_end=G["win_end"])
    t0 = time.time()
    m = make(name, ctx, cfg)
    t_init = time.time() - t0
    t0 = time.time()
    sc, used, lat, snap = S.run_stream(m, order, seg, X, y, BATCH, delays)
    wall = time.time() - t0
    yy = y[order]
    rows = [dict(key=key, method=name, seed=seed, segment="ALL", **metrics(yy, sc))]
    for g in np.unique(seg):
        k = seg == g
        rows.append(dict(key=key, method=name, seed=seed, segment=str(g), **metrics(yy[k], sc[k])))
    gen, var = texts.gen.values[order], texts.variant.values[order]
    for g in ("D", "E", "F", "G"):
        for tag, k in (("llm", (seg == g) & (gen == "llm")), ("human", (seg == g) & (gen == "human")),
                       ("para", (seg == g) & (var == "T1_para")),
                       ("translated", (seg == g) & (gen == "llm-translated"))):
            if k.sum() > 0:
                yk, sk = yy[k], sc[k]
                rows.append(dict(key=key, method=name, seed=seed, segment=f"{g}:{tag}", n=int(k.sum()),
                                 recall=float(((sk >= .5) & (yk == 1)).sum() / max((yk == 1).sum(), 1)),
                                 fpr=float(((sk >= .5) & (yk == 0)).sum() / max((yk == 0).sum(), 1))))
    brows = []
    for t in range(T):
        sl = slice(t * BATCH, min((t + 1) * BATCH, len(order)))
        mm = metrics(yy[sl], sc[sl])
        brows.append(dict(key=key, method=name, seed=seed, batch=t, segment=str(seg[sl.start]), mcc=mm["mcc"],
                          recall=mm["recall"], fpr=mm["fpr"]))
    srows = []
    if design == "abrupt" and key[0] in ("main", "chunk", "abl"):
        st = texts[texts.pool.isin(["STRESS", "STRESS_ATT", "STRESS2", "STRESS2_ATT"])]
        st = st.assign(grp=np.where(st.pool.str.startswith("STRESS2"), "S2:" + st.source.astype(str), "S1"))
        for when, f in snap.items():
            for (gk, v), grp in st.groupby(["grp", "variant"]):
                idx = grp.row.values
                pr = f(idx, X[idx]) >= 0.5
                srows.append(dict(key=key, method=name, seed=seed, when=when, sgrp=gk, variant=v,
                                  detect=float(pr.mean()), n=len(idx)))
    ev = dict(key=list(key), method=name, seed=seed, stream_labels=int(used), events=[int(e) for e in m.events],
              t_update=m.t_update, t_init=t_init, wall=wall, ms_score=lat, t_detect=getattr(m, "t_detect", 0.0),
              info=getattr(m, "info", []), audit_alarms=getattr(m, "audit_alarms", 0),
              n_batches=T, seg_bounds=[int(i) for i in range(1, len(seg)) if seg[i] != seg[i - 1]])
    return rows, brows, srows, ev


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--group", default="")
    ap.add_argument("--shard", default="0/1")
    ap.add_argument("--collect", action="store_true")
    ap.add_argument("--fake", action="store_true")
    ap.add_argument("--limit", type=int, default=0)
    a = ap.parse_args()
    J = registry()
    jobs = [(k, s) for k, spec in J.items() if not a.group or k[0] in a.group.split(",") for s in spec[3]]
    done = CACHE / "jobs_rev"
    done.mkdir(exist_ok=True, parents=True)
    fn = lambda k, s: done / (f"{k[0]}__{k[1]}__{s}".replace(" ", "_").replace("/", "-").replace("|", "_") + ".pkl")
    if a.collect:
        out = RESULTS / "revision"
        out.mkdir(exist_ok=True, parents=True)
        R, B, ST, EV, miss = [], [], [], [], 0
        for k, s in jobs:
            if not fn(k, s).exists():
                miss += 1
                continue
            r, b, st, e = pickle.load(open(fn(k, s), "rb"))
            for x in r + b + st:
                x["group"], x["name"] = k[0], k[1]
            R += r; B += b; ST += st; EV.append(e)
        pd.DataFrame(R).drop(columns=["key"]).to_csv(out / "summary.csv", index=False)
        pd.DataFrame(B).drop(columns=["key"]).to_csv(out / "batch.csv.gz", index=False)
        pd.DataFrame(ST).drop(columns=["key"]).to_csv(out / "stress.csv", index=False)
        json.dump(EV, open(out / "events.json", "w"))
        print(f"collected {len(EV)} jobs, {miss} missing")
        return
    todo = [(k, s) for k, s in jobs if not fn(k, s).exists()]
    i, n = map(int, a.shard.split("/"))
    todo = todo[i::n]
    print(f"{len(todo)} jobs in shard {a.shard}", flush=True)
    torch.set_num_threads(1)
    load(a.fake)
    if a.limit:
        todo = todo[:a.limit]
    t0 = time.time()
    for c, (k, s) in enumerate(todo):
        if fn(k, s).exists():
            continue
        r, b, st, e = job(k, J[k], s)
        pickle.dump((r, b, st, e), open(fn(k, s), "wb"))
        print(f"[{a.shard}] {c+1}/{len(todo)} {k} seed={s} wall={e['wall']:.1f}s MCC={r[0]['mcc']:.3f} "
              f"labels={e['stream_labels']} ({time.time()-t0:.0f}s)", flush=True)


if __name__ == "__main__":
    main()
