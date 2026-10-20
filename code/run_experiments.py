"""Stage 3: prequential stream experiments, ablations, sensitivity and stress tests.

usage: python run_experiments.py [--seeds 10] [--workers 4] [--fake]
"""
import argparse
import json
import time
import numpy as np
import pandas as pd
import torch
import os
import pickle
import multiprocessing as mp
from slide.config import CACHE, RESULTS, BATCH, LABEL_DELAY, STREAM_SEEDS
from slide.data import build_pools, compose_stream
from slide import stream as S
from slide.metrics import metrics

MAIN = ["TFIDF-LR", "Online-TFIDF-SGD", "Static-FT", "Static-MLP", "Online-MLP", "Periodic-Retrain",
        "ADWIN-Retrain", "PH-Retrain", "KSWIN-Score", "SLIDE"]
CTRL = ["KSWIN-Score+SLIDE-update", "Budget-Random", "Budget-Oracle"]   # equal-budget and update-rule controls
ABL = {"SLIDE w/o adversarial term": dict(lam_a=0.0),
       "SLIDE w/o replay": dict(lam_r=0.0),
       "SLIDE w/o proximal term": dict(lam_p=0.0),
       "SLIDE w/o surprisal psi": dict(use_psi=False),
       "SLIDE w/o KS gate": dict(use_ks=False),
       "SLIDE w/o SW gate": dict(use_sw=False),
       "SLIDE score-only test": dict(use_proj=False),
       "SLIDE projections-only test": dict(use_score=False),
       "SLIDE w/o adaptation": dict(adapt=False)}
SENS = {f"SLIDE eps={e}": dict(eps=e) for e in (0.25, 0.5, 2.0, 4.0)}
SENS.update({f"SLIDE alpha={a}": dict(alpha=a) for a in (0.001, 0.05)})
SENS.update({f"SLIDE label_win={w}": dict(label_win=w) for w in (250, 1000)})
SENS.update({f"SLIDE K={k}": dict(K=k) for k in (16, 256)})

G = {}


def load_features(fake=False):
    texts = pd.read_parquet(CACHE / ("texts_fake.parquet" if fake else "texts.parquet"))
    if fake:
        rng = np.random.default_rng(0)
        y = texts.label.values
        phi = rng.standard_normal((len(texts), 768)).astype(np.float32) + 0.3 * y[:, None]
        psi = rng.standard_normal((len(texts), 4)).astype(np.float32)
        pft = 1 / (1 + np.exp(-(phi[:, 0] * 2)))
    else:
        from slide.features import embed, surprisal
        phi, pft = embed(texts.text.tolist())
        psi = surprisal(texts.text.tolist())
    warm = (texts.pool == "WARM").values
    mu, sd = phi[warm].mean(0), phi[warm].std(0) + 1e-6
    mu2, sd2 = psi[warm].mean(0), psi[warm].std(0) + 1e-6
    Zphi, Zpsi = (phi - mu) / sd, (psi - mu2) / sd2
    return texts, np.hstack([Zphi, Zpsi]).astype(np.float32), Zphi.astype(np.float32), pft


def init(fake):
    texts, X, Xnp, pft = load_features(fake)
    torch.set_num_threads(1)                    # one thread per worker process; workers run in parallel
    n = len(build_pools())
    G.update(texts=texts, X=X, Xnp=Xnp, pft=pft, pools=texts.iloc[:n])


def make(name, ctx, cfg=None):
    if name == "TFIDF-LR":
        return S.TfidfStatic(ctx)
    if name == "Online-TFIDF-SGD":
        return S.OnlineHashSGD(ctx)
    if name == "Static-FT":
        return S.StaticFT(ctx)
    if name == "Static-MLP":
        return S.StaticHead(ctx)
    if name == "Online-MLP":
        return S.OnlineHead(ctx)
    if name == "Periodic-Retrain":
        return S.PeriodicHead(ctx)
    if name == "ADWIN-Retrain":
        return S.ErrorDetectorHead(ctx, "ADWIN")
    if name == "PH-Retrain":
        return S.ErrorDetectorHead(ctx, "PH")
    if name == "KSWIN-Score":
        return S.ScoreKSWIN(ctx)
    if name == "KSWIN-Score+SLIDE-update":
        return S.ScoreKSWINRobust(ctx)
    if name == "Budget-Random":
        return S.BudgetSchedule(ctx, "random")
    if name == "Budget-Oracle":
        return S.BudgetSchedule(ctx, "oracle")
    return S.SLIDE(ctx, cfg=cfg, name=name)


def job(arg):
    name, cfg, seed = arg
    texts, pools = G["texts"], G["pools"]
    use_psi = (cfg or {}).get("use_psi", True)
    X = G["X"] if use_psi else G["Xnp"]
    y = texts.label.values
    order, seg = compose_stream(pools, seed)
    warm_idx = np.flatnonzero((texts.pool == "WARM").values)
    ctx = dict(seed=seed, Xw=X[warm_idx], yw=y[warm_idx], p_ft=G["pft"], texts=texts.text.values,
               y=y, warm_idx=warm_idx, D=LABEL_DELAY, seg=seg, nbatch=BATCH)
    t0 = time.time()
    m = make(name, ctx, cfg)
    t_init = time.time() - t0
    t0 = time.time()
    sc, used, lat, snap = S.run_stream(m, order, seg, X, y, BATCH, LABEL_DELAY)
    wall = time.time() - t0
    yy = y[order]
    rows = [dict(method=name, seed=seed, segment="ALL", **metrics(yy, sc))]
    for g in "ABCDE":
        k = seg == g
        rows.append(dict(method=name, seed=seed, segment=g, **metrics(yy[k], sc[k])))
    T = int(np.ceil(len(order) / BATCH))
    brows = []
    for t in range(T):
        sl = slice(t * BATCH, min((t + 1) * BATCH, len(order)))
        mm = metrics(yy[sl], sc[sl])
        brows.append(dict(method=name, seed=seed, batch=t, segment=seg[sl.start], mcc=mm["mcc"], f1=mm["f1"],
                          pr_auc=mm["pr_auc"], recall=mm["recall"], fpr=mm["fpr"], f1_macro=mm["f1_macro"]))
    # generated vs human slices inside D/E (construct-validity check)
    gen = texts.gen.values[order]
    var = texts.variant.values[order]
    for g in "DE":
        for tag, k in (("llm", (seg == g) & (gen == "llm")), ("human", (seg == g) & (gen == "human")),
                       ("para", (seg == g) & (var == "T1_para"))):
            if k.sum() > 0:
                yk, sk = yy[k], sc[k]
                rows.append(dict(method=name, seed=seed, segment=f"{g}:{tag}", n=int(k.sum()),
                                 recall=float(((sk >= .5) & (yk == 1)).sum() / max((yk == 1).sum(), 1)),
                                 fpr=float(((sk >= .5) & (yk == 0)).sum() / max((yk == 0).sum(), 1))))
    # stress test on held-out phishing and its transformations
    srows = []
    st = texts[texts.pool.isin(["STRESS", "STRESS_ATT"])]
    for when, f in snap.items():
        for v, grp in st.groupby("variant"):
            idx = grp.row.values
            pr = f(idx, X[idx]) >= 0.5
            srows.append(dict(method=name, seed=seed, when=when, variant=v, detect=float(pr.mean()), n=len(idx)))
    ev = dict(method=name, seed=seed, labels=int(used + len(warm_idx)), stream_labels=int(used),
              events=[int(e) for e in m.events], t_update=m.t_update, t_init=t_init, wall=wall, ms_score=lat,
              t_detect=getattr(m, "t_detect", 0.0),
              dlog=[[int(a), float(b), float(c), float(d), bool(e)] for a, b, c, d, e in getattr(m, "log", [])])
    return rows, brows, srows, ev


def job_keyed(j):
    return j, job(j)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--seeds", type=int, default=len(STREAM_SEEDS))
    ap.add_argument("--sens_seeds", type=int, default=5)
    ap.add_argument("--workers", type=int, default=4)
    ap.add_argument("--fake", action="store_true")
    ap.add_argument("--only", default="")
    ap.add_argument("--tag", default="")
    ap.add_argument("--shard", default="", help="i/n: run only every n-th pending job starting at i, in-process")
    ap.add_argument("--collect", action="store_true", help="only assemble cached job files")
    a = ap.parse_args()
    jobs = []
    if a.only in ("", "main"):
        jobs += [(n, None, s) for n in MAIN + CTRL for s in range(a.seeds)]
    if a.only in ("", "abl"):
        jobs += [(n, c, s) for n, c in ABL.items() for s in range(a.seeds)]
    if a.only in ("", "sens"):
        jobs += [(n, c, s) for n, c in SENS.items() for s in range(a.sens_seeds)]
    out = RESULTS / f"raw{a.tag}"
    done_dir = CACHE / f"jobs{a.tag}"           # one file per finished job, so interrupted runs resume
    done_dir.mkdir(exist_ok=True, parents=True)
    key = lambda j: f"{j[0].replace(' ', '_').replace('/', '-')}__{j[2]}.pkl"
    todo = [j for j in jobs if not (done_dir / key(j)).exists()]
    print(f"{len(jobs) - len(todo)} cached, {len(todo)} to run", flush=True)
    t0 = time.time()
    if a.collect:
        todo = []
    if todo and a.shard:
        i, n = map(int, a.shard.split("/"))
        init(a.fake)
        for k, j in enumerate(todo[i::n]):
            if (done_dir / key(j)).exists():
                continue
            r, b, s, e = job(j)
            pickle.dump((r, b, s, e), open(done_dir / key(j), "wb"))
            print(f"[shard {i}] {k+1}/{len(todo[i::n])} {e['method']} seed={e['seed']} wall={e['wall']:.1f}s "
                  f"MCC={r[0]['mcc']:.3f} labels={e['stream_labels']} ({time.time()-t0:.0f}s)", flush=True)
        return
    if todo:
        ctx = mp.get_context("spawn")             # fork + OpenMP slows torch workers by an order of magnitude
        with ctx.Pool(a.workers, initializer=init, initargs=(a.fake,)) as p:
            for k, res in enumerate(p.imap_unordered(job_keyed, todo)):
                j, (r, b, s, e) = res
                pickle.dump((r, b, s, e), open(done_dir / key(j), "wb"))
                print(f"[{k+1}/{len(todo)}] {e['method']} seed={e['seed']} wall={e['wall']:.1f}s "
                      f"MCC={r[0]['mcc']:.3f} labels={e['stream_labels']} ({time.time()-t0:.0f}s)", flush=True)
    out.mkdir(exist_ok=True, parents=True)
    R, B, ST, EV = [], [], [], []
    for j in jobs:
        r, b, s, e = pickle.load(open(done_dir / key(j), "rb"))
        R += r; B += b; ST += s; EV.append(e)
    pd.DataFrame(R).to_csv(out / "summary.csv", index=False)
    pd.DataFrame(B).to_csv(out / "batch.csv", index=False)
    pd.DataFrame(ST).to_csv(out / "stress.csv", index=False)
    json.dump(EV, open(out / "events.json", "w"))


if __name__ == "__main__":
    main()
