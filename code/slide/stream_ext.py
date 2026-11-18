"""Revision methods and prequential runner.

Additions over stream.py
  * per-batch label delays (fixed D or random D_t), shared by every method;
  * partial-batch label purchases (equal-budget streaming baselines and the label audit);
  * SLIDE configuration switches: harmonic alpha-spending, update rule, chunked (window-max) scoring;
  * SLIDE-TFIDF: the SLIDE alarm (drift test on the frozen transformer representation) drives label
    purchases for an online TF-IDF learner;
  * Budget-Uniform / Budget-Uncertainty: the same label budget spread over the stream, for both learners;
  * SLIDE+Audit: SLIDE plus a small uniform label audit monitored by ADWIN (real-drift probe).
"""
import copy
import time
import numpy as np
from .config import BATCH
from sklearn.linear_model import SGDClassifier
from sklearn.feature_extraction.text import HashingVectorizer
from river import drift as rdrift
from .models import train_head, predict_proba, clone
from .drift import SlicedDriftTest
from . import stream as S

DEF = dict(S.DEF, spend="none", update="full", chunk=False, tf_epochs=5)


# --------------------------------------------------------------------------- helpers
class Delays:
    def __init__(self, T, D=2, mode="fixed", lo=1, hi=8, seed=0):
        if mode == "fixed":
            self.d = np.full(T, int(D))
        else:
            self.d = np.random.default_rng(7000 + seed).integers(lo, hi + 1, size=T)
        self.arr = {}
        for t, dt in enumerate(self.d):
            self.arr.setdefault(t + int(dt), []).append(t)

    def __getitem__(self, t):
        return int(self.d[min(t, len(self.d) - 1)])


def window_scores(model, ctx, idx, base):
    """max over windows: base = window-0 scores; windows 1.. scored with the same head."""
    st, en, WX = ctx["win_start"], ctx["win_end"], ctx["WX"]
    out = base.copy()
    a, b = st[idx], en[idx]
    has = b > a
    if not has.any():
        return out
    sel = np.concatenate([np.arange(x, y) for x, y in zip(a[has], b[has])])
    owner = np.repeat(np.flatnonzero(has), (b - a)[has])
    p = predict_proba(model, WX[sel])
    np.maximum.at(out, owner, p)
    return out


def make_tfidf(ctx):
    vec = HashingVectorizer(ngram_range=(1, 2), n_features=2 ** 20, alternate_sign=False, norm="l2")
    clf = SGDClassifier(loss="log_loss", alpha=1e-6, random_state=ctx["seed"])
    w = ctx["warm_idx"]
    for _ in range(5):
        clf.partial_fit(vec.transform(ctx["texts"][w]), ctx["y"][w], classes=[0, 1])
    return vec, clf


# --------------------------------------------------------------------------- baselines with delays
class DelayMixin:
    """full-label methods: labels of batch td arrive at td + D_td."""

    def due_labels(self, t, D):
        return [td for td in self.ctx["delays"].arr.get(t, []) if self.wants_labels(td)]


class OnlineHashSGD(DelayMixin, S.OnlineHashSGD):
    def __init__(self, ctx):
        self.ctx = ctx
        super().__init__(ctx)


class ErrorDetectorHead(DelayMixin, S.ErrorDetectorHead):
    def __init__(self, ctx, kind="ADWIN", cfg=None):
        self.ctx = ctx
        super().__init__(ctx, kind)
        self.cfg = dict(DEF, **(cfg or {}))

    def score(self, idx, X):
        s = predict_proba(self.model, X)
        return window_scores(self.model, self.ctx, idx, s) if self.cfg["chunk"] else s

    def snapshot(self):
        m, c, ctx = clone(self.model), self.cfg["chunk"], self.ctx
        return lambda idx, X: window_scores(m, ctx, idx, predict_proba(m, X)) if c else predict_proba(m, X)


class OnlineHead(DelayMixin, S.OnlineHead):
    def __init__(self, ctx):
        self.ctx = ctx
        super().__init__(ctx)


class StaticHead(S.StaticHead):
    def __init__(self, ctx, cfg=None):
        self.ctx = ctx
        super().__init__(ctx)
        self.cfg = dict(DEF, **(cfg or {}))

    def score(self, idx, X):
        s = predict_proba(self.model, X)
        return window_scores(self.model, self.ctx, idx, s) if self.cfg["chunk"] else s

    def snapshot(self):
        return self.score


# --------------------------------------------------------------------------- label-on-alarm family
class LabelOnAlarmExt(S.LabelOnAlarm):
    def __init__(self, ctx, robust, cfg=None):
        self.ctx = ctx
        super().__init__(ctx, robust=robust, cfg=dict(DEF, **(cfg or {})))
        self.cfg = dict(DEF, **(cfg or {}))
        lw = self.cfg["label_win"]
        self.label_sub = lw if lw < BATCH else None

    def purchase(self, t, nb):
        D = self.ctx["delays"][t]
        self.due.setdefault(t + D, []).extend(range(t - nb + 1, t + 1))
        self.retrain_at = t + D
        self.cooldown = t + D

    def maybe_retrain(self, t):
        if self.retrain_at is None or t < self.retrain_at or not self.lab_X:
            return
        u = self.cfg["update"]
        if u == "full":
            return super().maybe_retrain(t)
        t0 = time.time()
        Xn, yn = np.vstack(self.lab_X), np.concatenate(self.lab_y)
        c = self.cfg
        if u == "scratch":          # re-fit a fresh head on the purchased window only
            self.model = train_head(Xn, yn, seed=self.seed + t)
        elif u == "warm":           # warm start, window only, no regulariser
            self.model = train_head(Xn, yn, model=clone(self.model), seed=self.seed + t)
        elif u == "warm+replay":    # warm start with class-balanced replay, no adversarial / proximal term
            Xm, ym = self.mem.data()
            self.model = train_head(Xn, yn, model=clone(self.model), Xm=Xm, ym=ym, lam_r=c["lam_r"],
                                    seed=self.seed + t)
        else:
            raise ValueError(u)
        self.mem.add(Xn, yn)
        self.lab_X, self.lab_y, self.retrain_at = [], [], None
        self.t_update += time.time() - t0
        self.events.append(t)
        self.on_retrained(t)


class SLIDE(LabelOnAlarmExt):
    name = "SLIDE"

    def __init__(self, ctx, cfg=None, name=None):
        c0 = dict(DEF, **(cfg or {}))
        # the initial head uses the adversarial term only when the update rule does
        robust = c0["update"] == "full"
        LabelOnAlarmExt.__init__(self, ctx, robust=robust, cfg=cfg)
        if c0["update"] != "full":
            self.robust = False
        c = self.cfg
        if name:
            self.name = name
        self.det = SlicedDriftTest(ctx["Xw"].shape[1], K=c["K"], alpha=c["alpha"], use_ks=c["use_ks"],
                                   use_sw=c["use_sw"], use_proj=c["use_proj"], use_score=c["use_score"],
                                   seed=self.seed, spend=c["spend"])
        self.t_detect, self.log, self.info = 0.0, [], []
        self.need_ref, self.since_ref = True, 0

    # scorer of the detector itself (window-0 score feeds the drift test's score projection)
    def base_score(self, X):
        return predict_proba(self.model, X)

    def score(self, idx, X):
        s = self.base_score(X)
        return window_scores(self.model, self.ctx, idx, s) if self.cfg["chunk"] else s

    def snapshot(self):
        m, c, ctx = clone(self.model), self.cfg["chunk"], self.ctx
        return lambda idx, X: window_scores(m, ctx, idx, predict_proba(m, X)) if c else predict_proba(m, X)

    def ref_scores(self, X):
        return self.base_score(X)

    def step(self, t, X, nbatch):
        self.observe(t, X)
        c = self.cfg
        if not self.buf_X:
            return
        recent = np.vstack(self.buf_X)
        if self.need_ref:
            if len(recent) >= c["w_ref"]:
                Xr = recent[-c["w_ref"]:]
                self.det.set_reference(Xr, self.ref_scores(Xr), c["w_test"])
                self.need_ref, self.since_ref = False, 0
            return
        self.since_ref += len(X)
        if not c["adapt"] or self.since_ref < c["w_test"] or t <= self.cooldown:
            return
        t0 = time.time()
        W = recent[-c["w_test"]:]
        alarm, info = self.det.test(W, self.ref_scores(W))
        self.t_detect += time.time() - t0
        self.log.append((t, info["sw"], info["tau"], info["minp"], alarm))
        self.info.append(dict(t=int(t), alarm=bool(alarm), **{k: float(v) for k, v in info.items()}))
        if alarm:
            self.purchase(t, int(np.ceil(c["label_win"] / nbatch)))

    def on_retrained(self, t):
        self.need_ref = True
        self.buf_X = []


class SLIDETfidf(SLIDE):
    """SLIDE alarms (test on the frozen transformer representation, score projection = TF-IDF score);
    the purchased labels update an online TF-IDF learner instead of the transformer head."""
    name = "SLIDE-TFIDF"

    def __init__(self, ctx, cfg=None, name=None):
        SLIDE.__init__(self, ctx, cfg=dict(dict(update="warm"), **(cfg or {})), name=name)
        self.vec, self.clf = make_tfidf(ctx)
        self.texts = ctx["texts"]
        self.lab_idx = []
        self.cur_idx = None

    def score(self, idx, X):
        self.cur_idx = idx
        return self.clf.predict_proba(self.vec.transform(self.texts[idx]))[:, 1]

    def ref_scores(self, X):
        # the score projection of the test uses the TF-IDF learner's score of the same messages
        return self._buf_scores[-len(X):]

    def observe(self, t, X):
        super().observe(t, X)
        s = self.clf.predict_proba(self.vec.transform(self.texts[self.cur_idx]))[:, 1]
        self._sbuf = getattr(self, "_sbuf", []) + [s]
        self._sbuf = self._sbuf[-len(self.buf_X):]

    @property
    def _buf_scores(self):
        return np.concatenate(self._sbuf)

    def receive(self, t, idx, X, y):
        self.lab_X.append(X)
        self.lab_y.append(y)
        self.lab_idx.append(idx)

    def maybe_retrain(self, t):
        if self.retrain_at is None or t < self.retrain_at or not self.lab_X:
            return
        t0 = time.time()
        ii, yy = np.concatenate(self.lab_idx), np.concatenate(self.lab_y)
        Z = self.vec.transform(self.texts[ii])
        for _ in range(self.cfg["tf_epochs"]):
            self.clf.partial_fit(Z, yy)
        self.lab_X, self.lab_y, self.lab_idx, self.retrain_at = [], [], [], None
        self.t_update += time.time() - t0
        self.events.append(t)
        self.on_retrained(t)

    def on_retrained(self, t):
        self.need_ref = True
        self.buf_X, self._sbuf = [], []

    def snapshot(self):
        c = copy.deepcopy(self.clf)
        return lambda idx, X: c.predict_proba(self.vec.transform(self.texts[idx]))[:, 1]


class BudgetSchedule(LabelOnAlarmExt):
    """equal-budget timing controls with the SLIDE update rule (cf. stream.BudgetSchedule)."""

    def __init__(self, ctx, mode="random", cfg=None, learner="head"):
        LabelOnAlarmExt.__init__(self, ctx, robust=(dict(DEF, **(cfg or {}))["update"] == "full"), cfg=cfg)
        self.mode = mode
        self.name = ("Budget-Random" if mode == "random" else "Budget-Oracle") + ("-TFIDF" if learner == "tfidf" else "")
        T = int(np.ceil(len(ctx["seg"]) / ctx["nbatch"]))
        seg = ctx["seg"]
        b = [i for i in range(1, len(seg)) if seg[i] != seg[i - 1]]
        # boundaries = first message of a new segment that is not interleaved back (gradual: zone start)
        starts, seen = [], {seg[0]}
        for i in b:
            if seg[i] not in seen:
                starts.append(i)
                seen.add(seg[i])
        n_purchases = len(starts)          # random control buys as many windows as there are boundaries
        if mode == "random":
            rng = np.random.default_rng(1000 + ctx["seed"])
            self.when = set(int(v) for v in rng.choice(np.arange(2, T - 3), n_purchases, replace=False))
        else:
            self.when = set(int(i // ctx["nbatch"]) + 1 for i in starts)
        self.learner = learner
        if learner == "tfidf":
            self.vec, self.clf = make_tfidf(ctx)
            self.texts = ctx["texts"]
            self.lab_idx = []

    def score(self, idx, X):
        if self.learner == "tfidf":
            return self.clf.predict_proba(self.vec.transform(self.texts[idx]))[:, 1]
        return predict_proba(self.model, X)

    def snapshot(self):
        if self.learner == "tfidf":
            c = copy.deepcopy(self.clf)
            return lambda idx, X: c.predict_proba(self.vec.transform(self.texts[idx]))[:, 1]
        m = clone(self.model)
        return lambda idx, X: predict_proba(m, X)

    def step(self, t, X, nbatch):
        self.observe(t, X)
        if t in self.when:
            self.purchase(t, int(np.ceil(self.cfg["label_win"] / nbatch)))

    def receive(self, t, idx, X, y):
        super().receive(t, idx, X, y)
        if self.learner == "tfidf":
            self.lab_idx.append(idx)

    def maybe_retrain(self, t):
        if self.learner != "tfidf":
            return super().maybe_retrain(t)
        if self.retrain_at is None or t < self.retrain_at or not self.lab_X:
            return
        ii, yy = np.concatenate(self.lab_idx), np.concatenate(self.lab_y)
        Z = self.vec.transform(self.texts[ii])
        for _ in range(self.cfg["tf_epochs"]):
            self.clf.partial_fit(Z, yy)
        self.lab_X, self.lab_y, self.lab_idx, self.retrain_at = [], [], [], None
        self.events.append(t)


# --------------------------------------------------------------------------- streaming equal-budget baselines
class BudgetStream(S.Base):
    """buy n_b labels in every batch (uniform at random, or the n_b least confident), labels arrive after D_t;
    learner 'head' (incremental update of the transformer head, as Online-MLP) or 'tfidf' (online TF-IDF SGD)."""

    def __init__(self, ctx, n_b=18, mode="uniform", learner="tfidf"):
        super().__init__()
        self.ctx, self.n_b, self.mode, self.learner = ctx, n_b, mode, learner
        self.name = f"Budget-{mode.capitalize()}-{'TFIDF' if learner == 'tfidf' else 'MLP'}"
        self.rng = np.random.default_rng(3000 + ctx["seed"])
        self.pick = {}
        self.seed = ctx["seed"]
        if learner == "tfidf":
            self.vec, self.clf = make_tfidf(ctx)
            self.texts = ctx["texts"]
        else:
            self.model = train_head(ctx["Xw"], ctx["yw"], seed=self.seed)
        self.t = 0

    def score(self, idx, X):
        if self.learner == "tfidf":
            s = self.clf.predict_proba(self.vec.transform(self.texts[idx]))[:, 1]
        else:
            s = predict_proba(self.model, X)
        n = min(self.n_b, len(idx))
        if self.mode == "uniform":
            sel = self.rng.choice(len(idx), n, replace=False)
        else:
            sel = np.argsort(np.abs(s - 0.5))[:n]
        self.pick[self.t] = sel
        self.t += 1
        return s

    def wants_labels(self, t):
        return True

    def due_labels(self, t, D):
        return list(self.ctx["delays"].arr.get(t, []))

    def receive(self, t, idx, X, y):
        sel = self.pick.pop(t)
        t0 = time.time()
        if self.learner == "tfidf":
            self.clf.partial_fit(self.vec.transform(self.texts[idx[sel]]), y[sel])
        else:
            self.model = train_head(X[sel], y[sel], model=self.model, epochs=3, lr=5e-4, seed=self.seed + t)
        self.t_update += time.time() - t0
        return len(sel)

    def snapshot(self):
        if self.learner == "tfidf":
            c = copy.deepcopy(self.clf)
            return lambda idx, X: c.predict_proba(self.vec.transform(self.texts[idx]))[:, 1]
        m = clone(self.model)
        return lambda idx, X: predict_proba(m, X)


class SLIDEAudit(SLIDE):
    """SLIDE plus a uniform audit of n_a labels per batch whose prequential errors feed ADWIN;
    an ADWIN alarm buys the same 500-label window as a sliced alarm (covers real drift with P(X) unchanged)."""
    name = "SLIDE+Audit"

    def __init__(self, ctx, cfg=None, name=None, n_a=5):
        SLIDE.__init__(self, ctx, cfg=cfg, name=name)
        self.n_a = n_a
        self.arng = np.random.default_rng(4000 + ctx["seed"])
        self.audit = {}
        self.adwin = rdrift.ADWIN(delta=0.002)
        self.audit_alarms = 0

    def step(self, t, X, nbatch):
        sel = self.arng.choice(len(X), min(self.n_a, len(X)), replace=False)
        self.audit[t] = (sel, predict_proba(self.model, X[sel]))
        D = self.ctx["delays"][t]
        self.due.setdefault(t + D, []).append(-(t + 1))     # negative id marks an audit-only delivery
        super().step(t, X, nbatch)
        self._nb = nbatch

    def due_labels(self, t, D):
        ks = self.due.pop(t, [])
        out = sorted(set([k for k in ks if k >= 0] + [-k - 1 for k in ks if k < 0]))
        self._full = set(k for k in ks if k >= 0)
        return out

    def receive(self, t, idx, X, y):
        n = 0
        if t in self._full:
            self.lab_X.append(X)
            self.lab_y.append(y)
            n = len(y)
        if t in self.audit:
            sel, p = self.audit.pop(t)
            hit = False
            for e in ((p >= 0.5).astype(int) != y[sel]):
                self.adwin.update(float(e))
                hit |= self.adwin.drift_detected
            if t not in self._full:
                n = len(sel)
            # error alarm while no purchase is pending -> buy the most recent label window
            if hit and self.retrain_at is None and self.cur_t > self.cooldown:
                self.audit_alarms += 1
                self.purchase(self.cur_t, int(np.ceil(self.cfg["label_win"] / self._nb)))
                self.adwin = rdrift.ADWIN(delta=0.002)
        return n

    def observe(self, t, X):
        self.cur_t = t
        super().observe(t, X)


# --------------------------------------------------------------------------- runner
def run_stream(method, order, seg, Xall, y, nbatch, delays):
    T = int(np.ceil(len(order) / nbatch))
    scores = np.zeros(len(order))
    used, lat = 0, 0.0
    snap = {"t0": method.snapshot()}
    for t in range(T):
        sl = slice(t * nbatch, min((t + 1) * nbatch, len(order)))
        idx = order[sl]
        X = Xall[idx]
        t0 = time.time()
        scores[sl] = method.score(idx, X)
        lat += time.time() - t0
        if hasattr(method, "step"):
            method.step(t, X, nbatch)
        for td in method.due_labels(t, 0):
            s2 = slice(td * nbatch, min((td + 1) * nbatch, len(order)))
            ix = order[s2]
            sub = getattr(method, "label_sub", None)      # label windows smaller than a batch: random subset of it
            if sub and len(ix) > sub:
                ix = np.sort(np.random.default_rng(1000 * td + 7).choice(ix, sub, replace=False))
            r = method.receive(td, ix, Xall[ix], y[ix])
            used += len(ix) if r is None else r
        if hasattr(method, "maybe_retrain"):
            method.maybe_retrain(t)
        if sl.stop < len(order) and seg[sl.stop - 1] == "D" and seg[sl.stop] == "E":
            snap["preE"] = method.snapshot()
    snap["end"] = method.snapshot()
    return scores, used, lat / len(order) * 1000.0, snap
