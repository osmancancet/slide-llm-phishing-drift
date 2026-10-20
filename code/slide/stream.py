"""Prequential (test-then-train) evaluation with delayed and budgeted labels.

At batch t the detector scores X_t first; labels of batch t are revealed at batch t + D only to methods
that request them. Label consumption is counted per method.
"""
import time
import numpy as np
from sklearn.linear_model import SGDClassifier, LogisticRegression
from sklearn.feature_extraction.text import TfidfVectorizer, HashingVectorizer
from river import drift as rdrift
from .models import train_head, predict_proba, Reservoir, clone
from .drift import SlicedDriftTest

DEF = dict(lam_r=1.0, lam_a=0.5, eps=1.0, lam_p=1e-3, K=64, alpha=0.01, w_ref=1000, w_test=500,
           label_win=500, mem_cap=1000, use_ks=True, use_sw=True, use_proj=True, use_score=True,
           use_psi=True, adapt=True)


class Base:
    name = "base"
    labels_used = 0

    def __init__(self):
        self.events, self.t_update = [], 0.0

    def score(self, idx, X):
        raise NotImplementedError

    def receive(self, t, idx, X, y):
        """labels for batch t become available (only called if the method requested them)."""

    def wants_labels(self, t):
        return False

    def due_labels(self, t, D):
        """batches whose labels arrive at batch t (default: batch t - D, if requested)."""
        return [t - D] if t - D >= 0 and self.wants_labels(t - D) else []

    def snapshot(self):
        """frozen scorer (idx, X) -> scores, used for the stress test."""
        return self.score


class StaticFT(Base):
    name = "Static-FT"

    def __init__(self, ctx):
        super().__init__()
        self.p = ctx["p_ft"]

    def score(self, idx, X):
        return self.p[idx]


class TfidfStatic(Base):
    name = "TFIDF-LR"

    def __init__(self, ctx):
        super().__init__()
        self.texts = ctx["texts"]
        self.vec = TfidfVectorizer(ngram_range=(1, 2), min_df=2, max_features=100000, sublinear_tf=True)
        Xw = self.vec.fit_transform(self.texts[ctx["warm_idx"]])
        self.clf = LogisticRegression(C=10, max_iter=2000, class_weight="balanced").fit(Xw, ctx["y"][ctx["warm_idx"]])

    def score(self, idx, X):
        return self.clf.predict_proba(self.vec.transform(self.texts[idx]))[:, 1]


class OnlineHashSGD(Base):
    name = "Online-TFIDF-SGD"

    def __init__(self, ctx):
        super().__init__()
        self.texts = ctx["texts"]
        self.vec = HashingVectorizer(ngram_range=(1, 2), n_features=2 ** 20, alternate_sign=False, norm="l2")
        self.clf = SGDClassifier(loss="log_loss", alpha=1e-6, class_weight=None, random_state=ctx["seed"])
        w = ctx["warm_idx"]
        for _ in range(5):
            self.clf.partial_fit(self.vec.transform(self.texts[w]), ctx["y"][w], classes=[0, 1])

    def wants_labels(self, t):
        return True

    def score(self, idx, X):
        return self.clf.predict_proba(self.vec.transform(self.texts[idx]))[:, 1]

    def snapshot(self):
        import copy
        c = copy.deepcopy(self.clf)
        return lambda idx, X: c.predict_proba(self.vec.transform(self.texts[idx]))[:, 1]

    def receive(self, t, idx, X, y):
        t0 = time.time()
        self.clf.partial_fit(self.vec.transform(self.texts[idx]), y)
        self.t_update += time.time() - t0


class HeadMethod(Base):
    """shared machinery for methods built on the representation x = [phi, psi]."""

    def __init__(self, ctx, robust=False, cfg=None):
        super().__init__()
        self.cfg = dict(DEF, **(cfg or {}))
        self.seed = ctx["seed"]
        self.robust = robust
        Xw, yw = ctx["Xw"], ctx["yw"]
        self.model = train_head(Xw, yw, seed=self.seed, **self._loss_kw(initial=True))
        self.hist_X, self.hist_y = [Xw], [yw]

    def _loss_kw(self, initial=False):
        if not self.robust:
            return dict(lam_r=0.0, lam_a=0.0, eps=0.0, lam_p=0.0)
        c = self.cfg
        return dict(lam_r=c["lam_r"], lam_a=c["lam_a"], eps=c["eps"], lam_p=0.0 if initial else c["lam_p"])

    def score(self, idx, X):
        return predict_proba(self.model, X)

    def snapshot(self):
        m = clone(self.model)
        return lambda idx, X: predict_proba(m, X)

    def recent_labeled(self, n):
        X = np.vstack(self.hist_X)[-n:]
        y = np.concatenate(self.hist_y)[-n:]
        return X, y


class StaticHead(HeadMethod):
    name = "Static-MLP"


class OnlineHead(HeadMethod):
    """incremental head update on every delayed labelled batch (full supervision)."""
    name = "Online-MLP"

    def wants_labels(self, t):
        return True

    def receive(self, t, idx, X, y):
        t0 = time.time()
        self.model = train_head(X, y, model=self.model, epochs=3, lr=5e-4, seed=self.seed + t)
        self.t_update += time.time() - t0


class PeriodicHead(HeadMethod):
    """blind re-training from scratch on the last `win` labels every `period` batches."""
    name = "Periodic-Retrain"

    def __init__(self, ctx, period=8, win=2000):
        super().__init__(ctx)
        self.period, self.win, self.k = period, win, 0

    def wants_labels(self, t):
        return True

    def receive(self, t, idx, X, y):
        self.hist_X.append(X)
        self.hist_y.append(y)
        self.k += 1
        if self.k % self.period == 0:
            t0 = time.time()
            Xr, yr = self.recent_labeled(self.win)
            self.model = train_head(Xr, yr, seed=self.seed + t)
            self.t_update += time.time() - t0
            self.events.append(t)


class ErrorDetectorHead(HeadMethod):
    """supervised detector (ADWIN / Page-Hinkley) on the delayed 0/1 error stream -> retrain on last `win` labels."""

    def __init__(self, ctx, kind="ADWIN", win=1000):
        super().__init__(ctx)
        self.kind, self.win = kind, win
        self.name = f"{kind}-Retrain"
        self.det = rdrift.ADWIN(delta=0.002) if kind == "ADWIN" else rdrift.PageHinkley(threshold=50, min_instances=200)
        self.last = -10

    def wants_labels(self, t):
        return True

    def receive(self, t, idx, X, y):
        self.hist_X.append(X)
        self.hist_y.append(y)
        err = (predict_proba(self.model, X) >= 0.5).astype(int) != y
        hit = False
        for e in err:
            self.det.update(float(e))
            hit |= self.det.drift_detected
        if hit and t - self.last >= 2:
            t0 = time.time()
            Xr, yr = self.recent_labeled(self.win)
            self.model = train_head(Xr, yr, seed=self.seed + t)
            self.t_update += time.time() - t0
            self.events.append(t)
            self.last = t
            self.det = rdrift.ADWIN(delta=0.002) if self.kind == "ADWIN" else rdrift.PageHinkley(threshold=50, min_instances=200)


class LabelOnAlarm(HeadMethod):
    """common logic for unsupervised alarms that purchase the labels of the alarm window."""

    def __init__(self, ctx, robust, cfg=None):
        super().__init__(ctx, robust=robust, cfg=cfg)
        self.buf_X, self.buf_idx = [], []
        self.due = {}               # arrival batch -> list of purchased batches delivered then
        self.cooldown = -1
        self.mem = Reservoir(self.cfg["mem_cap"], ctx["Xw"].shape[1], seed=self.seed)
        self.mem.add(ctx["Xw"], ctx["yw"])
        self.lab_X, self.lab_y = [], []
        self.retrain_at = None
        self.D = ctx["D"]

    def observe(self, t, X):
        self.buf_X.append(X)
        self.buf_X = self.buf_X[-int(np.ceil(self.cfg["w_ref"] / len(X))) - 2:]

    def buffered(self):
        return sum(len(b) for b in self.buf_X)

    def due_labels(self, t, D):
        return [k for k in self.due.pop(t, []) if k >= 0]

    def purchase(self, t, nb):
        """buy labels of the last nb batches; all of them arrive D batches after the purchase."""
        self.due.setdefault(t + self.D, []).extend(range(t - nb + 1, t + 1))
        self.retrain_at = t + self.D       # update right after the labels arrive
        self.cooldown = t + self.D

    def receive(self, t, idx, X, y):
        self.lab_X.append(X)
        self.lab_y.append(y)

    def maybe_retrain(self, t):
        if self.retrain_at is None or t < self.retrain_at or not self.lab_X:
            return
        t0 = time.time()
        Xn, yn = np.vstack(self.lab_X), np.concatenate(self.lab_y)
        c = self.cfg
        if self.robust:
            Xm, ym = self.mem.data()
            self.model = train_head(Xn, yn, model=clone(self.model),
                                    Xm=Xm, ym=ym, lam_r=c["lam_r"], lam_a=c["lam_a"], eps=c["eps"],
                                    lam_p=c["lam_p"], seed=self.seed + t)
        else:
            self.model = train_head(Xn, yn, seed=self.seed + t)
        self.mem.add(Xn, yn)
        self.lab_X, self.lab_y, self.retrain_at = [], [], None
        self.t_update += time.time() - t0
        self.events.append(t)
        self.on_retrained(t)

    def on_retrained(self, t):
        pass


class SLIDE(LabelOnAlarm):
    name = "SLIDE"

    def __init__(self, ctx, cfg=None, name=None):
        super().__init__(ctx, robust=True, cfg=cfg)
        c = self.cfg
        if name:
            self.name = name
        self.det = SlicedDriftTest(ctx["Xw"].shape[1], K=c["K"], alpha=c["alpha"], use_ks=c["use_ks"],
                                   use_sw=c["use_sw"], use_proj=c["use_proj"], use_score=c["use_score"],
                                   seed=self.seed)
        self.t_detect, self.log = 0.0, []
        self.need_ref = True        # first reference = first w_ref stream messages (disjoint from training data)
        self.since_ref = 0

    def step(self, t, X, nbatch):
        """called after scoring batch t."""
        self.observe(t, X)
        c = self.cfg
        if not self.buf_X:
            return
        recent = np.vstack(self.buf_X)
        if self.need_ref:
            if len(recent) >= c["w_ref"]:
                Xr = recent[-c["w_ref"]:]
                self.det.set_reference(Xr, predict_proba(self.model, Xr), c["w_test"])
                self.need_ref, self.since_ref = False, 0
            return
        self.since_ref += len(X)
        # the test window must contain only messages that arrived after the reference
        if not c["adapt"] or self.since_ref < c["w_test"] or t <= self.cooldown:
            return
        t0 = time.time()
        W = recent[-c["w_test"]:]
        alarm, info = self.det.test(W, predict_proba(self.model, W))
        self.t_detect += time.time() - t0
        self.log.append((t, info["sw"], info["tau"], info["minp"], alarm))
        if alarm:
            self.purchase(t, int(np.ceil(c["label_win"] / nbatch)))

    def on_retrained(self, t):
        self.need_ref = True     # rebuild reference from the next w_ref post-adaptation samples
        self.buf_X = []


class ScoreKSWIN(LabelOnAlarm):
    """unsupervised KSWIN on the detector score; on alarm buys the same label budget, plain retrain."""
    name = "KSWIN-Score"

    def __init__(self, ctx):
        super().__init__(ctx, robust=False)
        self.det = rdrift.KSWIN(alpha=0.005, window_size=500, stat_size=150, seed=ctx["seed"])

    def step(self, t, X, nbatch):
        self.observe(t, X)
        if t <= self.cooldown:
            return
        hit = False
        for s in predict_proba(self.model, X):
            self.det.update(float(s))
            hit |= self.det.drift_detected
        if hit:
            self.purchase(t, int(np.ceil(self.cfg["label_win"] / nbatch)))

    def on_retrained(self, t):
        self.det = rdrift.KSWIN(alpha=0.005, window_size=500, stat_size=150, seed=self.seed + t)


class ScoreKSWINRobust(ScoreKSWIN):
    """KSWIN-Score alarms with the SLIDE update rule (replay, adversarial and proximal terms, warm start)."""
    name = "KSWIN-Score+SLIDE-update"

    def __init__(self, ctx):
        LabelOnAlarm.__init__(self, ctx, robust=True)
        self.det = rdrift.KSWIN(alpha=0.005, window_size=500, stat_size=150, seed=ctx["seed"])


class BudgetSchedule(LabelOnAlarm):
    """equal-budget controls: buy n_l labels at fixed batches, update with the SLIDE rule.
    mode='random': 4 purchase times drawn uniformly per realisation; mode='oracle': the true segment boundaries."""

    def __init__(self, ctx, mode="random", n_purchases=4):
        super().__init__(ctx, robust=True)
        self.name = "Budget-Random" if mode == "random" else "Budget-Oracle"
        T = int(np.ceil(len(ctx["seg"]) / ctx["nbatch"]))
        if mode == "random":
            rng = np.random.default_rng(1000 + ctx["seed"])
            self.when = set(int(v) for v in rng.choice(np.arange(2, T - ctx["D"] - 1), n_purchases, replace=False))
        else:
            seg = ctx["seg"]
            b = [i for i in range(1, len(seg)) if seg[i] != seg[i - 1]]
            self.when = set(int(i // ctx["nbatch"]) + 1 for i in b)    # the batch that completes 500 new-segment messages

    def step(self, t, X, nbatch):
        self.observe(t, X)
        if t in self.when:
            self.purchase(t, int(np.ceil(self.cfg["label_win"] / nbatch)))


def run_stream(method, order, seg, Xall, y, nbatch, D):
    """prequential loop; returns per-sample scores and label count."""
    T = int(np.ceil(len(order) / nbatch))
    scores = np.zeros(len(order))
    used = 0
    lat = 0.0
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
        # labels due at batch t arrive now, then the method may update
        for td in method.due_labels(t, D):
            s2 = slice(td * nbatch, min((td + 1) * nbatch, len(order)))
            method.receive(td, order[s2], Xall[order[s2]], y[order[s2]])
            used += s2.stop - s2.start
        if hasattr(method, "maybe_retrain"):
            method.maybe_retrain(t)
        # snapshot the scorer at the D->E boundary for the stress test
        if sl.stop < len(order) and seg[sl.stop - 1] == "D" and seg[sl.stop] == "E":
            snap["preE"] = method.snapshot()
    return scores, used, lat / len(order) * 1000.0, snap
