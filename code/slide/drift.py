"""Label-free drift test on the representation stream.

For reference R (n_r x d) and window W (n_w x d), projections u_k = <theta_k, x>, k=1..K, theta_k ~ U(S^{d-1}),
plus the detector score s(x) as projection K+1.
  KS gate : Holm-Bonferroni over the K+1 two-sample KS p-values at family-wise level alpha
  SW gate : SW_1(R,W) = (1/K) sum_k W_1(theta_k#R, theta_k#W) > tau_alpha, where tau_alpha is the
            (1-alpha) quantile of SW_1 between B random disjoint half-splits of R (null calibration)
Alarm = KS gate AND SW gate (each gate can be disabled for ablation).
"""
import numpy as np
from scipy.stats import ks_2samp, wasserstein_distance


def w1_sorted(a, b):
    return wasserstein_distance(a, b)


class SlicedDriftTest:
    def __init__(self, d, K=64, alpha=0.01, B=100, use_ks=True, use_sw=True, use_proj=True,
                 use_score=True, seed=0):
        rng = np.random.default_rng(seed)
        th = rng.standard_normal((d, K))
        self.theta = th / np.linalg.norm(th, axis=0, keepdims=True)
        self.K, self.alpha, self.B = K, alpha, B
        self.use_ks, self.use_sw, self.use_proj, self.use_score = use_ks, use_sw, use_proj, use_score
        self.rng = rng

    def _proj(self, X, s):
        cols = []
        if self.use_proj:
            cols.append(X @ self.theta)
        if self.use_score:
            cols.append(s[:, None])
        return np.hstack(cols)

    def _sw(self, P, Q):
        k = self.K if self.use_proj else 1
        return float(np.mean([w1_sorted(P[:, j], Q[:, j]) for j in range(k)]))

    def set_reference(self, X, s, n_w):
        self.P = self._proj(X, s)
        n = len(self.P)
        null = []
        for _ in range(self.B):
            perm = self.rng.permutation(n)
            m = min(n_w, n // 2)
            null.append(self._sw(self.P[perm[:m]], self.P[perm[m:2 * m]]))
        self.tau = float(np.quantile(null, 1 - self.alpha))

    def test(self, X, s):
        Q = self._proj(X, s)
        pv = np.array([ks_2samp(self.P[:, j], Q[:, j], method="asymp").pvalue for j in range(Q.shape[1])])
        # Holm step-down: reject H_(i) while p_(i) <= alpha / (m - i + 1)
        # Holm step-down rejects at least one hypothesis iff min_j p_j <= alpha / m
        order = np.sort(pv)
        m = len(order)
        ks_rej = bool(order[0] <= self.alpha / m)
        sw = self._sw(self.P, Q)
        sw_rej = sw > self.tau
        alarm = (ks_rej or not self.use_ks) and (sw_rej or not self.use_sw)
        return alarm, {"sw": sw, "tau": self.tau, "minp": float(order[0]), "ks": ks_rej, "swr": bool(sw_rej)}
