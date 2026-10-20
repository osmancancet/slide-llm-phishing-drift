"""Detection head h_theta and the composite SLIDE objective.

L_total(theta) = L_cls(S) + lam_r * L_rep(M) + lam_a * L_adv(S u M; eps) + lam_p * ||theta - theta_prev||^2
"""
import copy
import numpy as np
import torch
from torch import nn


class Head(nn.Module):
    def __init__(self, d, hdim=256, p=0.1):
        super().__init__()
        self.net = nn.Sequential(nn.Linear(d, hdim), nn.GELU(), nn.Dropout(p), nn.Linear(hdim, 2))

    def forward(self, x):
        return self.net(x)


def class_weights(y):
    c = np.bincount(y, minlength=2).astype(float)
    c[c == 0] = 1.0
    return torch.tensor(len(y) / (2.0 * c), dtype=torch.float)


def pgd_l2(model, x, y, eps, steps=3, w=None):
    """worst-case l2 perturbation of the (standardised) representation, ||delta||_2 <= eps."""
    if eps <= 0:
        return x
    delta = torch.zeros_like(x, requires_grad=True)
    alpha = 2.5 * eps / steps
    for _ in range(steps):
        loss = nn.functional.cross_entropy(model(x + delta), y, weight=w)
        g, = torch.autograd.grad(loss, delta)
        g = g / (g.norm(dim=1, keepdim=True) + 1e-12)
        delta = delta.detach() + alpha * g
        n = delta.norm(dim=1, keepdim=True)
        delta = (delta * torch.clamp(eps / (n + 1e-12), max=1.0)).requires_grad_(True)
    return (x + delta).detach()


def train_head(X, y, model=None, Xm=None, ym=None, lam_r=1.0, lam_a=0.0, eps=0.0, lam_p=0.0,
               epochs=30, bs=256, lr=1e-3, seed=0, pgd_steps=3):
    """Train (or continue training) the head on the labelled window S=(X,y) and replay M=(Xm,ym)."""
    torch.manual_seed(seed)
    d = X.shape[1]
    if model is None:
        model = Head(d)
    anchor = [p.detach().clone() for p in model.parameters()] if lam_p > 0 else None
    Xs, ys = torch.tensor(X, dtype=torch.float), torch.tensor(y, dtype=torch.long)
    use_m = Xm is not None and len(Xm) > 0 and lam_r > 0
    if use_m:
        Xr, yr = torch.tensor(Xm, dtype=torch.float), torch.tensor(ym, dtype=torch.long)
    w = class_weights(np.concatenate([y, ym]) if use_m else y)
    opt = torch.optim.AdamW(model.parameters(), lr=lr, weight_decay=1e-4)
    g = torch.Generator().manual_seed(seed)
    model.train()
    for _ in range(epochs):
        perm = torch.randperm(len(ys), generator=g)
        for i in range(0, len(ys), bs):
            b = perm[i:i + bs]
            xb, yb = Xs[b], ys[b]
            loss = nn.functional.cross_entropy(model(xb), yb, weight=w)
            if use_m:
                rb = torch.randint(len(yr), (len(b),), generator=g)
                xr, yrr = Xr[rb], yr[rb]
                loss = loss + lam_r * nn.functional.cross_entropy(model(xr), yrr, weight=w)
                xa, ya = torch.cat([xb, xr]), torch.cat([yb, yrr])
            else:
                xa, ya = xb, yb
            if lam_a > 0:
                xadv = pgd_l2(model, xa, ya, eps, pgd_steps, w)
                loss = loss + lam_a * nn.functional.cross_entropy(model(xadv), ya, weight=w)
            if anchor is not None:
                loss = loss + lam_p * sum(((p - a) ** 2).sum() for p, a in zip(model.parameters(), anchor))
            opt.zero_grad()
            loss.backward()
            opt.step()
    model.eval()
    return model


@torch.no_grad()
def predict_proba(model, X):
    model.eval()
    return torch.softmax(model(torch.tensor(X, dtype=torch.float)), -1)[:, 1].numpy()


def lipschitz_margin_bound(model):
    """Upper bound on the l2-Lipschitz constant of the logit margin f_1 - f_0 (GELU is 1.13-Lipschitz)."""
    W1 = model.net[0].weight.detach()
    W2 = model.net[3].weight.detach()
    v = (W2[1] - W2[0]).unsqueeze(0)
    return float(torch.linalg.matrix_norm(v, 2) * 1.1289 * torch.linalg.matrix_norm(W1, 2))


@torch.no_grad()
def certified_fraction(model, X, y, eps):
    """fraction of correctly classified points whose margin exceeds L*eps (provably invariant in the eps-ball)."""
    L = lipschitz_margin_bound(model)
    f = model(torch.tensor(X, dtype=torch.float))
    m = (f[:, 1] - f[:, 0]).numpy() * np.where(y == 1, 1, -1)
    return float(np.mean(m > L * eps)), L


class Reservoir:
    """class-balanced reservoir memory of capacity cap per class."""

    def __init__(self, cap, d, seed=0):
        self.cap, self.rng = cap, np.random.default_rng(seed)
        self.X = {0: np.zeros((0, d), np.float32), 1: np.zeros((0, d), np.float32)}
        self.n = {0: 0, 1: 0}

    def add(self, X, y):
        for c in (0, 1):
            for x in X[y == c]:
                self.n[c] += 1
                if len(self.X[c]) < self.cap:
                    self.X[c] = np.vstack([self.X[c], x[None]])
                else:
                    j = self.rng.integers(self.n[c])
                    if j < self.cap:
                        self.X[c][j] = x

    def data(self):
        X = np.vstack([self.X[0], self.X[1]])
        y = np.concatenate([np.zeros(len(self.X[0]), int), np.ones(len(self.X[1]), int)])
        return X, y

    def nbytes(self):
        return self.X[0].nbytes + self.X[1].nbytes


def clone(m):
    return copy.deepcopy(m)
