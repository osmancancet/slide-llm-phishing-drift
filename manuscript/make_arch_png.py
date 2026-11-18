"""Matplotlib rendering of the TikZ architecture figure (Fig. 1) for the Word build."""
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.patches import FancyBboxPatch

fig, ax = plt.subplots(figsize=(9, 3.1), dpi=300)
ax.set_xlim(0, 9); ax.set_ylim(0, 3.1); ax.axis("off")
B = {"x": (0.55, 2.4, "e-mail $x$"), "enc": (2.25, 2.4, "frozen encoder\n$z=\\Phi(x)\\in\\mathbb{R}^{768}$"),
     "h": (4.6, 2.4, "learner\n(head or TF-IDF)"), "s": (6.6, 2.4, "score\n$f_t(x)$"),
     "t": (2.25, 0.7, "sliced KS + $SW_1$ test\nlevel $\\alpha_j$"), "p": (5.1, 0.7, "buy labels of\nalarm window"),
     "u": (7.6, 0.7, "update learner,\nrebuild reference")}
W = {"x": 0.9, "enc": 1.9, "h": 1.5, "s": 0.9, "t": 1.9, "p": 1.5, "u": 1.6}
for k, (x, y, s) in B.items():
    w = W[k]
    ax.add_patch(FancyBboxPatch((x - w / 2, y - 0.38), w, 0.76, boxstyle="round,pad=0.02,rounding_size=0.06", fc="white", ec="black", lw=0.9))
    ax.text(x, y, s, ha="center", va="center", fontsize=8.5)
def arr(a, b, dashed=False, label=None, lx=0, ly=0.1):
    ax.annotate("", xy=b, xytext=a, arrowprops=dict(arrowstyle="-|>", lw=0.9, ls="--" if dashed else "-", color="black", shrinkA=0, shrinkB=0))
    if label:
        ax.text((a[0] + b[0]) / 2 + lx, (a[1] + b[1]) / 2 + ly, label, fontsize=7, ha="center", va="bottom")
arr((1.0, 2.4), (1.3, 2.4)); arr((3.2, 2.4), (3.85, 2.4)); arr((5.35, 2.4), (6.15, 2.4))
arr((2.25, 2.02), (2.25, 1.08), label="projections", lx=-0.45, ly=0)
arr((4.0, 2.02), (3.0, 1.08), label="score", lx=0.25, ly=0)
arr((3.2, 0.7), (4.35, 0.7), dashed=True, label="alarm")
arr((5.85, 0.7), (6.8, 0.7), dashed=True, label="$+D_t$")
arr((7.6, 1.08), (6.75, 2.02), dashed=True)
fig.tight_layout(pad=0.1)
fig.savefig("figures/arch.png", dpi=300)
