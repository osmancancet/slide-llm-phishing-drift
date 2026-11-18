"""Revision text table: original texts (texts.parquet) + new pools + extended stress attacks.

Extended attacks on the STRESS pool (600 held-out phishing):
  T3_pad_200 / T3_pad (400) / T3_pad_800 : benign legitimate prefix of 200 / 400 / 800 characters
  T4_suffix                              : 400-character benign suffix (lure stays in window 0)
  T5_sandwich                            : 400-character prefix and 400-character suffix
STRESS2 (200 PhishFuzzer + 200 Spanish phishing, never streamed): orig, T2_char_0.10, T3_pad (400).
"""
import numpy as np
import pandas as pd
from .config import CACHE
from .attacks import char_attack
from .data_ext import build_pools_ext


def pad_text(ham, rng, n):
    """a benign fragment of at least n characters, cut at a word boundary."""
    while True:
        t = ham[rng.integers(len(ham))]
        if len(t) >= n:
            cut = t.rfind(" ", 0, n)
            return t[:cut if cut > n // 2 else n]


def build_texts_v2():
    out = CACHE / "texts_v2.parquet"
    if out.exists():
        return pd.read_parquet(out)
    base = pd.read_parquet(CACHE / "texts.parquet")
    ext = build_pools_ext()
    e = ext[["uid", "text", "label", "pool", "source", "gen"]].copy()
    e["variant"] = "orig"
    rng = np.random.default_rng(8)
    ham = base[(base.pool == "E") & (base.label == 0)].text.values
    st = base[(base.pool == "STRESS") & (base.variant == "orig")]
    extra = []
    for uid, t in zip(st.uid, st.text):
        extra.append((uid, pad_text(ham, rng, 200) + " " + t, "T3_pad_200"))
        extra.append((uid, pad_text(ham, rng, 800) + " " + t, "T3_pad_800"))
        extra.append((uid, t + " " + pad_text(ham, rng, 400), "T4_suffix"))
        extra.append((uid, pad_text(ham, rng, 400) + " " + t + " " + pad_text(ham, rng, 400), "T5_sandwich"))
    ex = pd.DataFrame(extra, columns=["uid", "text", "variant"]).merge(
        st[["uid", "label", "source", "gen"]], on="uid")
    ex["pool"] = "STRESS_ATT"
    s2 = ext[ext.pool == "STRESS2"]
    extra2 = []
    for uid, t in zip(s2.uid, s2.text):
        extra2.append((uid, char_attack(t, 0.10, rng), "T2_char_0.10"))
        extra2.append((uid, pad_text(ham, rng, 400) + " " + t, "T3_pad"))
    ex2 = pd.DataFrame(extra2, columns=["uid", "text", "variant"]).merge(
        s2[["uid", "label", "source", "gen"]], on="uid")
    ex2["pool"] = "STRESS2_ATT"
    allt = pd.concat([base.drop(columns=["row"]), e, ex, ex2], ignore_index=True)
    allt["row"] = np.arange(len(allt))
    allt.to_parquet(out)
    return allt
