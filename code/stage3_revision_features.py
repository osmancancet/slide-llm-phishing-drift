"""Revision stage: attacked texts, extended text table and every feature the revision reads."""
from slide.attacks import build_attacked_texts
from slide.attacks_ext import build_texts_v2
from slide.features import embed, surprisal
from slide.features_ext import embed_windows

build_attacked_texts()
t = build_texts_v2()
txt = t.text.tolist()
embed(txt)
embed_windows(txt)
surprisal(txt)
print("stage3 done", len(t))
