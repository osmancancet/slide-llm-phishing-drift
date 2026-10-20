"""Stage 2: assemble attacked texts (after paraphrasing) and encode everything not yet cached."""
from slide.attacks import build_attacked_texts
from slide.features import embed, surprisal

t = build_attacked_texts()
txt = t.text.tolist()
embed(txt)
surprisal(txt)
print("stage2 done", len(t))
