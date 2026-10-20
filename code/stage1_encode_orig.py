"""Stage 1: fine-tune encoder on WARM, then encode every non-paraphrased pool text."""
from slide.data import build_pools
from slide.features import finetune, embed, surprisal

pools = build_pools()
warm = pools[pools.pool == "WARM"]
finetune(warm.text.values, warm.label.values)
rest = pools[pools.pool != "E_para"].text.tolist()
embed(rest)
surprisal(rest)
print("stage1 done")
