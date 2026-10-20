import os
from pathlib import Path

ROOT = Path(os.environ.get("SLIDE_ROOT", Path(__file__).resolve().parents[2]))
RAW = ROOT / "data_raw"
CACHE = Path(os.environ.get("SLIDE_CACHE", Path.home() / ".cache" / "slide"))
RESULTS = ROOT / "results"
for p in (CACHE, RESULTS):
    p.mkdir(parents=True, exist_ok=True)

POOL_SEED = 20261019
ENCODER = "distilroberta-base"
PPL_LM = "distilgpt2"
PARAPHRASER = "humarin/chatgpt_paraphraser_on_T5_base"
SIM_ENCODER = "sentence-transformers/all-MiniLM-L6-v2"
MAX_LEN = 128            # sub-word tokens seen by every neural component
PARA_CHARS = 600         # prefix length rewritten by the paraphrase attacker

# per-seed stream composition (sampled without replacement from disjoint pools)
STREAM_SEEDS = list(range(10))
BATCH = 250              # prequential batch size
LABEL_DELAY = 2          # labels of batch t become available after batch t+LABEL_DELAY
