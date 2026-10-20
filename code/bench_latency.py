"""Stage 4: per-component latency (ms/e-mail) and memory footprint on the test machine."""
import json
import time
import numpy as np
import pandas as pd
import torch
from transformers import AutoTokenizer, AutoModel, AutoModelForCausalLM
from slide.config import CACHE, RESULTS, ENCODER, PPL_LM, MAX_LEN
from slide.features import Cls, FT_DIR
from slide.models import Head, train_head, predict_proba, Reservoir
from slide.drift import SlicedDriftTest
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression

texts = pd.read_parquet(CACHE / "texts.parquet")
sample = texts.text.sample(512, random_state=0).tolist()
out = {}
for nt in (1, 4):
    torch.set_num_threads(nt)
    tok = AutoTokenizer.from_pretrained(FT_DIR)
    enc = Cls(AutoModel.from_pretrained(ENCODER)); enc.load_state_dict(torch.load(FT_DIR / "cls.pt")); enc.eval()
    lt = AutoTokenizer.from_pretrained(PPL_LM); lt.pad_token = lt.eos_token
    lm = AutoModelForCausalLM.from_pretrained(PPL_LM).eval()
    with torch.no_grad():
        for name, f in (("encoder", lambda c: enc.pool(**{k: v for k, v in zip(("ids", "att"), (lambda b: (b["input_ids"], b["attention_mask"]))(tok(c, truncation=True, max_length=MAX_LEN, padding=True, return_tensors="pt")))})),
                        ("surprisal_lm", lambda c: lm(**lt(c, truncation=True, max_length=MAX_LEN, padding=True, return_tensors="pt")))):
            f(sample[:32])
            t0 = time.time()
            for i in range(0, 256, 32):
                f(sample[i:i + 32])
            out[f"{name}_ms_{nt}t"] = (time.time() - t0) / 256 * 1000
torch.set_num_threads(1)
d = 772
X = np.random.default_rng(0).standard_normal((4000, d)).astype(np.float32)
y = (np.random.default_rng(1).random(4000) < 0.4).astype(int)
h = train_head(X[:500], y[:500], epochs=1)
t0 = time.time(); [predict_proba(h, X[i:i + 250]) for i in range(0, 4000, 250)]; out["head_ms"] = (time.time() - t0) / 4000 * 1000
det = SlicedDriftTest(d, K=64)
t0 = time.time(); det.set_reference(X[:1000], X[:1000, 0], 500); out["reference_build_s"] = time.time() - t0
t0 = time.time()
for _ in range(20):
    det.test(X[1000:1500], X[1000:1500, 0])
out["drift_test_ms_per_batch"] = (time.time() - t0) / 20 * 1000
out["drift_test_ms_per_email"] = out["drift_test_ms_per_batch"] / 250
mem = Reservoir(1000, d); mem.add(X[:2000], y[:2000])
for name, kw in (("retrain_plain_s", dict()), ("retrain_slide_s", dict(Xm=mem.data()[0], ym=mem.data()[1], lam_r=1.0, lam_a=0.5, eps=1.0, lam_p=1e-3))):
    t0 = time.time(); train_head(X[:500], y[:500], model=Head(d), **kw); out[name] = time.time() - t0
out["head_params"] = sum(p.numel() for p in Head(d).parameters())
out["encoder_params"] = sum(p.numel() for p in enc.parameters())
out["lm_params"] = sum(p.numel() for p in lm.parameters())
out["reservoir_MB"] = mem.nbytes() / 2 ** 20
out["reference_MB"] = (1000 * (64 + 1) * 8) / 2 ** 20
out["projection_MB"] = det.theta.nbytes / 2 ** 20
tv = TfidfVectorizer(ngram_range=(1, 2), min_df=2, max_features=100000, sublinear_tf=True)
W = texts[texts.pool == "WARM"]
lr = LogisticRegression(C=10, max_iter=2000).fit(tv.fit_transform(W.text), W.label)
t0 = time.time(); lr.predict_proba(tv.transform(sample)); out["tfidf_lr_ms"] = (time.time() - t0) / len(sample) * 1000
out["cpu"] = open("/proc/cpuinfo").read().split("model name")[1].split("\n")[0].strip(": ")
json.dump(out, open(RESULTS / "latency.json", "w"), indent=1)
print(json.dumps(out, indent=1))
