"""Transformer encoder fine-tuning (warm-up window only) and cached feature extraction.

phi(x)  : 768-d mean-pooled last hidden state of the fine-tuned DistilRoBERTa encoder
psi(x)  : 4 token-surprisal statistics of x under a frozen causal LM (distilgpt2):
          mean, std, 10th and 90th percentile of -log p(x_t | x_<t)
"""
import os
import hashlib
import pickle
import time
import numpy as np
import torch
from torch import nn
from transformers import AutoTokenizer, AutoModel, AutoModelForCausalLM
from .config import CACHE, ENCODER, PPL_LM, MAX_LEN

NT = int(os.environ.get("NT", 4))   # threads for encoding jobs (set inside the encoding functions only)
FT_DIR = CACHE / "encoder_ft"


def h(t):
    return hashlib.sha1(t.encode("utf-8", "ignore")).hexdigest()


class Cls(nn.Module):
    def __init__(self, enc):
        super().__init__()
        self.enc = enc
        self.out = nn.Linear(enc.config.hidden_size, 2)

    def pool(self, ids, att):
        hdn = self.enc(input_ids=ids, attention_mask=att).last_hidden_state
        m = att.unsqueeze(-1).float()
        return (hdn * m).sum(1) / m.sum(1)

    def forward(self, ids, att):
        return self.out(self.pool(ids, att))


def finetune(texts, labels, epochs=2, lr=2e-5, bs=16, seed=0):
    """Fine-tune on the warm-up window only (no stream data is ever seen here)."""
    torch.set_num_threads(NT)
    if (FT_DIR / "cls.pt").exists():
        return
    torch.manual_seed(seed)
    np.random.seed(seed)
    tok = AutoTokenizer.from_pretrained(ENCODER)
    model = Cls(AutoModel.from_pretrained(ENCODER))
    enc = tok(list(texts), truncation=True, max_length=MAX_LEN, padding="max_length", return_tensors="pt")
    y = torch.tensor(np.asarray(labels))
    w = torch.tensor(len(y) / (2.0 * np.bincount(y.numpy(), minlength=2)), dtype=torch.float)
    opt = torch.optim.AdamW(model.parameters(), lr=lr, weight_decay=0.01)
    steps = epochs * int(np.ceil(len(y) / bs))
    sch = torch.optim.lr_scheduler.LambdaLR(opt, lambda s: min(1.0, (s + 1) / (0.06 * steps)) * max(0.0, (steps - s) / steps))
    lossf = nn.CrossEntropyLoss(weight=w)
    model.train()
    t0, k = time.time(), 0
    for ep in range(epochs):
        perm = torch.randperm(len(y))
        for i in range(0, len(y), bs):
            b = perm[i:i + bs]
            loss = lossf(model(enc["input_ids"][b], enc["attention_mask"][b]), y[b])
            opt.zero_grad()
            loss.backward()
            nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            opt.step()
            sch.step()
            k += 1
            if k % 50 == 0:
                print(f"ep{ep} step{k}/{steps} loss={loss.item():.4f} {time.time()-t0:.0f}s", flush=True)
    FT_DIR.mkdir(exist_ok=True, parents=True)
    torch.save(model.state_dict(), FT_DIR / "cls.pt")
    tok.save_pretrained(FT_DIR)
    with open(FT_DIR / "train_time.txt", "w") as f:
        f.write(f"{time.time()-t0:.1f}\n")


class FeatureStore:
    """Content-addressed cache so every text is encoded exactly once."""

    def __init__(self, name):
        self.path = CACHE / f"{name}.pkl"
        self.d = pickle.load(open(self.path, "rb")) if self.path.exists() else {}

    def save(self):
        tmp = str(self.path) + ".tmp"
        pickle.dump(self.d, open(tmp, "wb"))
        os.replace(tmp, self.path)

    def missing(self, texts):
        seen, out = set(), []
        for t in texts:
            k = h(t)
            if k not in self.d and k not in seen:
                seen.add(k)
                out.append(t)
        return out

    def get(self, texts):
        return np.stack([self.d[h(t)] for t in texts])


@torch.no_grad()
def embed(texts, bs=64):
    torch.set_num_threads(NT)
    store = FeatureStore("phi")
    todo = store.missing(texts)
    if todo:
        tok = AutoTokenizer.from_pretrained(FT_DIR)
        model = Cls(AutoModel.from_pretrained(ENCODER))
        model.load_state_dict(torch.load(FT_DIR / "cls.pt"))
        model.eval()
        t0 = time.time()
        for i in range(0, len(todo), bs):
            chunk = todo[i:i + bs]
            b = tok(chunk, truncation=True, max_length=MAX_LEN, padding=True, return_tensors="pt")
            z = model.pool(b["input_ids"], b["attention_mask"])
            p = torch.softmax(model.out(z), -1)[:, 1]
            for t, v, q in zip(chunk, z.numpy().astype(np.float32), p.numpy()):
                store.d[h(t)] = np.concatenate([v, [q]]).astype(np.float32)
            if (i // bs) % 40 == 0:
                print(f"embed {i+len(chunk)}/{len(todo)} {time.time()-t0:.0f}s", flush=True)
                store.save()
        store.save()
    a = store.get(texts)
    return a[:, :-1], a[:, -1]          # (phi, p_ft)


@torch.no_grad()
def surprisal(texts, bs=32):
    torch.set_num_threads(NT)
    store = FeatureStore("psi")
    todo = store.missing(texts)
    if todo:
        tok = AutoTokenizer.from_pretrained(PPL_LM)
        tok.pad_token = tok.eos_token
        lm = AutoModelForCausalLM.from_pretrained(PPL_LM).eval()
        t0 = time.time()
        for i in range(0, len(todo), bs):
            chunk = todo[i:i + bs]
            b = tok(chunk, truncation=True, max_length=MAX_LEN, padding=True, return_tensors="pt")
            logits = lm(**b).logits[:, :-1]
            tgt = b["input_ids"][:, 1:]
            m = b["attention_mask"][:, 1:].bool()
            nll = torch.nn.functional.cross_entropy(logits.transpose(1, 2), tgt, reduction="none")
            for t, row, mm in zip(chunk, nll, m):
                v = row[mm].numpy()
                if v.size == 0:
                    v = np.zeros(1)
                store.d[h(t)] = np.array([v.mean(), v.std(), np.quantile(v, 0.1), np.quantile(v, 0.9)], np.float32)
            if (i // bs) % 80 == 0:
                print(f"surprisal {i+len(chunk)}/{len(todo)} {time.time()-t0:.0f}s", flush=True)
                store.save()
        store.save()
    return store.get(texts)
