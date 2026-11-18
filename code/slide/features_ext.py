"""Window-level representations for the truncation fix (revision).

The encoder reads L=128 sub-word tokens. A message is cut into consecutive windows of 126 content tokens
(plus <s> and </s>); window 0 is exactly the representation phi(x) used everywhere else. We keep windows
0..3 and, for longer messages, the final window, so at most five windows per message. The chunked scorer
reports max_w f(phi_w(x)); training, drift testing and label purchase still use window 0 only.
"""
import time
import numpy as np
import torch
from transformers import AutoTokenizer, AutoModel
from .features import FeatureStore, FT_DIR, Cls, NT, h
from .config import ENCODER, MAX_LEN

MAXW = 5
CT = MAX_LEN - 2


def windows(ids):
    """content-token windows of one message (list of id lists), window 0 first."""
    w = [ids[i:i + CT] for i in range(0, max(len(ids), 1), CT)]
    if len(w) > MAXW:
        w = w[:MAXW - 1] + [ids[-CT:]]
    return w


@torch.no_grad()
def embed_windows(texts, bs=64):
    """returns dict text-hash -> float16 array (n_extra_windows x 768) for windows 1..; empty if one window."""
    torch.set_num_threads(NT)
    store = FeatureStore("phi_win")
    todo = store.missing(texts)
    if todo:
        tok = AutoTokenizer.from_pretrained(FT_DIR)
        model = Cls(AutoModel.from_pretrained(ENCODER))
        model.load_state_dict(torch.load(FT_DIR / "cls.pt"))
        model.eval()
        ids = tok(todo, add_special_tokens=False, truncation=True, max_length=4096)["input_ids"]
        flat = []                       # (text index, window) for windows 1..
        for i, x in enumerate(ids):
            for w in windows(x)[1:]:
                flat.append((i, [tok.bos_token_id] + w + [tok.eos_token_id]))
        out = {i: [] for i in range(len(todo))}
        t0 = time.time()
        flat.sort(key=lambda r: len(r[1]))
        for k in range(0, len(flat), bs):
            chunk = flat[k:k + bs]
            m = max(len(c[1]) for c in chunk)
            inp = torch.full((len(chunk), m), tok.pad_token_id)
            att = torch.zeros((len(chunk), m), dtype=torch.long)
            for r, (_, w) in enumerate(chunk):
                inp[r, :len(w)] = torch.tensor(w)
                att[r, :len(w)] = 1
            z = model.pool(inp, att).numpy().astype(np.float16)
            for (i, _), v in zip(chunk, z):
                out[i].append(v)
            if (k // bs) % 100 == 0:
                print(f"windows {k + len(chunk)}/{len(flat)} {time.time() - t0:.0f}s", flush=True)
        for i, t in enumerate(todo):
            store.d[h(t)] = np.stack(out[i]) if out[i] else np.zeros((0, 768), np.float16)
        store.save()
    return {h(t): store.d[h(t)] for t in texts}
