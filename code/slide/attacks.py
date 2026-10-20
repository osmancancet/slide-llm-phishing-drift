"""Evasion transformations applied to phishing texts (threat model T1-T3 in the paper).

T1 paraphrase : neural paraphraser rewrites every sentence of the leading PARA_CHARS characters that does
                not carry the payload (URLs / e-mail addresses are kept verbatim); accepted if the
                MiniLM cosine similarity to the original is >= TAU_SIM, otherwise re-sampled (<=3 tries).
T2 character  : a fraction eps of eligible words receives a homoglyph swap, adjacent transposition
                or zero-width insertion.
T3 padding    : a benign human e-mail fragment is prepended ("good-word" / truncation attack).
"""
import re
import difflib
import numpy as np
import pandas as pd
import torch
from transformers import AutoTokenizer, AutoModel, AutoModelForSeq2SeqLM
from .config import CACHE, PARAPHRASER, SIM_ENCODER, PARA_CHARS
from .data import build_pools

TAU_SIM = 0.70
SENT = re.compile(r"(?<=[.!?])\s+")
PAYLOAD = re.compile(r"(https?://|www\.|\S+@\S+\.\S+)", re.I)
HOMO = {"a": "а", "e": "е", "o": "о", "p": "р", "c": "с", "x": "х", "i": "і"}


class Sim:
    def __init__(self):
        self.tok = AutoTokenizer.from_pretrained(SIM_ENCODER)
        self.m = AutoModel.from_pretrained(SIM_ENCODER).eval()

    @torch.no_grad()
    def emb(self, texts):
        b = self.tok(texts, truncation=True, max_length=256, padding=True, return_tensors="pt")
        h = self.m(**b).last_hidden_state
        m = b["attention_mask"].unsqueeze(-1).float()
        e = (h * m).sum(1) / m.sum(1)
        return torch.nn.functional.normalize(e, dim=-1)

    def cos(self, a, b):
        return (self.emb(a) * self.emb(b)).sum(-1).numpy()


def word_edit(a, b):
    """normalised word-level edit ratio in [0,1] (1 - difflib similarity)."""
    return 1.0 - difflib.SequenceMatcher(None, a.split(), b.split()).ratio()


def split_prefix(text):
    if len(text) <= PARA_CHARS:
        return text, ""
    cut = text.rfind(" ", 0, PARA_CHARS)
    cut = cut if cut > 0 else PARA_CHARS
    return text[:cut], text[cut:]


class Paraphraser:
    def __init__(self):
        self.tok = AutoTokenizer.from_pretrained(PARAPHRASER)
        self.m = AutoModelForSeq2SeqLM.from_pretrained(PARAPHRASER).eval()

    @torch.no_grad()
    def __call__(self, sents, seed):
        torch.manual_seed(seed)
        b = self.tok(["paraphrase: " + s for s in sents], truncation=True, max_length=128,
                     padding=True, return_tensors="pt")
        o = self.m.generate(**b, max_length=128, do_sample=True, top_p=0.9, temperature=0.9,
                            num_beams=1, repetition_penalty=1.2)
        return [self.tok.decode(x, skip_special_tokens=True) for x in o]


def paraphrase_text(text, para, sim, seed):
    head, tail = split_prefix(text)
    sents = [s for s in SENT.split(head) if s]
    rew_idx = [i for i, s in enumerate(sents) if not PAYLOAD.search(s) and len(s.split()) >= 3]
    best, best_sim = text, -1.0
    for k in range(3):
        out = list(sents)
        if rew_idx:
            for i, r in zip(rew_idx, para([sents[i] for i in rew_idx], seed * 7 + k)):
                out[i] = r
        cand = (" ".join(out) + tail).strip()
        s = float(sim.cos([head], [" ".join(out)])[0])
        if s > best_sim:
            best, best_sim = cand, s
        if s >= TAU_SIM:
            break
    return best, best_sim


def char_attack(text, eps, rng):
    words = text.split(" ")
    elig = [i for i, w in enumerate(words) if len(w) >= 4 and not PAYLOAD.search(w)]
    n = int(round(eps * len(elig)))
    for i in rng.choice(elig, size=min(n, len(elig)), replace=False) if elig else []:
        w = words[i]
        op = rng.integers(3)
        if op == 0:
            pos = [j for j, ch in enumerate(w) if ch.lower() in HOMO]
            if pos:
                j = int(rng.choice(pos))
                w = w[:j] + HOMO[w[j].lower()] + w[j + 1:]
        elif op == 1:
            j = int(rng.integers(1, len(w) - 2))
            w = w[:j] + w[j + 1] + w[j] + w[j + 2:]
        else:
            j = int(rng.integers(1, len(w) - 1))
            w = w[:j] + "​" + w[j:]
        words[i] = w
    return " ".join(words)


def paraphrase_many(texts, uids, para, sim, tries=3):
    """batched version of paraphrase_text: sentences of many e-mails share one generate() call."""
    heads, tails, sents, ridx = [], [], [], []
    for t in texts:
        hd, tl = split_prefix(t)
        ss = [x for x in SENT.split(hd) if x]
        heads.append(hd); tails.append(tl); sents.append(ss)
        ridx.append([i for i, x in enumerate(ss) if not PAYLOAD.search(x) and len(x.split()) >= 3])
    best = list(texts)
    best_sim = [-1.0] * len(texts)
    todo = list(range(len(texts)))
    for k in range(tries):
        flat = [(j, i) for j in todo for i in ridx[j]]
        outs = para([sents[j][i] for j, i in flat], int(uids[todo[0]]) * 7 + k) if flat else []
        new = {j: list(sents[j]) for j in todo}
        for (j, i), r in zip(flat, outs):
            new[j][i] = r
        joined = [" ".join(new[j]) for j in todo]
        sims = sim.cos([heads[j] for j in todo], joined)
        nxt = []
        for j, jn, sv in zip(todo, joined, sims):
            if sv > best_sim[j]:
                best[j], best_sim[j] = (jn + tails[j]).strip(), float(sv)
            if sv < TAU_SIM:
                nxt.append(j)
        todo = nxt
        if not todo:
            break
    return best, best_sim


def run_paraphrase(limit=None, chunk=12):
    out = CACHE / "paraphrased.parquet"
    pools = build_pools()
    sel = pools[pools.pool.isin(["E_para", "STRESS"])].copy()
    if limit:
        sel = sel.head(limit)
    done = pd.read_parquet(out) if out.exists() else pd.DataFrame(columns=["uid", "para", "sim", "edit"])
    todo = sel[~sel.uid.isin(done.uid)]
    # random processing order so that any prefix of the job is a random subset of each pool
    todo = todo.sample(frac=1.0, random_state=11)
    torch.set_num_threads(int(__import__("os").environ.get("NT", 4)))
    para, sim = Paraphraser(), Sim()
    U, T = todo.uid.values, todo.text.values
    for i in range(0, len(U), chunk):
        ps, ss = paraphrase_many(list(T[i:i + chunk]), U[i:i + chunk], para, sim)
        rows = [(int(u), p, s, word_edit(split_prefix(t)[0], split_prefix(p)[0]))
                for u, t, p, s in zip(U[i:i + chunk], T[i:i + chunk], ps, ss)]
        done = pd.concat([done, pd.DataFrame(rows, columns=done.columns)], ignore_index=True)
        done.to_parquet(out)
        print(f"paraphrased {len(done)}/{len(sel)}", flush=True)
    return done


def build_attacked_texts():
    """Final text table: original texts with E_para replaced by paraphrases, plus stress variants."""
    pools = build_pools()
    para = pd.read_parquet(CACHE / "paraphrased.parquet").set_index("uid")
    # the attack budget is binding: a rewrite below TAU_SIM after all tries is rejected and the original is sent
    orig_text = pools.set_index("uid").text
    failed = para.sim < TAU_SIM
    para.loc[failed, "para"] = orig_text.loc[para.index[failed]].values
    texts = pools[["uid", "text", "label", "pool", "source", "gen"]].copy()
    texts["variant"] = "orig"
    # rows of the attack pools that the paraphraser did not reach are excluded from every experiment
    for pl in ("E_para", "STRESS"):
        miss = (texts.pool == pl) & ~texts.uid.isin(para.index)
        texts.loc[miss, "pool"] = pl + "_unused"
    m = texts.pool == "E_para"
    texts.loc[m, "text"] = para.loc[texts.loc[m, "uid"], "para"].values
    texts.loc[m, "variant"] = np.where(failed.loc[texts.loc[m, "uid"]].values, "T1_rejected", "T1_para")
    rng = np.random.default_rng(7)
    st = pools[(pools.pool == "STRESS") & pools.uid.isin(para.index)]
    ham = pools[(pools.pool == "E") & (pools.label == 0)].text.values
    extra = []
    for uid, t in zip(st.uid, st.text):
        extra.append((uid, para.loc[uid, "para"], "T1_para"))
        for eps in (0.05, 0.10, 0.20):
            extra.append((uid, char_attack(t, eps, rng), f"T2_char_{eps:.2f}"))
        pad = ham[rng.integers(len(ham))][:400]
        extra.append((uid, pad + " " + t, "T3_pad"))
        extra.append((uid, char_attack(para.loc[uid, "para"], 0.10, rng), "T1+T2"))
    ex = pd.DataFrame(extra, columns=["uid", "text", "variant"]).merge(
        st[["uid", "label", "pool", "source", "gen"]], on="uid")
    ex["pool"] = "STRESS_ATT"
    allt = pd.concat([texts, ex], ignore_index=True)
    allt["row"] = np.arange(len(allt))
    allt.to_parquet(CACHE / "texts.parquet")
    return allt


if __name__ == "__main__":
    run_paraphrase()
