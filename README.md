# SLIDE: label-free drift detection and budgeted adaptation for LLM-generated phishing

Code for "SLIDE: label-free sliced-Wasserstein drift detection and budgeted adaptation for LLM-generated phishing in e-mail streams" (O. C. Çetlenbik, submitted to *Knowledge-Based Systems*). Every number, table and figure in the manuscript is written by these scripts; nothing is typed by hand.

## Requirements
CPU only (tested on a 4-core Intel Xeon 2.10 GHz, 15 GB RAM). Python 3 with the pinned versions in `requirements.txt` (CPU build of PyTorch). Hugging Face models are downloaded on first use: `distilroberta-base`, `distilgpt2`, `humarin/chatgpt_paraphraser_on_T5_base`, `sentence-transformers/all-MiniLM-L6-v2` (loaded through `transformers`).

## Data (all public, fetched by `download_data.py`)
| Source | Hugging Face id | Use |
|---|---|---|
| Seven human corpora (Enron, Ling-Spam, SpamAssassin, TREC-05/06/07, CEAS-08), 203,017 rows | `puyang2025/seven-phishing-email-datasets` | warm-up, segments A–E |
| GPTGen, 3,746 English LLM phishing/legitimate e-mails | `Dizzzy0x00/LLMGen-Phishing-Email-Dataset` | segments D, E, stress |
| SpearGen, 2,798 OpenAI spear-phishing e-mails | `kxm1k4m1/generate_phishing_email_final` | segments D, E, stress |

## Run
```bash
pip install -r code/requirements.txt
bash code/run_all.sh
```

The repository already contains every result file the paper reads (`results/`), so `python3 code/make_tables.py && python3 code/make_extra_tables.py` regenerate all tables and figures in seconds without rerunning the experiments. Reference checks are in `refs_verification.md`.
`SLIDE_ROOT` (default: parent folder) and `SLIDE_CACHE` (default `~/.cache/slide`) control where data, features and per-job results are stored. Stages are resumable: the paraphrase attack and every stream job are cached, so an interrupted run continues where it stopped.

| Stage | Script | Writes | Paper |
|---|---|---|---|
| 1 | `download_data.py` | `data_raw/` | Sec. 5.1 |
| 2 | `stage1_encode_orig.py` | fine-tuned encoder, φ and ψ features of original texts | Sec. 3.1 |
| 3 | `python -m slide.attacks` | T1 paraphrases (`paraphrased.parquet`) | Sec. 3.3 |
| 4 | `stage2_attacks_encode.py` | T1–T3 attacked texts and features | Sec. 6.5 |
| 5 | `audit_leakage.py` | `results/leakage_audit.*` | Sec. 6.1, Table 5 |
| 6 | `run_experiments.py --shard i/4`, then `--collect` | `results/raw/{summary,batch,stress}.csv`, `events.json` | Sec. 6.2–6.7 |
| 7 | `cert_and_displacement.py` | `results/cert_displacement.json` | Table 11 |
| 8 | `bench_latency.py` | `results/latency.json` | Table 14 |
| 9 | `make_pool_table.py`, `make_audit_table.py`, `make_tables.py`, `make_extra_tables.py` | `manuscript/tables/*.tex`, `manuscript/figures/*`, `results/numbers.json` | all tables and figures |

Runtime on the test machine: encoder fine-tuning about 25 min; T1 paraphrasing of 2,400 e-mails is the slowest stage on CPU (resumable); 240 stream jobs at 15–30 s each with four shards.

## Package layout (`slide/`)
- `config.py`: paths, seeds (pool 20261019, streams 0–9), model ids, stream constants.
- `data.py`: corpus loading, disjoint pools, five-segment stream composition.
- `attacks.py`: T1 paraphrase with MiniLM acceptance (cos ≥ 0.70, 3 tries), T2 character perturbation, T3 padding.
- `features.py`: encoder fine-tuning, mean-pooled φ, surprisal statistics ψ, content-hash cache.
- `models.py`: head, PGD in representation space, `train_head` for L_total, Lipschitz certificate, class-balanced reservoir.
- `drift.py`: `SlicedDriftTest` (Holm-KS over K+1 projections, calibrated sliced-W1 gate).
- `stream.py`: all ten methods and the prequential runner with delayed, purchasable labels.
- `metrics.py`, `stats.py`: metrics (incl. MCC, PR-AUC, Brier), Holm, Wilcoxon with rank-biserial r, Friedman/Nemenyi.

## Manuscript build
```bash
cd manuscript
tectonic --keep-intermediates --outdir build main.tex   # PDF; build/main.aux is used below
python3 build_docx_md.py ieee.csl                       # main.docx and main.md (pandoc + IEEE CSL)
```
