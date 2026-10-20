#!/usr/bin/env bash
# Full reproduction of every number, table and figure in the manuscript (CPU only).
set -euo pipefail
cd "$(dirname "$0")"
export SLIDE_ROOT=${SLIDE_ROOT:-$(cd .. && pwd)} SLIDE_CACHE=${SLIDE_CACHE:-$HOME/.cache/slide}
python download_data.py                 # public corpora -> data_raw/
python stage1_encode_orig.py            # fine-tune encoder on WARM, encode + surprisal of original texts
python -m slide.attacks                 # T1 paraphrase attack (resumable)
python stage2_attacks_encode.py         # attacked texts (T1-T3) + their features
python audit_leakage.py                 # Sec. 6.1
# main, ablation, sensitivity and stress jobs: four single-threaded shards, then assembly
for i in 0 1 2 3; do python run_experiments.py --shard $i/4 & done; wait
python run_experiments.py --collect
python cert_and_displacement.py         # Prop. 1 certificate, attack displacement
python bench_latency.py                 # Table: latency and memory
python make_pool_table.py
python make_audit_table.py
python make_tables.py                   # LaTeX tables, figures, results/numbers.json
python make_extra_tables.py             # latency and certificate tables, Pareto figure
