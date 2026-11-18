# Venue gate: SLIDE revision (2026-11-18)

## 1. Contribution type
Method, narrowly scoped: a label-free trigger that decides *when* to buy labels under a fixed budget.
Carrying sentence: at an equal label budget, buying labels right after a label-free sliced drift alarm beats uniform, uncertainty and random-time purchasing for two learners and four stream designs and comes within 0.002 MCC of a boundary oracle.

## 2. Ablation gate (each kept component isolated)
| Component | Evidence | Result |
|---|---|---|
| Random projections (vs score-only test) | Ablation table: 0.889 vs 0.828, p_Holm 0.0215 | pass |
| Adaptation on alarm | no adaptation 0.664 | pass |
| α-spending | null stream: P(any false alarm) 0.10 → 0.00 (per-test vs spending) | pass (stationary streams), costs delay on 7/60 detections |
| Two gates (KS ∧ SW) | identical alarms on A–G; SW alone false-alarms on 15% of null streams | pass for SW-only; KS alone equal to conjunction |
| Replay (optional) | stream MCC −0.005; end-of-stream retention 0.659 → 0.941 | kept as option, not default |
| Adversarial term, proximal penalty, surprisal ψ | no measurable gain | removed |
| Strongest simple baseline | Online-TFIDF-SGD with all 35,500 labels: 0.904 > 0.889 | **fails at unequal budget**; paper no longer claims superiority, framed as budgeted acquisition; equal-budget controls pass |

## 3. Novelty delta
Prior drift detectors (Shang et al. 2025 radial distance / novelty-aware, Hinder et al. 2024, MCE-CDD 2026, KSWIN, ADWIN, sliced-KS of Rabanser et al. 2019) detect drift; none measures the value of a detection as a label-purchase trigger against equal-budget samplers. Delta: the same labels are worth 0.07–0.12 MCC more when timed by the trigger.
Sibling check: SCOPE (KBS transfer, measurement study on PhishFuzzer leakage) shares the PhishFuzzer corpus only; no shared text, tables or claims. Do not submit both with overlapping corpus description verbatim.

## 4. Depth
- What: equal-budget controls, 10 realisations, Holm-corrected Wilcoxon. Present.
- Why: score-only ablation and segment/trajectory analysis (head confidently wrong on LLM segments). Present.
- So what: operating rules (when full supervision is better: delay < 8 batches; audit size for real drift; windowed scoring cost). Present.

## 5. Venues (subscription route only, no APC)
| Journal | Checked | Exclusions / notes | Same-type recent papers | Verdict |
|---|---|---|---|---|
| Neurocomputing | 2026-11-17 (scope, hybrid subscription route) | none for ML methods | ≥4 drift-detection method papers 2024–2026 (DOIs in cover letter) | **fit, recommended** |
| Expert Systems with Applications | 2026-11-17 | wants algorithmic innovation; SCOPE desk-reject 2026-10-22 | drift + security papers present | possible; higher desk risk |
| Knowledge-Based Systems | — | rejected after review 2026-11-17; 6-month bar | — | excluded |
| Computers & Security | — | AI/ML moratorium | — | excluded |

## 6. Editor first-page test
- Contribution type clear in first two abstract sentences: pass ("We study when a fixed label budget should be spent").
- Title promises what results deliver: pass (no superiority claim).
- Highlights carry numbers: pass (3,000 labels, 0.002 MCC, 20 streams).
- Cover letter cites journal's own papers: pass (4 Neurocomputing papers, all cited in the manuscript).
- Contradiction with baselines: none; Online-TFIDF advantage stated in abstract.

## Verdict
**GO** for Neurocomputing (pending Osman's choice on the decision card).
