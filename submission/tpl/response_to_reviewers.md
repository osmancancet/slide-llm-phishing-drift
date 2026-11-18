# Response to the reviewers of KNOSYS-D-26-25455

**Manuscript:** "When to buy labels: a label-free sliced drift trigger for budgeted adaptation of phishing detectors to LLM-generated e-mail" (revised version of "Label-free sliced-Wasserstein drift detection and budgeted adaptation for LLM-generated phishing in e-mail streams")

**Author:** Osman Can Çetlenbik

We thank both reviewers for detailed comments. They led to a substantially different paper. The method was reduced to the parts that the ablation supports. The evaluation now tests when labels are bought, on a longer stream with more shift sources, and with controls at equal label cost. Every number below comes from the released pipeline, and section and table numbers refer to the revised manuscript.

## Summary of main changes

1. **Method.** The adversarial term, the proximal penalty and the four surprisal features were removed, because none of them changed accuracy. Replay is kept as an option because it preserves detection of earlier campaigns. A harmonic α-spending schedule replaces the per-test level. It bounds the probability of any false alarm per reference epoch, with any overlap between windows (Proposition {{ref:prop:fwer}}).
2. **Stream.** 36,000 messages in seven segments instead of 28,000 in five. The new segments are PhishFuzzer phishing and valid mail, and machine-translated Spanish phishing. There are also three alternative stream designs (gradual, variable-length, permuted), a stationary null stream and a real-drift probe.
3. **Equal-budget controls.** At the label cost of SLIDE, we compare uniform, uncertainty, random-time and oracle-time purchasing. Each is run for both the transformer head and a TF-IDF learner.
4. **Sensitivity.** Grids over the number of projections K, the label window n_ℓ and the label delay D.
5. **Limits.** We measure retention of earlier campaigns, the real-drift blind spot and the audit size it takes to see it, benign padding at several lengths together with windowed scoring and its cost, and authorship slices.

## Reviewer 1

**1.1 Overlapping windows and repeated testing are unresolved.**
We agree. The revision spends the level over the tests of a reference epoch, α_j = α/(j(j+1)). Proposition 1 shows that the probability of any false alarm in the epoch is at most α, whatever the overlap and the dependence among the tests (Section {{ref:sec:spend}}). Section {{ref:sec:null}} measures the result on 20 stationary streams of 20,000 messages, each running about {{null|alpha-spending|tests:.0f}} tests. With the per-test level, {{null|per-test alpha|p_any:.0%}} of the streams raised a false alarm, and {{null|per-test alpha, KS only|p_any:.0%}} did with the KS gate alone. With α-spending, none did. The bound has a cost in power. On the main stream, {{spend|n_later:.0f}} of {{spend|n_bound:.0f}} detections came later, by up to {{spend|max_later:.0f}} batches, and the mean delay rose from {{spend|delay|none:.2f}} to {{spend|delay|harm:.2f}} batches. We report this in Section {{ref:sec:spend}}.

**1.2 Proposition 2 controls a single test only.**
Correct. The former proposition is replaced by the stream-level bound described in 1.1. The effect-size gate is now presented only as a lower bound on transport cost (Proposition {{ref:prop:sw}}), with no false-alarm claim.

**1.3 Limited generator diversity.**
We added two shift sources: PhishFuzzer, whose phishing and valid messages are generated from seed e-mails (segment F), and a machine-translated Spanish release (segment G). We also added a second held-out pool that was never streamed. The stream now has seven segments from six releases. We agree that this is still a small set of generators. Section {{ref:sec:validity}} says so and names the untested case of a generator–critic attacker.

**1.4 Real drift with stable marginals is omitted.**
We added a real-drift probe (Section {{ref:sec:realdrift}}). Labels of one region of legitimate mail flip while P(X) is unchanged. As expected, SLIDE raises no alarm. A labelled audit monitored by ADWIN did not detect the change with 5 or 10 labels per batch. With 25 labels per batch it raised an alarm in {{=[real|SLIDE+Audit n_a=25|p_upd]*10:.0f}} of 10 realisations, on average {{real|SLIDE+Audit n_a=25|lag:.1f}} batches after the change. R1 MCC rose from {{real|SLIDE|R1:.3f}} to {{real|SLIDE+Audit n_a=25|R1:.3f}}. We report this as a limitation and do not present it as a fix. The label-free trigger complements a labelled audit stream and does not replace it.

**1.5 Replay and proximal regularisation show no gain.**
We agree for stream MCC, and the proximal term was removed. Replay is different. A new retention measurement (Table {{ref:tab:rev-retention}}) scores the detector on a held-out pool from the sources of segments B–D. At the end of the stream, SLIDE without replay detects {{ret|SLIDE|end:.3f}} of that pool and SLIDE with replay {{ret|SLIDE+replay|end:.3f}}. Replay costs {{vs|SLIDE+replay|d:.3f}} MCC on the current traffic. Stream MCC cannot show this benefit because the stream never returns to an old source. The default configuration is therefore SLIDE without replay, and SLIDE+replay is an option for deployments where old campaigns return.

**Major comment: adversarial training, padding, narrow attacks, surprisal cost.**
- *Adversarial training.* Removed. In the revised ablation, the full objective and the warm-start update lie within {{=abs([abl|warm+replay|d]):.3f}} MCC of each other.
- *Benign padding.* We extended the attacks to 200-, 400- and 800-character prefixes, a 400-character suffix and a prefix-plus-suffix attack, on two pools (Tables {{ref:tab:rev-stress}} and {{ref:tab:rev-stress2}}). The 400-character prefix still defeats every head on the frozen encoder (SLIDE {{stress|SLIDE|T3_pad:.3f}}). We added optional windowed scoring, which takes the maximum score over up to five 126-token windows (Section {{ref:sec:chunk}}). It restores detection to {{stress|SLIDE chunked|T3_pad:.3f}}. It also raises the false-positive rate on the clean stream from {{main|SLIDE|fpr:.3f}} to {{chunk|SLIDE chunked|fpr:.3f}}, and we report this cost in full.
- *Narrow attacks.* Besides the new padding variants, the second pool tests never-streamed PhishFuzzer and Spanish phishing. We agree that adaptive attacks against the detector remain untested (Section {{ref:sec:validity}}).
- *Surprisal features.* Removed. They added nothing ({{abl|warm + psi|mcc:.3f}} with them, {{abl|warm (revised update)|mcc:.3f}} without) and required a second language model ({{lat_lm_ms:.0f}} ms per message).

## Reviewer 2

**2.1 K fixed at 64.**
Table {{ref:tab:rev-K}} varies K from 2 to 512. K = 2 misses {{K|2|miss:.0%}} of the boundaries. From K = 64 on, no boundary is missed. Stream MCC is flat from K = 32 to 512 ({{K|32|mcc:.3f}}–{{K|512|mcc:.3f}}), while test time grows from {{K|64|t_detect:.0f}} s to {{K|512|t_detect:.0f}} s per stream.

**2.2 Smaller annotation budgets.**
Table {{ref:tab:rev-budget}} varies the label window from 50 to 1,000 messages per alarm. With 50 labels per alarm, SLIDE reaches {{budget|SLIDE|50|mcc:.3f}} using {{budget|SLIDE|50|labels:,.0f}} labels per stream. With 100 per alarm it reaches {{budget|SLIDE|100|mcc:.3f}}, and with the default 500 it reaches {{budget|SLIDE|500|mcc:.3f}}.

**2.3 and 2.4 Five predefined phases; identical construction across realisations.**
The stream has seven segments. Each realisation draws different messages for every segment. Three further designs are run (Table {{ref:tab:rev-designs}}): gradual boundaries with 2,000-message mixing zones, variable segment lengths, and a random order of segments B–G. SLIDE ends between {{design|SLIDE|varlen:.3f}} and {{design|SLIDE|gradual:.3f}} under these designs, and the equal-budget samplers stay below it under every design. We agree that a composition of public corpora is not organisational traffic (Section {{ref:sec:validity}}).

**2.5 Annotation delays.**
Table {{ref:tab:rev-delay}} varies the delay from 1 to 16 batches and adds random delays. SLIDE falls from {{delay|SLIDE|1:.3f}} at D = 1 to {{delay|SLIDE|16:.3f}} at D = 16. The fully supervised online TF-IDF learner falls at the same rate, and its advantage disappears from D = 8 on.

**2.6 Coverage of languages, domains and generators.**
See 1.3. A Spanish segment and PhishFuzzer were added. The remaining limits are stated in Section {{ref:sec:validity}}.

**2.7 to 2.9 Authorship versus intent; 15.7% FPR on LLM-written legitimate mail.**
The surprisal features were removed, so the representation is the encoder embedding alone. The authorship probe still separates LLM-written from human legitimate mail ({{auth_probe:.4f}} macro-F1). We report per-slice results (Table {{ref:tab:rev-slices}}). The excess false-positive rate on LLM-written legitimate mail in segment D remains for the head ({{slice|SLIDE|D_llm|fpr:.3f}}). With the same labels and alarms, SLIDE-TFIDF lowers it to {{slice|SLIDE-TFIDF|D_llm|fpr:.3f}}, so the excess comes from the frozen-encoder head and not from the label budget. In segment F, legitimate and malicious LLM mail are equally frequent. There, the adaptive detectors keep this FPR between {{slice|SLIDE-TFIDF|F_llm|fpr:.3f}} and {{slice|SLIDE|F_llm|fpr:.3f}} once such mail is labelled. We state the remaining confound in Section {{ref:sec:validity}}.

**2.10 Online-TFIDF-SGD beats SLIDE.**
It still does: {{main|Online-TFIDF-SGD|mcc:.3f}} against {{main|SLIDE|mcc:.3f}}. We no longer claim otherwise. That comparison is not at matched budgets, because the online learner uses {{main|Online-TFIDF-SGD|labels:,.0f}} labels and SLIDE uses {{main|SLIDE|labels:,.0f}}. The revised paper therefore asks a different question: at a fixed budget, when should labels be bought? At equal cost, SLIDE exceeds uniform ({{main|Budget-Uniform-MLP|mcc:.3f}}), uncertainty ({{main|Budget-Uncertainty-MLP|mcc:.3f}}) and random-time ({{main|Budget-Random|mcc:.3f}}) purchasing in all ten realisations, and it is within {{=abs([vs|Budget-Oracle|d]):.3f}} MCC of an oracle. The same holds with the TF-IDF learner: SLIDE-TFIDF {{main|SLIDE-TFIDF|mcc:.3f}} against uniform {{main|Budget-Uniform-TFIDF|mcc:.3f}} and uncertainty {{main|Budget-Uncertainty-TFIDF|mcc:.3f}}. The abstract, the introduction and Section {{ref:sec:discussion}}.1 state that full supervision is preferable when labels are cheap and fast.
