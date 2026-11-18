# Response to the reviewers of KNOSYS-D-26-25455

**Manuscript:** "When to buy labels: a label-free sliced drift trigger for budgeted adaptation of phishing detectors to LLM-generated e-mail" (revised version of "Label-free sliced-Wasserstein drift detection and budgeted adaptation for LLM-generated phishing in e-mail streams")

**Author:** Osman Can Çetlenbik

We thank both reviewers for detailed comments. They led to a substantially different paper. The method was reduced to the parts that the ablation supports. The evaluation now tests when labels are bought, on a longer stream with more shift sources, and with controls at equal label cost. Every number below comes from the released pipeline, and section and table numbers refer to the revised manuscript.

## Summary of main changes

1. **Method.** The adversarial term, the proximal penalty and the four surprisal features were removed, because none of them changed accuracy. Replay is kept as an option because it preserves detection of earlier campaigns. A harmonic α-spending schedule replaces the per-test level. It bounds the probability of any false alarm per reference epoch, with any overlap between windows (Proposition 1).
2. **Stream.** 36,000 messages in seven segments instead of 28,000 in five. The new segments are PhishFuzzer phishing and valid mail, and machine-translated Spanish phishing. There are also three alternative stream designs (gradual, variable-length, permuted), a stationary null stream and a real-drift probe.
3. **Equal-budget controls.** At the label cost of SLIDE, we compare uniform, uncertainty, random-time and oracle-time purchasing. Each is run for both the transformer head and a TF-IDF learner.
4. **Sensitivity.** Grids over the number of projections K, the label window n_ℓ and the label delay D.
5. **Limits.** We measure retention of earlier campaigns, the real-drift blind spot and the audit size it takes to see it, benign padding at several lengths together with windowed scoring and its cost, and authorship slices.

## Reviewer 1

**1.1 Overlapping windows and repeated testing are unresolved.**
We agree. The revision spends the level over the tests of a reference epoch, α_j = α/(j(j+1)). Proposition 1 shows that the probability of any false alarm in the epoch is at most α, whatever the overlap and the dependence among the tests (Section 4.2). Section 6.7 measures the result on 20 stationary streams of 20,000 messages, each running about 75 tests. With the per-test level, 10% of the streams raised a false alarm, and 45% did with the KS gate alone. With α-spending, none did. The bound has a cost in power. On the main stream, 7 of 60 detections came later, by up to 6 batches, and the mean delay rose from 0.43 to 0.65 batches. We report this in Section 4.2.

**1.2 Proposition 2 controls a single test only.**
Correct. The former proposition is replaced by the stream-level bound described in 1.1. The effect-size gate is now presented only as a lower bound on transport cost (Proposition 2), with no false-alarm claim.

**1.3 Limited generator diversity.**
We added two shift sources: PhishFuzzer, whose phishing and valid messages are generated from seed e-mails (segment F), and a machine-translated Spanish release (segment G). We also added a second held-out pool that was never streamed. The stream now has seven segments from six releases. We agree that this is still a small set of generators. Section 7.3 says so and names the untested case of a generator–critic attacker.

**1.4 Real drift with stable marginals is omitted.**
We added a real-drift probe (Section 6.8). Labels of one region of legitimate mail flip while P(X) is unchanged. As expected, SLIDE raises no alarm. A labelled audit monitored by ADWIN did not detect the change with 5 or 10 labels per batch. With 25 labels per batch it raised an alarm in 6 of 10 realisations, on average 10.5 batches after the change. R1 MCC rose from 0.710 to 0.736. We report this as a limitation and do not present it as a fix. The label-free trigger complements a labelled audit stream and does not replace it.

**1.5 Replay and proximal regularisation show no gain.**
We agree for stream MCC, and the proximal term was removed. Replay is different. A new retention measurement (Table 10) scores the detector on a held-out pool from the sources of segments B–D. At the end of the stream, SLIDE without replay detects 0.659 of that pool and SLIDE with replay 0.941. Replay costs 0.005 MCC on the current traffic. Stream MCC cannot show this benefit because the stream never returns to an old source. The default configuration is therefore SLIDE without replay, and SLIDE+replay is an option for deployments where old campaigns return.

**Major comment: adversarial training, padding, narrow attacks, surprisal cost.**
- *Adversarial training.* Removed. In the revised ablation, the full objective and the warm-start update lie within 0.005 MCC of each other.
- *Benign padding.* We extended the attacks to 200-, 400- and 800-character prefixes, a 400-character suffix and a prefix-plus-suffix attack, on two pools (Tables 18 and 19). The 400-character prefix still defeats every head on the frozen encoder (SLIDE 0.034). We added optional windowed scoring, which takes the maximum score over up to five 126-token windows (Section 4.4). It restores detection to 0.801. It also raises the false-positive rate on the clean stream from 0.045 to 0.259, and we report this cost in full.
- *Narrow attacks.* Besides the new padding variants, the second pool tests never-streamed PhishFuzzer and Spanish phishing. We agree that adaptive attacks against the detector remain untested (Section 7.3).
- *Surprisal features.* Removed. They added nothing (0.890 with them, 0.889 without) and required a second language model (72 ms per message).

## Reviewer 2

**2.1 K fixed at 64.**
Table 11 varies K from 2 to 512. K = 2 misses 23% of the boundaries. From K = 64 on, no boundary is missed. Stream MCC is flat from K = 32 to 512 (0.887–0.887), while test time grows from 32 s to 126 s per stream.

**2.2 Smaller annotation budgets.**
Table 12 varies the label window from 50 to 1,000 messages per alarm. With 50 labels per alarm, SLIDE reaches 0.820 using 300 labels per stream. With 100 per alarm it reaches 0.853, and with the default 500 it reaches 0.888.

**2.3 and 2.4 Five predefined phases; identical construction across realisations.**
The stream has seven segments. Each realisation draws different messages for every segment. Three further designs are run (Table 14): gradual boundaries with 2,000-message mixing zones, variable segment lengths, and a random order of segments B–G. SLIDE ends between 0.873 and 0.886 under these designs, and the equal-budget samplers stay below it under every design. We agree that a composition of public corpora is not organisational traffic (Section 7.3).

**2.5 Annotation delays.**
Table 13 varies the delay from 1 to 16 batches and adds random delays. SLIDE falls from 0.898 at D = 1 to 0.745 at D = 16. The fully supervised online TF-IDF learner falls at the same rate, and its advantage disappears from D = 8 on.

**2.6 Coverage of languages, domains and generators.**
See 1.3. A Spanish segment and PhishFuzzer were added. The remaining limits are stated in Section 7.3.

**2.7 to 2.9 Authorship versus intent; 15.7% FPR on LLM-written legitimate mail.**
The surprisal features were removed, so the representation is the encoder embedding alone. The authorship probe still separates LLM-written from human legitimate mail (0.9996 macro-F1). We report per-slice results (Table 17). The excess false-positive rate on LLM-written legitimate mail in segment D remains for the head (0.166). With the same labels and alarms, SLIDE-TFIDF lowers it to 0.089, so the excess comes from the frozen-encoder head and not from the label budget. In segment F, legitimate and malicious LLM mail are equally frequent. There, the adaptive detectors keep this FPR between 0.064 and 0.124 once such mail is labelled. We state the remaining confound in Section 7.3.

**2.10 Online-TFIDF-SGD beats SLIDE.**
It still does: 0.904 against 0.889. We no longer claim otherwise. That comparison is not at matched budgets, because the online learner uses 35,500 labels and SLIDE uses 3,000. The revised paper therefore asks a different question: at a fixed budget, when should labels be bought? At equal cost, SLIDE exceeds uniform (0.818), uncertainty (0.794) and random-time (0.774) purchasing in all ten realisations, and it is within 0.002 MCC of an oracle. The same holds with the TF-IDF learner: SLIDE-TFIDF 0.885 against uniform 0.829 and uncertainty 0.834. The abstract, the introduction and Section 7.1 state that full supervision is preferable when labels are cheap and fast.
