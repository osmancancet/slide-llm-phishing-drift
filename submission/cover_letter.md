Osman Can Çetlenbik
Department of Statistics, Big Data Analyst Program
Manisa Vocational School of Technical Sciences
Manisa Celal Bayar University, Manisa 45140, Türkiye
osman.cetlenbik@cbu.edu.tr

To the Editor-in-Chief, *Neurocomputing*

Dear Editor,

Please consider the manuscript "When to buy labels: a label-free sliced drift trigger for budgeted adaptation of phishing detectors to LLM-generated e-mail" for publication in *Neurocomputing* as a full-length article.

Language models let attackers change the generator, wording or language of a phishing campaign at almost no cost. Verified labels for the new campaign arrive late and take analyst time. The manuscript asks when a detector should spend a fixed label budget. It proposes a label-free trigger: Kolmogorov–Smirnov tests on random projections of a frozen transformer representation, gated by a sliced-Wasserstein effect size. A harmonic α-spending schedule bounds the probability of any false alarm per reference epoch under overlapping windows. Each alarm buys a window of labels for any learner. We evaluate the trigger against samplers that spend the same number of labels. On a 36,000-message public stream with LLM-generated, paraphrased, PhishFuzzer and machine-translated Spanish segments, it reaches MCC 0.889 with 3,000 labels. Uniform, uncertainty and random-time purchasing reach 0.818, 0.794 and 0.774 at the same cost, and an oracle that knows the boundaries reaches 0.891. The trigger raised no false alarm on 20 stationary streams. The paper also reports what the trigger cannot do. A fully supervised online learner remains more accurate when labels are cheap and fast. Posterior change with unchanged inputs needs a labelled audit. A benign prefix defeats the 128-token encoder unless messages are scored in windows.

The manuscript fits the journal's interest in learning from non-stationary data. It builds on drift-detection work published in *Neurocomputing*: radial-distance and novelty-aware drift detection for neural networks (Shang et al., 2025, doi:10.1016/j.neucom.2025.131190 and doi:10.1016/j.neucom.2024.128933), feature-based analyses of concept drift (Hinder et al., 2024, doi:10.1016/j.neucom.2024.127968), and the recent survey of drift handling in security applications (Li et al., 2026, doi:10.1016/j.neucom.2026.134490). Those works detect drift. Ours asks what a detection should cost in labels, and it measures the value of the trigger at a fixed budget for two learners and four stream designs.

All data are public. Code, per-run results and the manuscript source are released under the MIT licence, and every number in the paper is read from files written by the released pipeline. The manuscript is original, has not been published, and is not under consideration elsewhere. An earlier version was reviewed and declined by another journal. That review led to a reduced method, a new evaluation and new experiments, and a point-by-point account of the changes is available on request.

Sincerely,

Osman Can Çetlenbik
