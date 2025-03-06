# A-Novel-Deep-Learning-Framework-for-RNA-Sequence-Structure-Co-Design

RNA’s diverse biological functions stem from its structural versatility, yet accu-
rately predicting and designing RNA sequences for a given 3D conformation (inverse
folding) remains an enormous challenge to this day. Here, we introduce a novel deep
learning framework that integrates Geometric Vector Perceptron (GVP) layers with a
Transformer for end-to-end RNA design. We construct a dataset consists of experi-
mentally solved RNA 3D structures, filtered and deduplicated from the BGSU RNA
list, and evaluate our method using two main metrics: recovery rate (sequence-level
match) and TM-score (structure-level similarity). On standard benchmarks and RNA-
Puzzles, our model consistently outperforms established approaches in both recovery
rate and TM-score, demonstrating robust performance across diverse RNA families
and lengths. Mask-Family validation using Rfam annotations further confirms the
model’s generalization ability. In addition, our sequence designs—folded back with Al-
phaFold3—maintain high fidelity to native structures, suggesting that geometric fea-
tures, captured via our created GVP layers, significantly enhance Transformer-based
sequence generation.
