<h1><b>RNA Structural Geometry with GVP and Transformers: A Novel Deep Framework for Sequence-Structure Co-Design </b></h1>

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

# RNA Design

This project provides tools for RNA design using deep learning and structural data. The code is designed to run in a UNIX environment.

## Requirements

Ensure you have the following packages installed:

- **Python**: 3.6.13
- **PyTorch**: 1.8.1
- **torch_geometric**: 1.7.0
- **torch_scatter**: 2.0.6
- **torch_cluster**: 1.5.9
- **tqdm**: 4.38.0
- **NumPy**: 1.19.4
- **scikit-learn**: 0.24.1
- **atom3d**: 0.2.1

> **Note:** Although these specific versions have been tested, any reasonably recent versions should work.

## General Usage

1. **Prepare Your PDB File**

   - Update the `file_path` in `RNA_design.py` to point to your local PDB file.
   - **Important:** If your PDB file has missing atoms, please use AREANA to complete the structure before running the script.
   - We provide two example in samples.

2. **Run the Script**

   Open your terminal and execute:
   ```bash
   python3 RNA_design.py
