<h1><b>RNA Structural Geometry with GVP and Transformers: A Novel Deep Framework for Sequence-Structure Co-Design </b></h1>

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
