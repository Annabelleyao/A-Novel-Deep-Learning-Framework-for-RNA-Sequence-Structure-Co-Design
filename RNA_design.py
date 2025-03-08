import torch
from main_models import *

def main():
    file_path = "1c2x_C.pdb"  # 请替换为实际 PDB 文件路径
    args = gvp_args()  # or however you initialize your args
    model = CombinedNetwork(encoder_args=args)

    # Load the saved weights
    model.load_state_dict(torch.load("RNA_de.pth"))
    model.eval()  # Set the model to evaluation mode if needed
    rna_seq = model(file_path)
    print("Generated RNA Sequence:", rna_seq)
    # To save the model weights:


if __name__ == "__main__":
    main()
