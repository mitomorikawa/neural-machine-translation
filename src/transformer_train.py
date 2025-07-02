import os
import sys

project_root_path = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

sys.path.append(project_root_path)

import library.trainer as trainer
import library.nn_architectures as nn_architectures
import torch

loader = trainer.TensorLoader()
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"Using device: {device}")
src_train = loader.load("../data/eng_train.pt", device=device)
tgt_train = loader.load("../data/fra_train.pt", device=device)
src_val = loader.load("../data/eng_val.pt", device=device)
tgt_val = loader.load("../data/fra_val.pt", device=device)
hidden_size = 128
input_size = 9783
output_size = 15532
encoder = nn_architectures.TransformerEncoder(input_size, hidden_size, num_layer=1)
decoder = nn_architectures.TransformerDecoder(hidden_size, output_size,num_layer=1)
