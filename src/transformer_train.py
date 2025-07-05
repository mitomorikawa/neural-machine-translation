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
batch_size = 128
src_seq_len = 55  # Maximum source sequence length
tgt_seq_len = 69  # Maximum target sequence length
encoder = nn_architectures.TransformerEncoder(input_size, hidden_size, src_seq_len, num_layer=1).to(device)
decoder = nn_architectures.TransformerDecoder(hidden_size, output_size, tgt_seq_len, num_layer=1).to(device)

train_instance = trainer.Trainer(
    encoder=encoder,
    decoder=decoder,
    loss_fn=torch.nn.CrossEntropyLoss(ignore_index=2),
    lr=0.001,
    n_epochs=100
)

train_dataloader = loader.create_dataloader(src_train, tgt_train, batch_size=batch_size)
val_dataloader = loader.create_dataloader(src_val, tgt_val, batch_size=batch_size)
train_instance.train(train_dataloader, val_dataloader, encoder_name="transformer_encoder", decoder_name="transformer_decoder")