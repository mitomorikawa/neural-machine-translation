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
encoder = nn_architectures.EncoderRNN(9783, 128).to(device)
decoder = nn_architectures.AttnDecoderRNN(128, 15532, 69).to(device)

train_instance = trainer.Trainer(
    encoder=encoder,
    decoder=decoder,
    loss_fn=torch.nn.CrossEntropyLoss(ignore_index=2),
    lr=0.001,
    n_epochs=100
)

train_dataloader = loader.create_dataloader(src_train, tgt_train, batch_size=16)
val_dataloader = loader.create_dataloader(src_val, tgt_val, batch_size=16)
train_instance.train(train_dataloader, val_dataloader)