import os
import sys
import argparse

project_root_path = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

sys.path.append(project_root_path)

import library.trainer as trainer
import library.nn_architectures as nn_architectures
import torch

def main():
    parser = argparse.ArgumentParser(description='Train Bahdanau attention model')
    parser.add_argument('--hidden_size', type=int, default=1024, help='Hidden size for encoder/decoder')
    parser.add_argument('--batch_size', type=int, default=256, help='Batch size for training')
    parser.add_argument('--lr', type=float, default=0.001, help='Learning rate')
    parser.add_argument('--n_epochs', type=int, default=100, help='Number of epochs')
    
    args = parser.parse_args()
    
    loader = trainer.TensorLoader()
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")
    src_train = loader.load("../data/eng_train.pt", device=device)
    tgt_train = loader.load("../data/fra_train.pt", device=device)
    src_val = loader.load("../data/eng_val.pt", device=device)
    tgt_val = loader.load("../data/fra_val.pt", device=device)
    
    input_size = 9783
    output_size = 15532
    encoder = nn_architectures.RNNEncoder(input_size, args.hidden_size).to(device)
    decoder = nn_architectures.RNNDecoder(args.hidden_size, output_size, 69).to(device)

    train_instance = trainer.Trainer(
        encoder=encoder,
        decoder=decoder,
        loss_fn=torch.nn.CrossEntropyLoss(ignore_index=2),
        lr=args.lr,
        n_epochs=args.n_epochs
    )

    train_dataloader = loader.create_dataloader(src_train, tgt_train, batch_size=args.batch_size)
    val_dataloader = loader.create_dataloader(src_val, tgt_val, batch_size=args.batch_size)
    train_instance.train(train_dataloader, val_dataloader, encoder_name="bahdanau_encoder", decoder_name="bahdanau_decoder")

if __name__ == "__main__":
    main()