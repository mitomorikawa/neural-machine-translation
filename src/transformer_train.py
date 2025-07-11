import os
import sys
import argparse

project_root_path = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

sys.path.append(project_root_path)

import library.trainer as trainer
import library.nn_architectures as nn_architectures
import torch

def main():
    parser = argparse.ArgumentParser(description='Train Transformer model')
    parser.add_argument('--hidden_size', type=int, default=256, help='Hidden size (d_model) for transformer')
    parser.add_argument('--batch_size', type=int, default=256, help='Batch size for training')
    parser.add_argument('--lr', type=float, default=0.001, help='Learning rate')
    parser.add_argument('--n_epochs', type=int, default=400, help='Number of epochs')
    parser.add_argument('--num_layers', type=int, default=3, help='Number of transformer layers')
    parser.add_argument('--dropout', type=float, default=0.3, help='Dropout probability')
    parser.add_argument('--relposenc', type=bool, default=False, help='Relative or absolute positional encoding')
    parser.add_argument('--linear_hidden_ratio', type=int, default=8, help='Feedforward hidden layer ratio')
    parser.add_argument('--label_smoothing', type=float, default=0.1, help='Label smoothing value')
    parser.add_argument('--warmup_steps', type=int, default=4000, help='Warmup steps for learning rate scheduler')
    
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
    src_seq_len = 55  # Maximum source sequence length
    tgt_seq_len = 68  # Maximum target sequence length

    encoder = nn_architectures.TransformerEncoder(input_size, args.hidden_size, src_seq_len, 
                                                 num_layer=args.num_layers, dropout_p=args.dropout, relposenc=args.relposenc,
                                                 linear_hidden_ratio=args.linear_hidden_ratio).to(device)
    decoder = nn_architectures.TransformerDecoder(args.hidden_size, output_size, tgt_seq_len, 
                                                 num_layer=args.num_layers, dropout_p=args.dropout, relposenc=args.relposenc,
                                                 linear_hidden_ratio=args.linear_hidden_ratio).to(device)


    train_instance = trainer.Trainer(
        encoder=encoder,
        decoder=decoder,
        loss_fn=torch.nn.CrossEntropyLoss(ignore_index=2, label_smoothing=args.label_smoothing),
        lr=args.lr,
        n_epochs=args.n_epochs,
        transformer=True,
        d_model=args.hidden_size,
        warmup_steps=args.warmup_steps
    )

    train_dataloader = loader.create_dataloader(src_train, tgt_train, batch_size=args.batch_size)
    val_dataloader = loader.create_dataloader(src_val, tgt_val, batch_size=args.batch_size)
    train_instance.train(train_dataloader, val_dataloader, encoder_name="transformer_encoder", decoder_name="transformer_decoder")

if __name__ == "__main__":
    main()