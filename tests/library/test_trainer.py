import os
import sys

project_root_path = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

sys.path.append(project_root_path)

import src.library.trainer as trainer
import src.library.nn_architectures as nn_architectures
import torch

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
def test_TensorLoader():
    loader = trainer.TensorLoader()
    src_train = loader.load("test_data/src_train.pt", device = device)
    expected_src_train = torch.tensor([[5, 5], [9, 9], [1, 1], [6, 6], [0, 0], [2, 2], [7, 7], [4, 4]], dtype=torch.long,device=device)

    if not torch.equal(src_train, expected_src_train):
        raise AssertionError(f"""TensorLoader did not produce the expected output.
                              Expected: {expected_src_train}
                                 Got: {src_train}
                              """)
    
    src_train = loader.load("../../data/eng_train.pt", device = device)[:10]
    tgt_train = loader.load("../../data/fra_train.pt", device = device)[:10]

    dataloader = loader.create_dataloader(src_train, tgt_train, batch_size=2)
    for batch in dataloader:
        src_idx, tgt_idx = batch
        if src_idx.shape != (2, 55):
            raise AssertionError(f"Batch source tensor shape is incorrect: {src_idx.shape}, expected (2, 55)")
        if tgt_idx.shape != (2, 69):
            raise AssertionError(f"Batch target tensor shape is incorrect: {tgt_idx.shape}, expected (2, 69)")
    print("DataLoader test passed.")

def test_Trainer():
    loader = trainer.TensorLoader()
    src_train = loader.load("../../data/eng_train.pt", device = device)
    tgt_train = loader.load("../../data/fra_train.pt", device = device)
    src_val = loader.load("../../data/eng_val.pt", device = device)
    tgt_val = loader.load("../../data/fra_val.pt", device = device)
    encoder = nn_architectures.EncoderRNN(9783, 16).to(device)
    decoder = nn_architectures.AttnDecoderRNN(16, 15532, 69).to(device)
    mini_trainer = trainer.Trainer(
        encoder=encoder,
        decoder=decoder,
        loss_fn=torch.nn.CrossEntropyLoss(ignore_index=2),
        lr=0.001,
        n_epochs=10
    )
    train_dataloader = loader.create_dataloader(src_train, tgt_train, batch_size=16)
    val_dataloader = loader.create_dataloader(src_val, tgt_val, batch_size=16)
    mini_trainer.train(train_dataloader, val_dataloader)



if __name__ == "__main__":
    test_TensorLoader()
    test_Trainer()