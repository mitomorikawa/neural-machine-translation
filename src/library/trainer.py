""" 
This module contains the Trainer class, which is responsible for training the model.
"""

import torch
from torch.utils.tensorboard import SummaryWriter
from torch.cuda.amp import autocast, GradScaler
import time
from datetime import datetime
import os 
from tqdm import tqdm

os.makedirs("../models", exist_ok=True)

class TransformerScheduler:
    """
    Implements the learning rate schedule from the Transformer paper:
    lr = d_model^(-0.5) * min(step_num^(-0.5), step_num * warmup_steps^(-1.5))
    """
    
    def __init__(self, optimizer, d_model, warmup_steps=4000):
        self.optimizer = optimizer
        self.d_model = d_model
        self.warmup_steps = warmup_steps
        self.current_step = 0
        self.base_lr = d_model ** -0.5
        
    def step(self):
        self.current_step += 1
        lr = self.get_lr()
        for param_group in self.optimizer.param_groups:
            param_group['lr'] = lr
            
    def get_lr(self):
        arg1 = self.current_step ** -0.5
        arg2 = self.current_step * (self.warmup_steps ** -1.5)
        return self.base_lr * min(arg1, arg2) 

class TensorLoader:
    """
    This class is responsible for loading tensors from a pt file.
    """

    def load(self, file_path: str, device=torch.device("cuda" if torch.cuda.is_available() else "cpu")):
        """
        Loads a tensor from the specified file path
        
        Returns:
            torch.Tensor: The loaded tensor.
        """
        tensor = torch.load(file_path, map_location=device, weights_only=True)
        return tensor
    
    def create_dataloader(self, src_idx: torch.Tensor, tgt_idx: torch.Tensor, batch_size: int):
        """
        Creates a DataLoader from the source and target tensors.
        
        Args:
            src_idx (torch.Tensor): Source tensor indices.
            tgt_idx (torch.Tensor): Target tensor indices.
            batch_size (int): Size of each batch.
        
        Returns:
            torch.utils.data.DataLoader: DataLoader containing the source and target tensors.
        """
        dataset = torch.utils.data.TensorDataset(src_idx, tgt_idx)

        return torch.utils.data.DataLoader(dataset, batch_size=batch_size, shuffle=True)
    
class Trainer:
    """
    This class is responsible for training the model.

    attributes:
        encoder (nn.Module): The encoder neural network.
        decoder (nn.Module): The decoder neural network.
        optimizer (torch.optim.Optimizer): The optimizer for training.
        loss_fn (callable): The loss function to compute the loss.
        lr (float): Learning rate for the optimizer.
        n_epochs (int): Number of epochs to train the model.
        patience (int): Number of epochs with no improvement after which training will be stopped early.
        transformer (bool): Whether to use a transformer architecture or not.
    """
    
    def __init__(self, encoder, decoder, loss_fn, lr, n_epochs, patience=5, transformer=False, d_model=None, warmup_steps=4000, use_amp=False):
        self.encoder = encoder
        self.decoder = decoder
        self.loss_fn = loss_fn
        self.lr = lr
        self.n_epochs = n_epochs
        self.patience = patience
        self.device = next(encoder.parameters()).device 
        self.transformer = transformer
        self.d_model = d_model
        self.warmup_steps = warmup_steps
        self.use_amp = use_amp
        self.scaler = GradScaler() if use_amp else None

    def train_epoch(self, dataloader, encoder_optimizer, decoder_optimizer, encoder_scheduler=None, decoder_scheduler=None):
        """ 
        Trains the model for one epoch.

        params:
            dataloader (torch.utils.data.DataLoader): The DataLoader containing the training data.
            encoder_optimizer: Optimizer for encoder
            decoder_optimizer: Optimizer for decoder
            encoder_scheduler: Optional learning rate scheduler for encoder
            decoder_scheduler: Optional learning rate scheduler for decoder

        returns:
            float: The average loss for the epoch.
        """
        self.encoder.train()
        self.decoder.train()
        total_loss = 0.0
        for src_idx, tgt_idx in tqdm(dataloader, desc="Training Epoch", leave=False):
            src_idx = src_idx.to(self.device)
            tgt_idx = tgt_idx.to(self.device)
            encoder_optimizer.zero_grad()
            decoder_optimizer.zero_grad()
            
            if self.use_amp:
                with autocast(dtype=torch.float16):
                    encoder_outputs, encoder_hidden = self.encoder(src_idx)
                    decoder_outputs, _, _ = self.decoder(encoder_outputs, encoder_hidden, tgt_idx, src_idx)
                    if self.transformer:
                        # For transformer: decoder predicts next token, so we compare outputs with shifted targets
                        # decoder_outputs[:, i] should predict tgt_idx[:, i+1]
                        target = tgt_idx[:, 1:]  # Remove <sos> token from targets
                        decoder_outputs = decoder_outputs[:, :-1]  # Remove last prediction
                    else:
                        target = tgt_idx
                    loss = self.loss_fn(decoder_outputs.reshape(-1, decoder_outputs.size(-1)), target.reshape(-1))
                
                self.scaler.scale(loss).backward()
                self.scaler.unscale_(encoder_optimizer)
                self.scaler.unscale_(decoder_optimizer)
                torch.nn.utils.clip_grad_norm_(self.encoder.parameters(), max_norm=1.0)
                torch.nn.utils.clip_grad_norm_(self.decoder.parameters(), max_norm=1.0)
                self.scaler.step(encoder_optimizer)
                self.scaler.step(decoder_optimizer)
                self.scaler.update()
            else:
                encoder_outputs, encoder_hidden = self.encoder(src_idx)
                
                if self.transformer:
                    # For transformer: decoder predicts next token, so we compare outputs with shifted targets
                    # decoder_outputs[:, i] should predict tgt_idx[:, i+1]
                    target = tgt_idx[:, 1:]  # Remove <sos> token from targets
                    tgt_idx_truncated = tgt_idx[:, :-1]  # Remove last prediction for decoder input
                    decoder_outputs, _, _ = self.decoder(encoder_outputs, encoder_hidden, tgt_idx_truncated, src_idx)
                else:
                    decoder_outputs, _, _ = self.decoder(encoder_outputs, encoder_hidden, tgt_idx, src_idx)
                    target = tgt_idx
                loss = self.loss_fn(decoder_outputs.reshape(-1, decoder_outputs.size(-1)), target.reshape(-1))
                loss.backward()
                torch.nn.utils.clip_grad_norm_(self.encoder.parameters(), max_norm=1.0)
                torch.nn.utils.clip_grad_norm_(self.decoder.parameters(), max_norm=1.0)
                encoder_optimizer.step()
                decoder_optimizer.step()
            
            if encoder_scheduler:
                encoder_scheduler.step()
            if decoder_scheduler:
                decoder_scheduler.step()
            total_loss += loss.item()

        return total_loss / len(dataloader)
    
    def train(self, train_dataloader, val_dataloader, encoder_name="encoder", decoder_name="decoder"):
        """ 
        Trains the model using the provided training and validation data loaders.
        params:
            train_dataloader (torch.utils.data.DataLoader): DataLoader for training data.
            val_dataloader (torch.utils.data.DataLoader): DataLoader for validation data.
            encoder_name (str): Name for saving the encoder model.
            decoder_name (str): Name for saving the decoder model.
        returns:
            None
        """
        start = time.time()
        # For transformer, use specific Adam parameters from the paper
        if self.transformer:
            encoder_optimizer = torch.optim.Adam(self.encoder.parameters(), lr=1.0, betas=(0.9, 0.98), eps=1e-9)
            decoder_optimizer = torch.optim.Adam(self.decoder.parameters(), lr=1.0, betas=(0.9, 0.98), eps=1e-9)
        else:
            encoder_optimizer = torch.optim.Adam(self.encoder.parameters(), lr=self.lr)
            decoder_optimizer = torch.optim.Adam(self.decoder.parameters(), lr=self.lr)
        
        # Create schedulers if using transformer with d_model specified
        encoder_scheduler = None
        decoder_scheduler = None
        if self.transformer and self.d_model:
            encoder_scheduler = TransformerScheduler(encoder_optimizer, self.d_model, self.warmup_steps)
            decoder_scheduler = TransformerScheduler(decoder_optimizer, self.d_model, self.warmup_steps)
        
        writer = SummaryWriter(log_dir="../runs/training_logs")
        best_vloss = float('inf')
        epochs_no_improve = 0  # ADDED LINE: Counter for early stopping
        for epoch in range(self.n_epochs):
            epoch_train_loss = self.train_epoch(train_dataloader, encoder_optimizer, decoder_optimizer, encoder_scheduler, decoder_scheduler)
            epoch_val_total_loss = 0

            self.encoder.eval()
            self.decoder.eval()

            with torch.no_grad():
                for src_idx, tgt_idx in val_dataloader:
                    src_idx = src_idx.to(self.device)
                    tgt_idx = tgt_idx.to(self.device)
                    if self.use_amp:
                        with autocast(dtype=torch.float16):
                            encoder_outputs, encoder_hidden = self.encoder(src_idx)
                            decoder_outputs, _, _ = self.decoder(encoder_outputs, encoder_hidden, tgt_idx, src_idx)
                            if self.transformer:
                                # For transformer: decoder predicts next token, so we compare outputs with shifted targets
                                target = tgt_idx[:, 1:]  # Remove <sos> token from targets
                                decoder_outputs = decoder_outputs[:, :-1]  # Remove last prediction
                            else:
                                target = tgt_idx
                            loss = self.loss_fn(decoder_outputs.reshape(-1, decoder_outputs.size(-1)), target.reshape(-1))
                    else:
                        encoder_outputs, encoder_hidden = self.encoder(src_idx)
                        if self.transformer:
                            # For transformer: decoder predicts next token, so we compare outputs with shifted targets
                            target = tgt_idx[:, 1:]  # Remove <sos> token from targets
                            tgt_idx_truncated = tgt_idx[:, :-1]  # Remove last prediction
                            decoder_outputs, _, _ = self.decoder(encoder_outputs, encoder_hidden, tgt_idx_truncated, src_idx)
                        else:
                            decoder_outputs, _, _ = self.decoder(encoder_outputs, encoder_hidden, tgt_idx, src_idx)
                            target = tgt_idx
                        loss = self.loss_fn(decoder_outputs.reshape(-1, decoder_outputs.size(-1)), target.reshape(-1))
                    epoch_val_total_loss += loss.item()
            elapsed = time.time() - start
            h, rem = divmod(elapsed, 3600)
            m, s = divmod(rem, 60)
            epoch_val_loss = epoch_val_total_loss / len(val_dataloader)
            print(f"{int(h)}h {int(m)}m {int(s)}s elapsed. ",
                f"Epoch {epoch+1}/{self.n_epochs}, ",
                  f"Train Loss: {epoch_train_loss:.4f}, ",
                  f"Validation Loss: {epoch_val_loss:.4f}")
        
            writer.add_scalar('Loss/train', epoch_train_loss, epoch) 
            writer.add_scalar('Loss/val', epoch_val_loss, epoch)   
            writer.flush()   
            if epoch_val_loss < best_vloss:
                best_vloss = epoch_val_loss
                epochs_no_improve = 0 # ADDED LINE: Reset counter on improvement
                # Your existing model saving logic is already correct here
                encoder_path = '../models/{}_{}_{}'.format(encoder_name,datetime.now().strftime('%Y%m%d_%H%M%S'), epoch)
                torch.save(self.encoder.state_dict(), encoder_path)
                decoder_path = '../models/{}_{}_{}'.format(decoder_name,datetime.now().strftime('%Y%m%d_%H%M%S'), epoch)
                torch.save(self.decoder.state_dict(), decoder_path)
            else:
                epochs_no_improve += 1 # ADDED LINE: Increment counter if no improvement

        # ADDED LINE: Check for early stopping
            if epochs_no_improve >= self.patience:
                print(f'\nEarly stopping triggered after {self.patience} epochs with no improvement.')
                break # Exit the training loop
        writer.close()