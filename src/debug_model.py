import os
import sys

project_root_path = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
sys.path.append(project_root_path)

import library.translator as translator
import library.nn_architectures as nn_architectures
import torch
import pickle

# Load models
encoder_path = "../models/transformer_encoder_20250706_040900_39"
decoder_path = "../models/transformer_decoder_20250706_040900_39"
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# Initialize models
hidden_size = 256
encoder = nn_architectures.TransformerEncoder(9783, hidden_size, 55, num_layer=6)
decoder = nn_architectures.TransformerDecoder(hidden_size, 15532, 69, num_layer=6)

# Load state dicts
encoder.load_state_dict(torch.load(encoder_path, map_location=device, weights_only=True))
decoder.load_state_dict(torch.load(decoder_path, map_location=device, weights_only=True))
encoder = encoder.to(device)
decoder = decoder.to(device)
encoder.eval()
decoder.eval()

# Load vocabulary
with open("../data/fra_vocab.pkl", "rb") as f:
    tgt_vocab = pickle.load(f)
tgt_idx2word = tgt_vocab["idx2word"]

print("Target vocabulary stats:")
print(f"Total vocabulary size: {len(tgt_idx2word)}")
print(f"<sos> token index: 0, word: '{tgt_idx2word.get(0, 'NOT FOUND')}'")
print(f"<eos> token index: 1, word: '{tgt_idx2word.get(1, 'NOT FOUND')}'")
print(f"<pad> token index: 2, word: '{tgt_idx2word.get(2, 'NOT FOUND')}'")
print()

# Create a simple test input
src_idx = torch.tensor([[3, 4, 5, 6, 1] + [2]*50]).to(device)  # Simple sequence with padding
print(f"Source input shape: {src_idx.shape}")

# Get encoder output
with torch.no_grad():
    encoder_output, _ = encoder(src_idx)
    print(f"Encoder output shape: {encoder_output.shape}")
    
    # Test decoder with teacher forcing
    tgt_idx = torch.tensor([[0, 10, 20, 30, 1] + [2]*64]).to(device)  # <sos> + some tokens + <eos> + padding
    decoder_output, _, _ = decoder(encoder_output, 0, tgt_idx, src_idx)
    print(f"Decoder output shape: {decoder_output.shape}")
    
    # Check predictions
    print("\nDecoder predictions (teacher forcing):")
    for i in range(5):
        probs = torch.exp(decoder_output[0, i, :])  # Convert log probs to probs
        top5_probs, top5_idx = torch.topk(probs, 5)
        print(f"Position {i} (should predict token at position {i+1}):")
        print(f"  Target token: {tgt_idx[0, i+1].item()} ('{tgt_idx2word.get(tgt_idx[0, i+1].item(), 'UNK')}')")
        print(f"  Top 5 predictions:")
        for j in range(5):
            idx = top5_idx[j].item()
            prob = top5_probs[j].item()
            word = tgt_idx2word.get(idx, 'UNK')
            print(f"    {idx} ('{word}'): {prob:.4f}")
    
    # Test greedy inference
    print("\n\nTesting greedy inference:")
    output = decoder.infer_greedy(src_idx, encoder_output)
    print(f"Output shape: {output.shape}")
    print(f"Output tokens: {output[0].tolist()[:10]}")  # First 10 tokens
    
    # Decode the output
    decoded_words = []
    for idx in output[0].tolist():
        if idx == 0 or idx == 2:
            continue
        elif idx == 1:
            break
        decoded_words.append(tgt_idx2word.get(idx, 'UNK'))
    
    print(f"Decoded text: {' '.join(decoded_words) if decoded_words else '<EMPTY>'}")