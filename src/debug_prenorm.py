import os
import sys

project_root_path = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
sys.path.append(project_root_path)

import library.translator as translator
import library.nn_architectures as nn_architectures
import torch
import pickle
import numpy as np

# Load models
encoder_path = "../models/transformer_encoder_20250706_141602_5"
decoder_path = "../models/transformer_decoder_20250706_141602_5"
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# Initialize models with correct architecture
hidden_size = 256
encoder = nn_architectures.TransformerEncoder(9783, hidden_size, 55, num_layer=2)
decoder = nn_architectures.TransformerDecoder(hidden_size, 15532, 69, num_layer=2)

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

print("=== Model Architecture Debug ===")
print(f"Encoder layers: {encoder.num_layer}")
print(f"Decoder layers: {decoder.num_layer}")
print(f"Hidden size: {hidden_size}")

# Create a simple test input
src_idx = torch.tensor([[3, 4, 5, 6, 1] + [2]*50]).to(device)  # Simple sequence with padding

# Check layer outputs
print("\n=== Checking Encoder Layer Outputs ===")
with torch.no_grad():
    # Get embedding
    embedding = encoder.embedding(src_idx)
    embedding = embedding * (encoder.hidden_size ** 0.5)
    embedding = encoder.dropout(embedding)
    
    print(f"Initial embedding stats:")
    print(f"  Mean: {embedding.mean().item():.6f}")
    print(f"  Std: {embedding.std().item():.6f}")
    print(f"  Max: {embedding.max().item():.6f}")
    print(f"  Min: {embedding.min().item():.6f}")
    
    # Pass through encoder layers
    x = embedding
    for i, layer in enumerate(encoder.encoderlayers):
        x = layer(x)
        print(f"\nAfter encoder layer {i}:")
        print(f"  Mean: {x.mean().item():.6f}")
        print(f"  Std: {x.std().item():.6f}")
        print(f"  Max: {x.max().item():.6f}")
        print(f"  Min: {x.min().item():.6f}")
    
    # Apply final layer norm
    x = encoder.final_layer_norm(x)
    print(f"\nAfter final layer norm:")
    print(f"  Mean: {x.mean().item():.6f}")
    print(f"  Std: {x.std().item():.6f}")
    print(f"  Max: {x.max().item():.6f}")
    print(f"  Min: {x.min().item():.6f}")
    
    encoder_output = x

print("\n=== Checking Decoder Outputs ===")
# Test decoder with a simple input
tgt_idx = torch.tensor([[0] + [2]*68]).to(device)  # Just SOS token

with torch.no_grad():
    # Get decoder embedding
    embedding = decoder.embedding(tgt_idx)
    embedding = embedding * (decoder.hidden_size ** 0.5)
    embedding = decoder.dropout(embedding)
    
    print(f"Initial decoder embedding stats:")
    print(f"  Mean: {embedding.mean().item():.6f}")
    print(f"  Std: {embedding.std().item():.6f}")
    
    # Pass through decoder layers
    x = embedding
    for i, layer in enumerate(decoder.decoderlayers):
        x = layer(x, encoder_output)
        print(f"\nAfter decoder layer {i}:")
        print(f"  Mean: {x.mean().item():.6f}")
        print(f"  Std: {x.std().item():.6f}")
        print(f"  Max: {x.max().item():.6f}")
        print(f"  Min: {x.min().item():.6f}")
    
    # Apply final layer norm
    x = decoder.final_layer_norm(x)
    print(f"\nAfter final layer norm:")
    print(f"  Mean: {x.mean().item():.6f}")
    print(f"  Std: {x.std().item():.6f}")
    print(f"  Max: {x.max().item():.6f}")
    print(f"  Min: {x.min().item():.6f}")
    
    # Apply output projection
    output = decoder.output(x)
    print(f"\nAfter output projection (before softmax):")
    print(f"  Mean: {output.mean().item():.6f}")
    print(f"  Std: {output.std().item():.6f}")
    print(f"  Max: {output.max().item():.6f}")
    print(f"  Min: {output.min().item():.6f}")
    
    # Apply log softmax
    log_probs = torch.nn.functional.log_softmax(output, dim=-1)
    probs = torch.exp(log_probs)
    
    print(f"\nAfter softmax:")
    print(f"  Max probability: {probs.max().item():.6f}")
    print(f"  Min probability: {probs.min().item():.6f}")
    
    # Check top predictions
    top_probs, top_idx = torch.topk(probs[0, 0, :], 10)
    print(f"\nTop 10 predictions for first position:")
    for i in range(10):
        idx = top_idx[i].item()
        prob = top_probs[i].item()
        word = tgt_idx2word.get(idx, 'UNK')
        print(f"  {idx} ('{word}'): {prob:.6f}")

print("\n=== Testing Greedy Inference ===")
# Test greedy inference
with torch.no_grad():
    output = decoder.infer_greedy(src_idx, encoder_output)
    print(f"Generated tokens: {output[0].tolist()[:20]}")
    
    # Count token frequencies
    token_counts = {}
    for token in output[0].tolist():
        if token in token_counts:
            token_counts[token] += 1
        else:
            token_counts[token] = 1
    
    print(f"\nToken frequency in output:")
    for token, count in sorted(token_counts.items(), key=lambda x: x[1], reverse=True)[:10]:
        word = tgt_idx2word.get(token, 'UNK')
        print(f"  {token} ('{word}'): {count} times")

print("\n=== Checking Model Parameters ===")
# Check if weights are reasonable
for name, param in decoder.named_parameters():
    if 'weight' in name:
        print(f"\n{name}:")
        print(f"  Mean: {param.mean().item():.6f}")
        print(f"  Std: {param.std().item():.6f}")
        print(f"  Max: {param.max().item():.6f}")
        print(f"  Min: {param.min().item():.6f}")
        if torch.isnan(param).any():
            print(f"  WARNING: Contains NaN values!")
        if torch.isinf(param).any():
            print(f"  WARNING: Contains Inf values!")