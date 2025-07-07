import os
import sys

project_root_path = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
sys.path.append(project_root_path)

import library.nn_architectures as nn_architectures
import torch
import torch.nn.functional as F

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# Test the attention mechanism
print("=== Testing Attention Mechanism ===")

# Create a simple attention layer
hidden_size = 256
heads = 8
seq_len = 10
attention = nn_architectures.TransformerAttention(seq_len, hidden_size, heads=heads, relposenc=False).to(device)

# Create test input
batch_size = 1
x = torch.randn(batch_size, seq_len, hidden_size).to(device)

print(f"Input stats:")
print(f"  Mean: {x.mean().item():.6f}, std: {x.std().item():.6f}")

# Manually compute attention to see what's happening
with torch.no_grad():
    # Project to Q, K, V
    Q = attention.Wq(x)
    K = attention.Wk(x)
    V = attention.Wv(x)
    
    print(f"\nAfter projection:")
    print(f"  Q - Mean: {Q.mean().item():.6f}, std: {Q.std().item():.6f}")
    print(f"  K - Mean: {K.mean().item():.6f}, std: {K.std().item():.6f}")
    print(f"  V - Mean: {V.mean().item():.6f}, std: {V.std().item():.6f}")
    
    # Reshape for multi-head attention
    from einops import rearrange
    Q = rearrange(Q, 'b l (h d) -> b h l d', h=heads)
    K = rearrange(K, 'b l (h d) -> b h l d', h=heads)
    V = rearrange(V, 'b l (h d) -> b h l d', h=heads)
    
    # Scale Q
    head_dim = hidden_size // heads
    Q_scaled = Q / (head_dim ** 0.5)
    
    print(f"\nAfter scaling Q by sqrt(d_k)={head_dim**0.5:.2f}:")
    print(f"  Q_scaled - Mean: {Q_scaled.mean().item():.6f}, std: {Q_scaled.std().item():.6f}")
    
    # Compute attention scores
    scores = torch.einsum('b h i d, b h j d -> b h i j', Q_scaled, K)
    
    print(f"\nAttention scores (before softmax):")
    print(f"  Mean: {scores.mean().item():.6f}, std: {scores.std().item():.6f}")
    print(f"  Max: {scores.max().item():.6f}, Min: {scores.min().item():.6f}")
    
    # Apply softmax
    attention_weights = F.softmax(scores, dim=-1)
    
    print(f"\nAttention weights (after softmax):")
    print(f"  Mean: {attention_weights.mean().item():.6f}, std: {attention_weights.std().item():.6f}")
    print(f"  Max: {attention_weights.max().item():.6f}, Min: {attention_weights.min().item():.6f}")
    
    # Compute attention output
    attention_output = torch.einsum('b h q k, b h k d -> b h q d', attention_weights, V)
    attention_output = rearrange(attention_output, 'b h l d -> b l (h d)')
    
    print(f"\nAttention output (before Wo):")
    print(f"  Mean: {attention_output.mean().item():.6f}, std: {attention_output.std().item():.6f}")
    
    # Apply output projection
    final_output = attention.Wo(attention_output)
    
    print(f"\nFinal attention output:")
    print(f"  Mean: {final_output.mean().item():.6f}, std: {final_output.std().item():.6f}")
    print(f"  Max: {final_output.max().item():.6f}, Min: {final_output.min().item():.6f}")

print("\n=== Testing with Large Input ===")
# Test with larger input (simulating what happens after a few layers)
x_large = torch.randn(batch_size, seq_len, hidden_size).to(device) * 100

print(f"\nLarge input stats:")
print(f"  Mean: {x_large.mean().item():.6f}, std: {x_large.std().item():.6f}")

with torch.no_grad():
    output_large = attention(x_large, x_large, x_large)
    print(f"\nAttention output for large input:")
    print(f"  Mean: {output_large.mean().item():.6f}, std: {output_large.std().item():.6f}")
    print(f"  Max: {output_large.max().item():.6f}, Min: {output_large.min().item():.6f}")

print("\n=== Key Observations ===")
print("1. When inputs have large magnitudes, attention outputs also become large")
print("2. Even though attention weights sum to 1, the VALUES being weighted can be huge")
print("3. This causes the weighted sum (attention output) to also be large")
print("4. The output projection (Wo) can further amplify these values")