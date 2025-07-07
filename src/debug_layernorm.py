import os
import sys

project_root_path = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
sys.path.append(project_root_path)

import torch
import torch.nn as nn

# Test layer normalization behavior
hidden_size = 256
layernorm = nn.LayerNorm(hidden_size)

print("=== Testing Layer Normalization Behavior ===")

# Test 1: Normal input
x = torch.randn(1, 10, hidden_size)
print(f"\nTest 1 - Normal input:")
print(f"  Input mean: {x.mean().item():.6f}, std: {x.std().item():.6f}")
y = layernorm(x)
print(f"  Output mean: {y.mean().item():.6f}, std: {y.std().item():.6f}")

# Test 2: Large input
x = torch.randn(1, 10, hidden_size) * 1000
print(f"\nTest 2 - Large input (scaled by 1000):")
print(f"  Input mean: {x.mean().item():.6f}, std: {x.std().item():.6f}")
y = layernorm(x)
print(f"  Output mean: {y.mean().item():.6f}, std: {y.std().item():.6f}")

# Test 3: Very large input
x = torch.randn(1, 10, hidden_size) * 10000
print(f"\nTest 3 - Very large input (scaled by 10000):")
print(f"  Input mean: {x.mean().item():.6f}, std: {x.std().item():.6f}")
y = layernorm(x)
print(f"  Output mean: {y.mean().item():.6f}, std: {y.std().item():.6f}")

# Test what happens with residual connections
print("\n=== Testing Residual Connection Behavior ===")

# Simulate what happens in a transformer layer
input_tensor = torch.randn(1, 10, hidden_size)
attention_output = torch.randn(1, 10, hidden_size) * 100  # Large attention output

print(f"\nOriginal input stats:")
print(f"  Mean: {input_tensor.mean().item():.6f}, std: {input_tensor.std().item():.6f}")
print(f"Attention output stats:")
print(f"  Mean: {attention_output.mean().item():.6f}, std: {attention_output.std().item():.6f}")

# Pre-norm: normalize first, then add residual
normalized_input = layernorm(input_tensor)
residual_prenorm = attention_output + input_tensor
print(f"\nPre-norm residual stats:")
print(f"  Mean: {residual_prenorm.mean().item():.6f}, std: {residual_prenorm.std().item():.6f}")

# Post-norm: add residual first, then normalize
residual_postnorm = attention_output + input_tensor
normalized_postnorm = layernorm(residual_postnorm)
print(f"\nPost-norm output stats:")
print(f"  Mean: {normalized_postnorm.mean().item():.6f}, std: {normalized_postnorm.std().item():.6f}")

print("\n=== Understanding the Problem ===")
print("In pre-norm architecture:")
print("1. We normalize the input BEFORE feeding to attention/FFN")
print("2. Then we add the (potentially large) attention output to the original (unnormalized) input")
print("3. This can cause activation explosion if the attention outputs are large")
print("4. The explosion compounds through layers as each layer's output becomes the next layer's input")