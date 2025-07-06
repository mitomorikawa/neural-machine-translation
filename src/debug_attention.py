import os
import sys

project_root_path = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
sys.path.append(project_root_path)

import library.nn_architectures as nn_architectures
import torch

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# Create a simple attention layer
seq_len = 10
hidden_size = 256
attention = nn_architectures.TransformerAttention(seq_len, hidden_size, heads=8, mask=True)
attention = attention.to(device)
attention.eval()

# Create test input
batch_size = 1
test_input = torch.randn(batch_size, seq_len, hidden_size).to(device)

# Check if positional embedding is initialized
print(f"Positional embedding shape: {attention.positional_embedding.shape}")
print(f"Positional embedding requires grad: {attention.positional_embedding.requires_grad}")
print(f"First few values of positional embedding: {attention.positional_embedding[:5, :5]}")

# Test forward pass
with torch.no_grad():
    # Run attention without any modifications
    output = attention(test_input, test_input, test_input)
    print(f"\nOutput shape: {output.shape}")
    
    # Check if outputs at different positions are different
    print("\nChecking if outputs are position-dependent:")
    for i in range(min(5, seq_len)):
        print(f"Position {i} output norm: {output[0, i].norm().item():.4f}")
    
    # Check if all positions have identical output (which would be bad)
    all_same = True
    for i in range(1, seq_len):
        if not torch.allclose(output[0, 0], output[0, i], atol=1e-5):
            all_same = False
            break
    
    print(f"\nAll positions have identical output: {all_same}")
    
    # Test the relative positional encoding directly
    query = test_input.view(batch_size, seq_len, 8, 32).permute(0, 2, 1, 3)  # b h l d
    posenc = attention.relativePositionalEncoding(query)
    print(f"\nRelative positional encoding output shape: {posenc.shape}")
    print(f"Relative positional encoding contains non-zero values: {(posenc != 0).any().item()}")