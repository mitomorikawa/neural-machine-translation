import os
import sys

project_root_path = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
sys.path.append(project_root_path)

import library.nn_architectures as nn_architectures
import torch

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# Initialize models (untrained)
hidden_size = 256
encoder = nn_architectures.TransformerEncoder(9783, hidden_size, 55, num_layer=2)  # Fewer layers for debugging
decoder = nn_architectures.TransformerDecoder(hidden_size, 15532, 69, num_layer=2)
encoder = encoder.to(device)
decoder = decoder.to(device)
encoder.eval()
decoder.eval()

# Create test input
src_idx = torch.tensor([[3, 4, 5, 6, 1] + [2]*50]).to(device)
tgt_idx = torch.tensor([[0, 10, 20, 30, 1] + [2]*64]).to(device)

print("Testing with untrained model (should show different predictions per position):")
with torch.no_grad():
    # Encoder forward
    encoder_output, _ = encoder(src_idx)
    print(f"Encoder output shape: {encoder_output.shape}")
    
    # Check if encoder outputs are position-dependent
    print("\nEncoder outputs at different positions (norm):")
    for i in range(5):
        print(f"  Position {i}: {encoder_output[0, i].norm().item():.4f}")
    
    # Decoder forward
    decoder_output, _, _ = decoder(encoder_output, 0, tgt_idx, src_idx)
    print(f"\nDecoder output shape: {decoder_output.shape}")
    
    # Check predictions at different positions
    print("\nTop prediction at each position:")
    for i in range(5):
        probs = torch.exp(decoder_output[0, i, :])
        top_prob, top_idx = torch.max(probs, dim=0)
        print(f"  Position {i}: token {top_idx.item()} (prob: {top_prob.item():.4f})")
    
    # Check if all positions predict the same thing
    all_same = True
    first_pred = decoder_output[0, 0, :].argmax()
    for i in range(1, 5):
        if decoder_output[0, i, :].argmax() != first_pred:
            all_same = False
            break
    print(f"\nAll positions predict the same token: {all_same}")

# Now let's load the trained model and test again
print("\n" + "="*50)
print("Testing with trained model:")

encoder_path = "../models/transformer_encoder_20250706_040900_39"
decoder_path = "../models/transformer_decoder_20250706_040900_39"

encoder = nn_architectures.TransformerEncoder(9783, hidden_size, 55, num_layer=6)
decoder = nn_architectures.TransformerDecoder(hidden_size, 15532, 69, num_layer=6)
encoder.load_state_dict(torch.load(encoder_path, map_location=device, weights_only=True))
decoder.load_state_dict(torch.load(decoder_path, map_location=device, weights_only=True))
encoder = encoder.to(device)
decoder = decoder.to(device)
encoder.eval()
decoder.eval()

with torch.no_grad():
    # Encoder forward
    encoder_output, _ = encoder(src_idx)
    
    # Check if encoder outputs are position-dependent
    print("\nEncoder outputs at different positions (norm):")
    for i in range(5):
        print(f"  Position {i}: {encoder_output[0, i].norm().item():.4f}")
    
    # Decoder forward
    decoder_output, _, _ = decoder(encoder_output, 0, tgt_idx, src_idx)
    
    # Check predictions at different positions
    print("\nTop prediction at each position:")
    for i in range(5):
        probs = torch.exp(decoder_output[0, i, :])
        top_prob, top_idx = torch.max(probs, dim=0)
        print(f"  Position {i}: token {top_idx.item()} (prob: {top_prob.item():.4f})")
    
    # Check attention scores in first decoder layer
    print("\nChecking decoder self-attention in first layer:")
    # We'll need to modify the model to expose attention scores, so let's just check the output
    
    # Check if decoder embeddings are collapsing
    embeddings = decoder.embedding(tgt_idx)
    print(f"\nDecoder embedding norms:")
    for i in range(5):
        print(f"  Position {i} (token {tgt_idx[0, i].item()}): {embeddings[0, i].norm().item():.4f}")