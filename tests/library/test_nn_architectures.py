import os
import sys

project_root_path = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

sys.path.append(project_root_path)

import src.library.nn_architectures as nn_architectures
import torch

def test_BahdanauEncoder():
    batch_size = 32
    seq_len = 10
    vocab_size = 5000
    hidden_size = 256
    
    encoder = nn_architectures.RNNEncoder(input_size=vocab_size, hidden_size=hidden_size)
    
    input_tensor = torch.randint(0, vocab_size, (batch_size, seq_len))
    
    output, hidden = encoder(input_tensor)
    
    assert output.shape == (batch_size, seq_len, hidden_size * 2), f"Expected output shape {(batch_size, seq_len, hidden_size * 2)}, got {output.shape}"
    assert hidden.shape == (1, batch_size, hidden_size * 2), f"Expected hidden shape {(1, batch_size, hidden_size * 2)}, got {hidden.shape}"
    
    print("test_BahdanauEncoder passed!")

def test_BahdanauDecoder():
    batch_size = 15
    encoder_seq_len = 10
    vocab_size = 5
    hidden_size = 1
    max_len = 10
    
    encoder = nn_architectures.RNNEncoder(input_size=vocab_size, hidden_size=hidden_size)
    decoder = nn_architectures.RNNDecoder(hidden_size=hidden_size, output_size=vocab_size, max_len=max_len)
    
    encoder_input = torch.randint(0, vocab_size, (batch_size, encoder_seq_len))
    encoder_outputs, encoder_hidden = encoder(encoder_input)
    
    target_tensor = torch.randint(0, vocab_size, (batch_size, max_len))
    decoder_outputs, decoder_hidden, attentions = decoder(encoder_outputs, encoder_hidden, greedy = True)
    
    assert decoder_outputs.shape == (batch_size, max_len, vocab_size), f"Expected decoder outputs shape {(batch_size, max_len, vocab_size)}, got {decoder_outputs.shape}"
    assert decoder_hidden.shape == (1, batch_size, hidden_size), f"Expected decoder hidden shape {(1, batch_size, hidden_size)}, got {decoder_hidden.shape}"
    assert attentions.shape == (batch_size, max_len, encoder_seq_len), f"Expected attentions shape {(batch_size, max_len, encoder_seq_len)}, got {attentions.shape}"
    
    batch_size = 1
    beam_width = 3
    encoder_input = torch.randint(0, vocab_size, (batch_size, encoder_seq_len))
    encoder_outputs, encoder_hidden = encoder(encoder_input)
    decoder_outputs_no_teacher, _, _ = decoder(encoder_outputs, encoder_hidden, target_tensor=None)
    assert decoder_outputs_no_teacher.shape == (batch_size, max_len, vocab_size), f"Expected decoder outputs shape without teacher forcing {(batch_size, max_len, vocab_size)}, got {decoder_outputs_no_teacher.shape}"

    # Test beam search decoding
    beam_width = 3
    batch_size = 1  # Beam search typically works with batch size of 1
    
    decoder_outputs_beam, _, _ = decoder(encoder_outputs, encoder_hidden, target_tensor=None, beam_width=beam_width)
    assert decoder_outputs_beam.shape == (batch_size, max_len, vocab_size), f"Expected decoder outputs shape for beam search {(batch_size, max_len, vocab_size)}, got {decoder_outputs_beam.shape}"

    
    print("test_BahdanauDecoder passed!")

def test_TransormerAttention():
    batch_size = 2
    seq_len = 5
    hidden_size = 8
    num_heads = 2
    
    attention_layer = nn_architectures.TransformerAttention(seq_len, hidden_size, heads=num_heads)
    
    query = torch.randn(batch_size, seq_len, hidden_size)
    key = torch.randn(batch_size, seq_len, hidden_size)
    value = torch.randn(batch_size, seq_len, hidden_size)
    query_reshaped = query.view(batch_size, seq_len, num_heads, hidden_size // num_heads).transpose(1, 2)

    query_pos = attention_layer.relativePositionalEncoding(query_reshaped)
    if query_pos.shape != (batch_size, num_heads, seq_len, seq_len):
        raise ValueError(f"Expected query_pos shape {(batch_size, num_heads, seq_len, seq_len)}, got {query_pos.shape}")

    output = attention_layer(query, key, value)
    
    if output.shape != (batch_size, seq_len, hidden_size):
        raise ValueError(f"Expected output shape {(batch_size, seq_len, hidden_size)}, got {output.shape}")
    
def test_Transormer():
    batch_size = 2
    seq_len = 5
    hidden_size = 8
    num_heads = 2
    encoder_input = torch.randint(0, 50, (batch_size, seq_len))
    encoder = nn_architectures.TransformerEncoder(50, hidden_size, seq_len,heads=num_heads)

    encoder_output,encoder_hidden = encoder(encoder_input)
    if encoder_output.shape != (batch_size, seq_len, hidden_size):
        raise ValueError(f"Expected encoder output shape {(batch_size, seq_len, hidden_size)}, got {encoder_output.shape}")
    print("test_TransormerAttention passed!")
    output_size = 60
    tgt_seq_len = 6
    decoder_input = torch.randint(0, output_size, (batch_size, tgt_seq_len))
    decoder = nn_architectures.TransformerDecoder(hidden_size, output_size, tgt_seq_len)
    decoder_output, _,_ = decoder(encoder_output,encoder_hidden,decoder_input,encoder_input)
    if decoder_output.shape!=(batch_size, tgt_seq_len, output_size):
        raise ValueError(f"Expected decoder output shape {(batch_size, tgt_seq_len, output_size)}, got {decoder_output.shape}")
    print("test_Transormer passed!")

def test_TransformerPaddingMask():
    batch_size = 2
    seq_len = 10
    hidden_size = 16
    num_heads = 4
    vocab_size = 50
    output_size = 60
    tgt_seq_len = 8
    
    # Create encoder input with padding tokens (padding_idx=2)
    encoder_input = torch.randint(3, vocab_size, (batch_size, seq_len))
    # Add some padding tokens at the end
    encoder_input[:-1, -3:] = 2  # Last 3 tokens are padding
    encoder_input[-1, -2:] = 2  # Last 2 tokens are padding for the last sequence
    
    # Create decoder input with padding tokens
    decoder_input = torch.randint(3, output_size, (batch_size, tgt_seq_len))
    # Add some padding tokens at the end
    decoder_input[1:, -2:] = 2  # Last 2 tokens are padding
    decoder_input[0, -5:] = 2  # Last token is padding for the first sequence
    
    encoder = nn_architectures.TransformerEncoder(vocab_size, hidden_size, seq_len, heads=num_heads, num_layer=2)
    decoder = nn_architectures.TransformerDecoder(hidden_size, output_size, tgt_seq_len, heads=num_heads, num_layer=2)
    
    # Forward pass through encoder
    encoder_output, _ = encoder(encoder_input)
    
    # Forward pass through decoder with padding masks
    decoder_output, _, _ = decoder(encoder_output, None, decoder_input, encoder_input)
    
    assert decoder_output.shape == (batch_size, tgt_seq_len, output_size), f"Expected shape {(batch_size, tgt_seq_len, output_size)}, got {decoder_output.shape}"
    
    # Test that attention works correctly with padding
    # The output should be valid even with padding tokens
    assert not torch.isnan(decoder_output).any(), "Output contains NaN values"
    assert not torch.isinf(decoder_output).any(), "Output contains infinite values"
    
    print("test_TransformerPaddingMask passed!")

if __name__ == "__main__":
    test_BahdanauEncoder()
    test_BahdanauDecoder()
    test_TransormerAttention()
    test_Transormer()
    test_TransformerPaddingMask()
    print("All tests passed!")