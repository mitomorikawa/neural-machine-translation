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
    
    encoder = nn_architectures.EncoderRNN(input_size=vocab_size, hidden_size=hidden_size)
    
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
    
    encoder = nn_architectures.EncoderRNN(input_size=vocab_size, hidden_size=hidden_size)
    decoder = nn_architectures.AttnDecoderRNN(hidden_size=hidden_size, output_size=vocab_size, max_len=max_len)
    
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

if __name__ == "__main__":
    test_BahdanauEncoder()
    test_BahdanauDecoder()
    print("All tests passed!")

