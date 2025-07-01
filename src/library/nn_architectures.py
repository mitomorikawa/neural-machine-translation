""" 
This module contains neural network architectures for encoding and decoding sequences.
"""

import torch
import torch.nn as nn
import torch.nn.functional as F

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
class EncoderRNN(nn.Module):
    def __init__(self, input_size, hidden_size, dropout_p=0.1, padding_idx=2):
        super(EncoderRNN, self).__init__()
        self.hidden_size = hidden_size

        self.embedding = nn.Embedding(input_size, hidden_size, padding_idx=padding_idx)
        self.gru = nn.GRU(hidden_size, hidden_size, batch_first=True, bidirectional=True)
        self.dropout = nn.Dropout(dropout_p)

    def forward(self, input):
        embedded = self.dropout(self.embedding(input))
        output, hidden = self.gru(embedded)
        # For bidirectional GRU, concatenate the final hidden states
        # hidden is of shape (num_layers * num_directions, batch, hidden_size)
        hidden = torch.cat((hidden[0:1], hidden[1:2]), dim=2)  # (1, batch, hidden_size*2)
        return output, hidden
    
class BahdanauAttention(nn.Module):
    def __init__(self, hidden_size):
        super(BahdanauAttention, self).__init__()
        self.Wa = nn.Linear(hidden_size, hidden_size)
        self.Ua = nn.Linear(hidden_size * 2, hidden_size)  # *2 for bidirectional encoder
        self.Va = nn.Linear(hidden_size, 1)

    def forward(self, query, keys):
        scores = self.Va(torch.tanh(self.Wa(query) + self.Ua(keys)))
        scores = scores.squeeze(2).unsqueeze(1)

        weights = F.softmax(scores, dim=-1)
        context = torch.bmm(weights, keys)

        return context, weights

class AttnDecoderRNN(nn.Module):
    def __init__(self, hidden_size, output_size, max_len, dropout_p=0.1):
        super(AttnDecoderRNN, self).__init__()
        self.embedding = nn.Embedding(output_size, hidden_size)
        self.hidden_size = hidden_size
        self.output_size = output_size
        self.attention = BahdanauAttention(hidden_size)
        self.gru = nn.GRU(3 * hidden_size, hidden_size, batch_first=True)  # 3*hidden_size: embedding + bidirectional context
        self.out = nn.Linear(hidden_size, output_size)
        self.dropout = nn.Dropout(dropout_p)
        self.max_len = max_len

    def forward(self, encoder_outputs, encoder_hidden, target_tensor=None, greedy=True, beam_width=5):
        """
        args
        encoder_outputs (torch.Tensor): The outputs from the encoder.
        encoder_hidden (torch.Tensor): The hidden state from the encoder.
        target_tensor (torch.Tensor, optional): The target tensor for teacher forcing. If None, uses greedy decoding.
        greedy (bool): If True, uses greedy decoding. If False, uses beam search decoding.
        beam_width (int): The width of the beam for beam search decoding. Only used if greedy is False.

        """
        batch_size = encoder_outputs.size(0)
        decoder_input = torch.empty(batch_size, 1, dtype=torch.long, device=device).fill_(0)
       
        # Transform encoder hidden state from (1, batch, hidden_size*2) to (1, batch, hidden_size)
        decoder_hidden = nn.Linear(encoder_hidden.size(2), self.hidden_size).to(device)(encoder_hidden)
        decoder_outputs = []
        attentions = []

        if greedy: # Greedy decoding
            for i in range(self.max_len):
                decoder_output, decoder_hidden, attn_weights = self.forward_step(
                    decoder_input, decoder_hidden, encoder_outputs
                )
                decoder_outputs.append(decoder_output)
                attentions.append(attn_weights)

                if target_tensor is not None:
                    # Teacher forcing: Feed the target as the next input
                    decoder_input = target_tensor[:, i].unsqueeze(1) # Teacher forcing
                else:
                    # Without teacher forcing: use its own predictions as the next input
                    _, topi = decoder_output.topk(1)
                    decoder_input = topi.squeeze(-1).detach()  # detach from history as input
            decoder_outputs = torch.cat(decoder_outputs, dim=1)
            decoder_outputs = F.log_softmax(decoder_outputs, dim=-1)
            attentions = torch.cat(attentions, dim=1)
            return decoder_outputs, decoder_hidden, attentions
        
        else: # Beam search decoding. Assume batch_size = 1, thus  
            # Initialize beam with start token
            sequences = []
            for _ in range(beam_width):
                sequences.append({
                    'inputs': [decoder_input.clone()],
                    'outputs': [],
                    'hidden': decoder_hidden.clone(),
                    'attentions': [],
                    'score': 0.0
                })
            
            for i in range(self.max_len):
                all_candidates = []
                
                for s, seq in enumerate(sequences):
                    # Get predictions for current sequence
                    decoder_output, next_hidden, attn_weights = self.forward_step(
                        seq['inputs'][-1], seq['hidden'], encoder_outputs
                    )
                    decoder_output = F.log_softmax(decoder_output, dim=-1)
                    
                    # Get top k predictions. shape: (batch_size=1, 1, beam_width)
                    topk_scores, topk_indices = decoder_output.topk(beam_width)
                    
                    for k in range(beam_width):
                        # Create new candidate sequence
                        candidate = {
                            'inputs': seq['inputs'] + [topk_indices[:, :, k].squeeze(1).detach()],
                            'outputs': seq['outputs'] + [decoder_output.detach()],
                            'hidden': next_hidden.clone(),
                            'attentions': seq['attentions'] + [attn_weights.detach()],
                            'score': seq['score'] + topk_scores[0, 0, k].item()
                        }
                        all_candidates.append(candidate)
                
                # Sort by score (descending - higher is better for log probs)
                ordered = sorted(all_candidates, key=lambda x: x['score'], reverse=True)
                sequences = ordered[:beam_width]

                
                
            
            # Return the best sequence outputs
            best_sequence = sequences[0]
            decoder_outputs = torch.cat(best_sequence['outputs'], dim=1)
            attentions = torch.cat(best_sequence['attentions'], dim=1)
            
            return decoder_outputs, best_sequence['hidden'], attentions
            
    def forward_step(self, input, hidden, encoder_outputs):
        embedded =  self.dropout(self.embedding(input))
        if embedded.dim() == 2: 
            embedded = embedded.unsqueeze(1)
        query = hidden.permute(1, 0, 2)
        context, attn_weights = self.attention(query, encoder_outputs)
        input_gru = torch.cat((embedded, context), dim=-1)

        output, hidden = self.gru(input_gru, hidden)
        output = self.out(output)

        return output, hidden, attn_weights