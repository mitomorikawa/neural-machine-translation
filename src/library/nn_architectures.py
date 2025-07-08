""" 
This module contains neural network architectures for encoding and decoding sequences.
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from einops import rearrange

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
class RNNEncoder(nn.Module):
    def __init__(self, input_size, hidden_size, dropout_p=0.1, padding_idx=2):
        super(RNNEncoder, self).__init__()
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

class RNNDecoder(nn.Module):
    def __init__(self, hidden_size, output_size, max_len, dropout_p=0.1):
        super(RNNDecoder, self).__init__()
        self.embedding = nn.Embedding(output_size, hidden_size)
        self.hidden_size = hidden_size
        self.output_size = output_size
        self.attention = BahdanauAttention(hidden_size)
        self.gru = nn.GRU(3 * hidden_size, hidden_size, batch_first=True)  # 3*hidden_size: embedding + bidirectional context
        self.out = nn.Linear(hidden_size, output_size)
        self.dropout = nn.Dropout(dropout_p)
        self.max_len = max_len

    def forward(self, encoder_outputs, encoder_hidden, target_tensor=None, source_tensor=None, greedy=True, beam_width=5):
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
    
def absolutePositionalEncoding(batch_size, seq_len, hidden_size):
    """
    Generate absolute positional encoding for a sequence.
    
    Args:
        batch_size (int): Size of the batch.
        seq_len (int): Length of the sequence.
        hidden_size (int): Size of the hidden layer.
        
    Returns:
        torch.Tensor: Positional encoding tensor of shape (batch_size, seq_len, hidden_size).
    """
    pe = torch.zeros(seq_len, hidden_size)
    position = torch.arange(seq_len, dtype=torch.float).unsqueeze(1)  # (seq_len, 1)
    div_term = torch.exp(torch.arange(0, hidden_size, 2, dtype=torch.float) * -(torch.log(torch.tensor(10000.0)) / hidden_size))  # (hidden_size // 2)
    pe[:, 0::2] = torch.sin(position * div_term)  # even indices
    if hidden_size % 2 == 0:
        pe[:, 1::2] = torch.cos(position * div_term)  # odd indices
    else:
        pe[:, 1::2] = torch.cos(position * div_term[:-1])  # handle odd hidden_size
    # Expand to batch size
    pe = pe.unsqueeze(0).expand(batch_size, -1, -1)
    return pe.to(device)
    
    
class TransformerAttention(nn.Module):
    def __init__(self, query_seq_len, hidden_size, heads=8, mask=False, cross_attention=False, relposenc=True):
        super(TransformerAttention, self).__init__()
        self.Wq = nn.Linear(hidden_size, hidden_size)
        self.Wk = nn.Linear(hidden_size, hidden_size)
        self.Wv = nn.Linear(hidden_size, hidden_size)
        self.hidden_size = hidden_size
        self.heads = heads
        self.head_dim = hidden_size // heads
        self.softmax = nn.Softmax(dim=-1)
        self.Wo = nn.Linear(hidden_size, hidden_size)
        self.mask = mask
        self.cross_attention = cross_attention
        self.query_seq_len = query_seq_len
        # Relative position embeddings: max_relative_position * 2 + 1 (for negative, zero, and positive positions)
        max_relative_position = query_seq_len - 1
        self.relative_positions_num = 2 * max_relative_position + 1
        self.positional_embedding = torch.nn.Parameter(torch.zeros(self.relative_positions_num, self.head_dim))
        # Initialize with Xavier/Glorot uniform
        nn.init.xavier_uniform_(self.positional_embedding, gain=1.0)
        self.relposenc= relposenc
        
    def forward(self, query, key, value, padding_mask=None):
        """
        Compute the attention scores.
        Args:
            query (torch.Tensor): Query tensor of shape (batch_size, seq_len, hidden_size).
            key (torch.Tensor): Key tensor of shape (batch_size, seq_len, hidden_size).
            value (torch.Tensor): Value tensor of shape (batch_size, seq_len, hidden_size).
            padding_mask (torch.Tensor): padding mask of shape (b, h, query_seq_len, key_seq_len). Every element is 0 or -inf.
        Returns:
            torch.Tensor: attention output tensor of shape (batch_size, seq_len, hidden_size).
        
        """
        query = self.Wq(query)
        key = self.Wk(key)
        value = self.Wv(value)
        batch_size = query.shape[0]
        
        query, key, value = map(lambda x: rearrange(x, 'b l (h d) -> b h l d', h=self.heads), (query, key, value))
        query /= (self.head_dim ** 0.5)
        attention = torch.einsum('b h i d, b h j d -> b h i j', query, key)

        if (not self.cross_attention) and self.relposenc:
            positional_query = self.relativePositionalEncoding(query)
            attention += positional_query
        if self.mask:
            mask_tensor = torch.triu(torch.full((batch_size, self.heads, self.query_seq_len, self.query_seq_len), float('-inf'), device=query.device), diagonal=1)
            attention += mask_tensor
        if padding_mask is not None:
            attention += padding_mask
        attention_score = self.softmax(attention)
        attention_output = torch.einsum('b h q k, b h k d -> b h q d', attention_score, value)
        attention_output = rearrange(attention_output, 'b h l d -> b l (h d)')
        attention_output = self.Wo(attention_output)
        return attention_output

    def relativePositionalEncoding(self, query): 
        """ 
        Compute relative positional encoding for self-attention.

        Args:
            query (torch.Tensor): Query tensor of shape (batch_size, heads, seq_len, head_dim).
        returns:
            torch.Tensor: Relative position bias of shape (batch_size, heads, seq_len, seq_len).
        """
        batch_size, n_heads, seq_len, d_head = query.shape
        
        # Create relative position matrix
        # relative_positions[i, j] = j - i (position of key relative to query)
        positions = torch.arange(seq_len, device=query.device)
        relative_positions = positions.unsqueeze(0) - positions.unsqueeze(1)  # (seq_len, seq_len)
        
        # Clip relative positions to max range and shift to positive indices
        max_relative_position = self.query_seq_len - 1
        relative_positions = relative_positions.clamp(-max_relative_position, max_relative_position)
        relative_positions = relative_positions + max_relative_position  # shift to [0, 2*max_relative_position]
        
        # Get embeddings for each relative position
        rel_embeddings = self.positional_embedding[relative_positions]  # (seq_len, seq_len, head_dim)
        
        # Compute attention bias: query @ relative_position_embeddings
        # query: (batch_size, heads, seq_len, head_dim)
        # rel_embeddings: (seq_len, seq_len, head_dim)
        # result: (batch_size, heads, seq_len, seq_len)
        attention_bias = torch.einsum('bhqd,qkd->bhqk', query, rel_embeddings)
        
        return attention_bias
    
class TransformerEncoderLayer(nn.Module):
    """ 
    One layer of transformer encoder.
    Attributes:
        attention (TransformerAttention): Multi-head attention layer.
        layerNorm1 (nn.LayerNorm): Layer normalization after attention.
        linear1 (nn.Linear): First linear layer in the feedforward network.
        relu (nn.ReLU): Activation function.
        linear2 (nn.Linear): Second linear layer in the feedforward network.
    """
    def __init__(self, src_seq_len, hidden_size, heads, relposenc, dropout_p=0.1):
        super(TransformerEncoderLayer, self).__init__()
        self.attention = TransformerAttention(src_seq_len, hidden_size, heads=heads, relposenc=relposenc)
        self.layerNorm1 = nn.LayerNorm(hidden_size)
        self.layerNorm2 = nn.LayerNorm(hidden_size)
        self.linear1 = nn.Linear(hidden_size, 4*hidden_size)
        self.relu = nn.ReLU()
        self.linear2 = nn.Linear(4*hidden_size, hidden_size)
        self.dropout1 = nn.Dropout(dropout_p)
        self.dropout2 = nn.Dropout(dropout_p)

    def forward(self, encoder_layer_input, padding_mask=None):
        """
        Args:
            encoder_layer_input (torch.Tensor): Input tensor of shape (batch_size, seq_len, hidden_size).
            padding_mask (torch.Tensor): Padding mask of shape (batch_size, num_heads, seq_len, seq_len).
        
        Returns:
            torch.Tensor: Encoded output tensor of shape (batch_size, seq_len, hidden_size).
        """

        # Multihead attention
        attention_output = self.attention(encoder_layer_input, encoder_layer_input, encoder_layer_input, padding_mask)
        attention_output = self.dropout1(attention_output)
        # Layernorm and add. 
        attention_output = self.layerNorm1(attention_output + encoder_layer_input)
        # Feedforward network
        ff_output = self.linear2(self.relu(self.linear1(attention_output)))
        ff_output = self.dropout2(ff_output)
        # Layernorm and add. Ciyld add dropout here
        output = self.layerNorm2(ff_output + attention_output)
        return output
    
class TransformerEncoder(nn.Module):
    """ 
        Encoder for the Transformer model.
        Attributes:
            - input_size (int): Size of the input vocabulary.
            - hidden_size (int): Size of the hidden layer.
            - heads (int): Number of attention heads.
            - num_layer (int): Number of transformer encoder layers.
            - dropout_p (float): Dropout probability.
        """
    def __init__(self, input_size, hidden_size, src_seq_len, heads=8, num_layer=6, dropout_p=0.1, relposenc=True):
        super(TransformerEncoder, self).__init__()
        self.embedding = nn.Embedding(input_size, hidden_size, padding_idx=2)
        self.dropout = nn.Dropout(dropout_p)
        self.num_layer = num_layer
        self.hidden_size = hidden_size
        self.src_seq_len = src_seq_len
        self.encoderlayers = nn.ModuleList([TransformerEncoderLayer(src_seq_len, hidden_size, heads, relposenc) for _ in range(num_layer)])
        self.relposenc = relposenc
        # Initialize weights
        self._init_weights()
        
    def forward(self, encoder_input):
        """
        Args:
            encoder_input (torch.Tensor): Input tensor of shape (batch_size, seq_len).
        
        Returns:
            torch.Tensor: Encoded output tensor of shape (batch_size, seq_len, hidden_size).
            0: for consistency with RNNEncoder
        """
        # Create padding mask from input (assuming padding_idx=2)
        padding_mask = (encoder_input == 2).unsqueeze(1).unsqueeze(1)  # (batch_size, 1, 1, seq_len)
        padding_mask = padding_mask.expand(-1, self.encoderlayers[0].attention.heads, encoder_input.size(1), -1)  # (batch_size, heads, seq_len, seq_len)
        padding_mask = padding_mask.float().masked_fill(padding_mask, float('-inf'))

        embedding = self.embedding(encoder_input)
        # Scale embeddings by sqrt(d_model) as in the paper
        embedding = embedding * (self.hidden_size ** 0.5)
        embedding = self.dropout(embedding)
        batch_size = embedding.size(0)
        if not self.relposenc:
            # If relative positional encoding is not used, create absolute positional encoding
            pos_encoding = absolutePositionalEncoding(batch_size, self.src_seq_len, self.hidden_size)
            embedding += pos_encoding# Add absolute positional encoding
        for i in range(self.num_layer):
            embedding = self.encoderlayers[i](embedding, padding_mask)
        return embedding,0
    
    def _init_weights(self):
        # Initialize embeddings with smaller std
        nn.init.normal_(self.embedding.weight, mean=0.0, std=self.hidden_size ** -0.5)
    
class TransformerDecoderLayer(nn.Module):
    """ 
    One layer of transformer decoder.
    Attributes:
        attention1 (TransformerAttention): Multi-head attention layer for the first attention.
        attention2 (TransformerAttention): Multi-head attention layer for the second attention.
        layerNorm1 (nn.LayerNorm): Layer normalization after the first attention.
        layerNorm2 (nn.LayerNorm): Layer normalization after the second attention.
        layerNorm3 (nn.LayerNorm): Layer normalization after the feedforward network.
        linear1 (nn.Linear): First linear layer in the feedforward network.
        relu (nn.ReLU): Activation function.
        linear2 (nn.Linear): Second linear layer in the feedforward network.
    """
    def __init__(self, tgt_seq_len, hidden_size, heads, relposenc, dropout_p=0.1):
        super(TransformerDecoderLayer, self).__init__()
        self.attention1 = TransformerAttention(tgt_seq_len, hidden_size, heads=heads, mask=True, relposenc=relposenc)
        self.attention2 = TransformerAttention(tgt_seq_len, hidden_size, heads=heads, cross_attention=True,relposenc=relposenc)
        self.layerNorm1 = nn.LayerNorm(hidden_size)
        self.layerNorm2 = nn.LayerNorm(hidden_size)
        self.layerNorm3 = nn.LayerNorm(hidden_size)
        self.linear1 = nn.Linear(hidden_size, 4*hidden_size)
        self.relu = nn.ReLU()
        self.linear2 = nn.Linear(4*hidden_size, hidden_size)
        self.dropout1 = nn.Dropout(dropout_p)
        self.dropout2 = nn.Dropout(dropout_p)
        self.dropout3 = nn.Dropout(dropout_p)

        
    def forward(self, decoder_layer_input, encoder_output, self_padding_mask=None, cross_padding_mask=None):
        """
        Args:
            decoder_layer_input (torch.Tensor): Input tensor of shape (batch_size, seq_len, hidden_size).
            encoder_output (torch.Tensor): Encoder output tensor of shape (batch_size, seq_len, hidden_size).
            self_padding_mask (torch.Tensor): Padding mask for self-attention of shape (batch_size, num_heads, tgt_seq_len, tgt_seq_len).
            cross_padding_mask (torch.Tensor): Padding mask for cross-attention of shape (batch_size, num_heads, tgt_seq_len, src_seq_len).
        
        Returns:
            torch.Tensor: Decoded output tensor of shape (batch_size, seq_len, hidden_size).
        """
        # Masked multihead attention
        attention1_output = self.attention1(decoder_layer_input, decoder_layer_input, decoder_layer_input, self_padding_mask)
        attention1_output = self.dropout1(attention1_output)
        # Add and Layer Normalisation
        attention1_output = self.layerNorm1(attention1_output + decoder_layer_input)
        # Cross attention (no mask needed)
        attention2_output = self.attention2(attention1_output, encoder_output, encoder_output, cross_padding_mask)
        attention2_output = self.dropout2(attention2_output)
        # Add and Layer Normalisation
        attention2_output = self.layerNorm2(attention2_output + attention1_output)
        # Feedforward network
        ff_output = self.linear2(self.relu(self.linear1(attention2_output)))
        ff_output = self.dropout3(ff_output)
        # Add and Layer Normalisation
        output = self.layerNorm3(ff_output + attention2_output)
        return output

class TransformerDecoder(nn.Module):
    """ 
    Decoder for the Transformer model.
    Attributes:
        - hidden_size (int): Size of the hidden layer.
        - output_size (int): Size of the output vocabulary.
        - heads (int): Number of attention heads.
        - num_layer (int): Number of transformer decoder layers.
        - dropout_p (float): Dropout probability.

    """
    def __init__(self, hidden_size, output_size, tgt_seq_len, heads=8, num_layer=6, dropout_p=0.1, relposenc=True): 
        super(TransformerDecoder, self).__init__()
        self.embedding = nn.Embedding(output_size, hidden_size, padding_idx=2)
        self.dropout = nn.Dropout(dropout_p)
        self.num_layer = num_layer
        self.hidden_size = hidden_size
        self.decoderlayers = nn.ModuleList([TransformerDecoderLayer(tgt_seq_len, hidden_size, heads=heads, relposenc=relposenc) for _ in range(num_layer)])
        self.output = nn.Linear(hidden_size, output_size)
        self.tgt_seq_len = tgt_seq_len
        self.relposenc = relposenc
        # Initialize weights
        self._init_weights()

    def forward(self, encoder_outputs, encoder_hidden, decoder_input, encoder_input=None, greedy=True, beam_width=5):
        """ 
        Args:
            encoder_outputs (torch.Tensor): Encoder output tensor of shape (batch_size, seq_len, hidden_size).
            encoder_hidden (any): For consistency with Bahdanau decoder
            decoder_input (torch.Tensor): Input tensor of shape (batch_size, seq_len).
            encoder_input (torch.Tensor): Encoder input tensor of shape (batch_size, seq_len) for creating padding masks.
        Returns:
            torch.Tensor: Decoded output tensor of shape (batch_size, seq_len, output_size).
            0: For consistency with RNNDecoder
        """        
        num_heads = self.decoderlayers[0].attention1.heads
        
        # Create self-attention padding mask for decoder (assuming padding_idx=2)
        self_padding_mask = (decoder_input == 2).unsqueeze(1).unsqueeze(1)  # (batch_size, 1, 1, tgt_seq_len)
        self_padding_mask = self_padding_mask.expand(-1, num_heads, self.tgt_seq_len, -1)  # (batch_size, heads, tgt_seq_len, tgt_seq_len)
        self_padding_mask = self_padding_mask.float().masked_fill(self_padding_mask, float('-inf'))
        
        # Create cross-attention padding mask based on encoder input
        if encoder_input is not None:
            encoder_padding = (encoder_input == 2).unsqueeze(1).unsqueeze(1)  # (batch_size, 1, 1, src_seq_len)
            cross_padding_mask = encoder_padding.expand(-1, num_heads, self.tgt_seq_len, -1)  # (batch_size, heads, tgt_seq_len, src_seq_len)
            cross_padding_mask = cross_padding_mask.float().masked_fill(cross_padding_mask, float('-inf'))
        else:
            cross_padding_mask = None
        
        embedding = self.embedding(decoder_input)
        # Scale embeddings by sqrt(d_model) as in the paper
        embedding = embedding * (self.hidden_size ** 0.5)
        embedding = self.dropout(embedding)
        batch_size = embedding.size(0)
        if not self.relposenc:
            # If relative positional encoding is not used, create absolute positional encoding
            pos_encoding = absolutePositionalEncoding(batch_size, self.tgt_seq_len, self.hidden_size)
            embedding += pos_encoding
        for i in range(self.num_layer):
            embedding = self.decoderlayers[i](embedding, encoder_outputs, self_padding_mask, cross_padding_mask)
        output = self.output(embedding)
        return output, 0,0
    
    def _init_weights(self):
        # Initialize embeddings with smaller std
        nn.init.normal_(self.embedding.weight, mean=0.0, std=self.hidden_size ** -0.5)
        # Initialize output projection
        nn.init.xavier_uniform_(self.output.weight)
        nn.init.constant_(self.output.bias, 0.0)
    
    def infer_greedy(self, encoder_input, encoder_output):
        """
        Greedy decoding for inference.
        
        Args:
            encoder_input (torch.Tensor): Encoder input tensor of shape (1, seq_len).
            encoder_output (torch.Tensor): Encoder output tensor of shape (1, seq_len, hidden_size).
        
        Returns:
            torch.Tensor: Decoded output tensor of shape (1, seq_len).
        """
        # Initialize with SOS token
        tokens = [0]
        
        # Create decoder input filled with padding
        decoder_input = torch.full((1, self.tgt_seq_len), 2, dtype=torch.long, device=encoder_output.device)
        decoder_input[0, 0] = 0  # SOS token
        
        for i in range(1, self.tgt_seq_len):
            # Get predictions
            decoder_output, _, _ = self.forward(encoder_output, 0, decoder_input, encoder_input)
            
            # The decoder output at position i-1 predicts the token at position i
            # So to get the prediction for position i, we look at output from position i-1
            next_token = decoder_output[0, i-1, :].argmax().item()
            
            # Add to sequence
            tokens.append(next_token)
            decoder_input[0, i] = next_token
            
            # Stop if we hit EOS token
            if next_token == 1:  # EOS token
                print("<EOS>")
        
        # Return the result
        return decoder_input
    
    def infer(self, encoder_input, encoder_output, beam_width=5):
        """
        Args:
            encoder_input (torch.Tensor): Encoder input tensor of shape (1, seq_len).
            encoder_output (torch.Tensor): Encoder output tensor of shape (1, seq_len, hidden_size).
            beam_width (int): Width of the beam for beam search decoding.
        
        Returns:
            torch.Tensor: Decoded output tensor of shape (1, seq_len).
        """
        
        # Initialize beam with just the start token
        sequences = []
        for _ in range(beam_width):
            sequences.append({
                'tokens': [0],  # Start with SOS token
                'score': 0.0
            })
        
        complete_sequences = []
        
        for i in range(self.tgt_seq_len - 1):
            all_candidates = []
            
            for seq in sequences:
                # Skip if sequence already ended
                if seq['tokens'][-1] == 1:  # EOS token
                    complete_sequences.append(seq)
                    continue
                
                # Create decoder input from current sequence
                current_length = len(seq['tokens'])
                decoder_input = torch.full((1, self.tgt_seq_len), 2, dtype=torch.long, device=encoder_output.device)
                decoder_input[0, :current_length] = torch.tensor(seq['tokens'], device=encoder_output.device)
                
                # Get predictions for current sequence
                decoder_output, _, _ = self.forward(encoder_output, 0, decoder_input, encoder_input)
                
                # Get top k predictions at the current position (where next token should be generated)
                # Make sure we don't go out of bounds
                if current_length >= self.tgt_seq_len:
                    continue  # Skip if sequence is already at max length
                topk_scores, topk_indices = decoder_output[0, current_length, :].topk(beam_width)
                
                for k in range(beam_width):
                    # Create new candidate
                    candidate = {
                        'tokens': seq['tokens'] + [topk_indices[k].item()],
                        'score': seq['score'] + topk_scores[k].item()
                    }
                    
                    # Check if this candidate ends with EOS
                    if topk_indices[k] == 1:  # EOS token
                        complete_sequences.append(candidate)
                    else:
                        all_candidates.append(candidate)
            
            # Select top beam_width candidates
            if all_candidates:
                sequences = sorted(all_candidates, key=lambda x: x['score'], reverse=True)[:beam_width]
            else:
                break  # All sequences have ended
            
            # Early stopping if we have enough complete sequences
            if len(complete_sequences) >= beam_width:
                break
        
        # Add any remaining sequences as complete
        complete_sequences.extend(sequences)
        
        # Return the best sequence with length normalization
        if complete_sequences:
            best_sequence = max(complete_sequences, key=lambda x: x['score'] / len(x['tokens']))
            # Convert to tensor format
            result = torch.full((1, self.tgt_seq_len), 2, dtype=torch.long, device=encoder_output.device)
            result[0, :len(best_sequence['tokens'])] = torch.tensor(best_sequence['tokens'], device=encoder_output.device)
            return result
        else:
            # Fallback: return empty sequence
            result = torch.full((1, self.tgt_seq_len), 2, dtype=torch.long, device=encoder_output.device)
            result[0, 0] = 0  # SOS token
            return result

        

        
    


            
            

    