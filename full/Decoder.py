import torch.nn as nn
import torch

class Decoder(nn.Module):
     def __init__(self, input_dim, tgt_vocab_size, intermediate_attn_dim, n_loops, feed_forward, self_attn_layer, cross_attn_layer, positional_encoding, embedding_layer, pad_token):
        super(Decoder, self).__init__()
        self.embedding_layer = embedding_layer
        self.positional_encoding = positional_encoding
        self.self_attn_layer = self_attn_layer
        self.cross_attn_layer = cross_attn_layer
        self.FF_layer = feed_forward
        self.tgt_vocab_size = tgt_vocab_size
        self.input_dim = input_dim
        self.n_loops = n_loops
        self.pad_token = pad_token

        self.projectbacktovocab = torch.nn.Linear(intermediate_attn_dim, tgt_vocab_size)

        self.norm1 = torch.nn.LayerNorm(tgt_vocab_size)
        self.norm2 = torch.nn.LayerNorm(tgt_vocab_size)
        self.norm3 = torch.nn.LayerNorm(tgt_vocab_size)

     def forward(self, x, encoder_output):
        mask = self.generate_padding_mask(x)
        # print(mask, 'mask')
        embedding = self.embedding_layer(x)
        embedding = self.positional_encoding(embedding)
        attn, prob = self.self_attn_layer.forward(embedding, embedding, embedding, mask)
        x = self.norm1(attn + embedding)
        attn, prob = self.cross_attn_layer.forward(query_input=x, key_input=encoder_output, value_input=encoder_output)
        attn = self.projectbacktovocab(attn)
        x = self.norm2(x + attn)

        ff_output = self.FF_layer(x)
        x = self.norm3(x + ff_output)
        return x
     
     def generate_padding_mask(self, caption):
            """
            Generate combined padding and causal mask for decoder self-attention.
            Args:
                caption: Input caption tensor of shape (batch_size, seq_len, vocab_size)
            Returns:
                Attention mask of shape (batch_size, seq_len, seq_len) where:
                - pad tokens are masked with 0
                - future tokens are masked with 0 (causal masking)
                - valid tokens are marked with 1
            """
            batch_size, seq_length, _ = caption.shape
            
            # Get padding mask by checking if the last index (pad token) is 1
            #padding is always final token
            padding_mask = (caption[:, :, -1] != 1).bool()  # [batch_size, seq_len]

            # Create causal mask (lower triangular matrix)
            # causal_mask = torch.tril(torch.ones(seq_length, seq_length)).to(caption.device)
            
            # Expand padding mask to [batch_size, seq_len, seq_len]
            padding_mask = padding_mask.unsqueeze(1) * padding_mask.unsqueeze(2)
            # Create final mask by combining padding and causal masks
            final_mask = padding_mask
            
            return final_mask



