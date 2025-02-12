import torch.nn as nn
import torch

class Decoder(nn.Module):
     def __init__(self, input_dim, tgt_vocab_size, intermediate_attn_dim, n_loops, feed_forward, self_attn_layer, cross_attn_layer, positional_encoding, embedding_layer):
        super(Decoder, self).__init__()
        self.embedding_layer = embedding_layer
        self.positional_encoding = positional_encoding
        self.self_attn_layer = self_attn_layer
        self.cross_attn_layer = cross_attn_layer
        self.FF_layer = feed_forward
        self.tgt_vocab_size = tgt_vocab_size
        self.input_dim = input_dim
        self.n_loops = n_loops

        self.projectbacktovocab = torch.nn.Linear(intermediate_attn_dim, tgt_vocab_size)

        self.norm1 = torch.nn.LayerNorm(tgt_vocab_size)
        self.norm2 = torch.nn.LayerNorm(tgt_vocab_size)
        self.norm3 = torch.nn.LayerNorm(tgt_vocab_size)

     def forward(self, x, encoder_output):
        embedding = self.embedding_layer(x)
        embedding = self.positional_encoding(embedding)
        attn, prob = self.self_attn_layer.forward(embedding, embedding, embedding, True)
        x = self.norm1(attn + embedding)
        attn, prob = self.cross_attn_layer.forward(query_input=x, key_input=encoder_output, value_input=encoder_output)
        attn = self.projectbacktovocab(attn)
        x = self.norm2(x + attn)

        ff_output = self.FF_layer(x)
        x = self.norm3(x + ff_output)
        return x



