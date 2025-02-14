import torch
import torch.nn as nn
import math
import copy
class IdentityEmbedding(nn.Module):
    def __init__(self):
        super().__init__()
    
    def forward(self, x):
        return x

class Embedding_Layer(nn.Module):
    def __init__(self, input_dim, embedding_dim):
        super(Embedding_Layer, self).__init__()
        self.embedding = nn.Linear(input_dim, embedding_dim)
        #input dimension is the dimension of each patch
        #embedding_dim is the dimensionality that each patch is embedded onto
    def forward(self, x):
        return self.embedding(x)



class PositionalEncoding(nn.Module):
    # Just add a randomised matrix to 
    def __init__(self, first_dim, embedding_dimension):
        super(PositionalEncoding, self).__init__()
        self.first_dim = first_dim
        self.embedding_dimension = embedding_dimension
        self.positional_encoding = torch.nn.Parameter(torch.rand(first_dim, embedding_dimension))

    def forward(self, x):
        return x + self.positional_encoding
    

class FeedForwardLayer(nn.Module):
    def __init__(self, embedding_dim, hidden_layer_dim):
        super(FeedForwardLayer, self).__init__()
        self.ff1 = torch.nn.Linear(embedding_dim, hidden_layer_dim)
        self.ff2 = torch.nn.Linear(hidden_layer_dim, embedding_dim)
        self.relu = torch.nn.ReLU()
    
    def forward(self, x):
        return self.ff2(self.relu(self.ff1(x)))
    


class EncoderLayer(nn.Module):
    def __init__(self, embedding_dim, positional_encoding, attention_layer, feed_forward, input_dim):
        super(EncoderLayer, self).__init__()
        self.positional_encoding = positional_encoding
        self.attn_layer = attention_layer
        self.FF_layer = feed_forward
        self.embedding_dim = embedding_dim
        self.input_dim = input_dim

        self.norm1 = torch.nn.LayerNorm(embedding_dim)
        self.norm2 = torch.nn.LayerNorm(embedding_dim)

    
    def forward(self, x):
        embedding = self.positional_encoding(x)

        attn, prob = self.attn_layer.forward(embedding, embedding, embedding, None)
        x = self.norm1(embedding + attn)
        ff_output = self.FF_layer(x)
        x = self.norm2(x + ff_output)

        return x

def clones(module, N):
    "Produce N identical layers."
    return nn.ModuleList([copy.deepcopy(module) for _ in range(N)])


class Encoder(nn.Module):
    def __init__(self, input_dim, embedding_dim, n_loops, layer):
        super(Encoder, self).__init__()
        self.embedding_layer = Embedding_Layer(input_dim, embedding_dim)
        self.layers = clones(layer, n_loops)
        self.norm1 = torch.nn.LayerNorm(embedding_dim)
        self.encoder_layers = clones(layer, n_loops)
    
    def forward(self, x):
        "Pass the input (and mask) through each layer in turn."
        x = self.embedding_layer(x)
        for layer in self.layers:
            x = layer(x)
        return self.norm1(x)
    


