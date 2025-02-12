import torch
import torch.nn as nn
import math

class IdentityEmbedding(nn.Module):
    def __init__(self):
        super().__init__()
        # self.linear = nn.Linear(input_dim, embedding_dim)
    
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

class Attention_Layer(nn.Module):
    def __init__(self, d_model, num_heads):
        super(Attention_Layer, self).__init__()
        self.d_model = d_model
        self.num_heads = num_heads

        self.W_q = torch.nn.Linear(d_model, d_model)
        self.W_k = torch.nn.Linear(d_model, d_model)
        self.W_v = torch.nn.Linear(d_model, d_model)
    
    def forward(self, query_input, key_input, value_input, mask=None):
        dim_k = self.d_model // self.num_heads
        query = self.W_q(query_input)
        key = self.W_k(key_input)
        value = self.W_v(value_input)
        
        query_key = torch.matmul(query, key.transpose(-1, -2)) / math.sqrt(dim_k)
        prob = query_key.softmax(dim=-1)
        weighted_attention = torch.matmul(prob, value)
        return weighted_attention, prob
    
class Cross_Attention_Layer(nn.Module):
    def __init__(self, encoder_output_dim, decoder_dim, d_model, num_heads):
        super(Cross_Attention_Layer, self).__init__()
        self.d_model = d_model
        self.num_heads = num_heads

        self.W_q = torch.nn.Linear(decoder_dim, d_model)

        self.W_k = torch.nn.Linear(encoder_output_dim, d_model)
        self.W_v = torch.nn.Linear(encoder_output_dim, d_model)
    
    def forward(self, query_input, key_input, value_input, mask=None):
        dim_k = self.d_model // self.num_heads
        query = self.W_q(query_input)
        key = self.W_k(key_input)
        value = self.W_v(value_input)
        
        query_key = torch.matmul(query, key.transpose(-1, -2)) / math.sqrt(dim_k)
        prob = query_key.softmax(dim=-1)
        weighted_attention = torch.matmul(prob, value)
        return weighted_attention, prob
    


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



class Encoder(nn.Module):
    def __init__(self, input_dim, embedding_dim, n_loops, feed_forward, attention_layer, positional_encoding):
        super(Encoder, self).__init__()
        self.embedding_layer = Embedding_Layer(input_dim, embedding_dim)
        self.positional_encoding = positional_encoding
        self.attn_layer = attention_layer
        self.FF_layer = feed_forward
        self.embedding_dim = embedding_dim
        self.input_dim = input_dim
        self.n_loops = n_loops

        self.norm1 = torch.nn.LayerNorm(embedding_dim)
        self.norm2 = torch.nn.LayerNorm(embedding_dim)


        #remove below when doing full model
        self.intermediate = torch.nn.Linear(64*16, 40)
        self.relu = torch.nn.ReLU()
        self.flatten = torch.nn.Flatten()
        self.final = torch.nn.Linear(40, 40)
    
    def get_qkv(self, q_input, k_input, v_input, embedding_dim):
        query = torch.nn.Linear(q_input.size(-1), embedding_dim)
        key = torch.nn.Linear(k_input.size(-1), embedding_dim)
        value = torch.nn.Linear(v_input.size(-1), embedding_dim)
        return query, key, value

    
    def forward(self, x):
        embedding = self.embedding_layer(x)

        embedding = self.positional_encoding(embedding)

        # for i in range(self.n_loops): #To do, to add loops, we must clone Encoder layers which consist of ff + attn
        # query, key, value = self.get_qkv(x, x, x, self.embedding_dim)
        attn, prob = self.attn_layer.forward(embedding, embedding, embedding, None)
        x = self.norm1(embedding + attn)
        ff_output = self.FF_layer(x)
        x = self.norm2(x + ff_output)

        return x
    
    def predict_single_integer(self, x):
        if len(x.shape) == 2:  # Single instance case [16, 64]
            x = x.unsqueeze(0)  # Add batch dimension [1, 16, 64]
        
        # Now x is [batch, 16, 64] in both cases
        x = x.reshape(x.size(0), -1)  # Flatten to [batch, 1024]
        flatten = self.flatten(x)
        x = self.intermediate(flatten)
        x = self.relu(x)
        return self.final(x)
    
    def predict_numbers(self, x):
        if len(x.shape) == 2:  # Single instance case [16, 64]
            x = x.unsqueeze(0)  # Add batch dimension [1, 16, 64]
        
        # Now x is [batch, 16, 64] in both cases
        x = x.reshape(x.size(0), -1)  # Flatten to [batch, 1024]
        flatten = self.flatten(x)
        x = self.intermediate(flatten)
        x = self.relu(x)
        x = self.final(x)
        # Reshape to [batch, 4, 10] for 4 digit predictions
        return x.view(x.size(0), 4, 10)


