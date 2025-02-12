import torch
from torch import nn


class Transformer(nn.Module):
    def __init__(self, d_model, encoder, decoder, tgt_vocab_size):
        super(Transformer, self).__init__()

        self.encoder = encoder
        self.decoder = decoder

        self.fc = nn.Linear(tgt_vocab_size, tgt_vocab_size)

    def forward(self, image, caption):
        enc_output = self.encoder.forward(image) #does the embedding
        dec_output = self.decoder.forward(caption, enc_output)

        output = self.fc(dec_output)
        return output
    
