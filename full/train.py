import torch
import torch.nn as nn
import torch.optim as optim
import torch.utils.data as data
import math
import copy
import torch
from torchvision import datasets, transforms
import matplotlib.pyplot as plt
import numpy as np
import tqdm

from dataset import ImageDataset
from Transformer import Transformer
from Decoder import Decoder
from Encoder import Encoder, FeedForwardLayer, Attention_Layer, PositionalEncoding, IdentityEmbedding, Cross_Attention_Layer

idx_to_token = {
    0: '0', 1: '1', 2: '2', 3: '3', 4: '4',
    5: '5', 6: '6', 7: '7', 8: '8', 9: '9',
    10: '<START>'
}

device = torch.device("cpu")

def load_dataset():
    transform = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize((0.1307,), (0.3081,))
    ])
    # Download and load the training data
    train_dataset = datasets.MNIST(root='./data', 
                             train=True, 
                             download=True, 
                             transform=transform
                             )
    return train_dataset

def generate_random_image(train_dataset):
    images = []
    labels = []
    for _ in range(4):
        idx = torch.randint(len(train_dataset), size=(1,)).item()
        img, label = train_dataset[idx]
        images.append(img.squeeze().numpy())
        labels.append(label)

# Concatenate images into a 2x2 grid
    top_row = np.concatenate([images[0], images[1]], axis=1)
    bottom_row = np.concatenate([images[2], images[3]], axis=1)
    final_image = np.concatenate([top_row, bottom_row], axis=0)
    return final_image, labels

def split_image_to_patches(image):
    blocks = []  # Initialize empty list to store blocks

    for i in range(4):
        for j in range(4):
            # Extract 14x14 block
            block = image[i*14:(i+1)*14, j*14:(j+1)*14]
            blocks.append(block)  # Add block to our list
    return blocks

def create_dataset(num_images, train_dataset):
    dataset = []
    
    for _ in range(num_images):
        image, labels = generate_random_image(train_dataset)
        dataset.append((image, labels))
    
    return dataset

train_dataset = load_dataset()
dataset = create_dataset(10000, train_dataset)

dataset_ready = ImageDataset(dataset)
batch_size = 20
dataloader = torch.utils.data.DataLoader(
    dataset_ready,
    batch_size=batch_size,
    shuffle=True,
)



embedding_dim = 64
input_dimension_images = 196
hidden_layer_dimension = 32
num_heads = 1
seq_length_images = 16
n_loops = 1

fflayer_encoder = FeedForwardLayer(embedding_dim, hidden_layer_dimension)
attention_layer_encoder = Attention_Layer(embedding_dim, num_heads)
positional_layer_encoder = PositionalEncoding(seq_length_images,embedding_dim)
encoder = Encoder(input_dimension_images, embedding_dim, 1, fflayer_encoder,attention_layer=attention_layer_encoder, positional_encoding=positional_layer_encoder)

input_dimension_decoder = 5
tgt_vocab_size = 11
dim_model_decoder = 24
fflayer_decoder = FeedForwardLayer(tgt_vocab_size, tgt_vocab_size)
self_attention_layer_decoder = Attention_Layer(tgt_vocab_size, num_heads)
cross_attention_layer_decoder = Cross_Attention_Layer(embedding_dim, tgt_vocab_size, dim_model_decoder, num_heads)
positional_layer_decoder = PositionalEncoding(input_dimension_decoder,tgt_vocab_size)
embedding_layer_decoder = IdentityEmbedding()


decoder = Decoder(input_dimension_decoder,tgt_vocab_size, dim_model_decoder, n_loops, fflayer_decoder, self_attention_layer_decoder, cross_attention_layer_decoder, positional_layer_decoder, embedding_layer_decoder)
transformer = Transformer(embedding_dim, encoder, decoder, tgt_vocab_size)

learning_rate = 0.001
optimizer = torch.optim.SGD(transformer.parameters(), learning_rate)
epochs = 5
criterion = nn.CrossEntropyLoss()

def train():
    transformer.train()
    total_loss = 0
    for epoch in range(epochs):
        epoch_loss = 0
        progress_bar = tqdm.tqdm(
            dataloader, desc=f"Epoch {epoch + 1}/{epochs}"
        )

        for batch_idx, batch in enumerate(progress_bar):
            
            batch_loss = 0
            image, label = batch['image'].to(device), batch['label'].to(device)
            
            optimizer.zero_grad()

            output = transformer.forward(image, label)


            # Apply softmax across the vocabulary dimension (dim=2)
            output_probabilities = torch.softmax(output, dim=2)
            predicted_digits = torch.argmax(output_probabilities, dim=2)  # Shape: [batch_size, 4]
            true_indices = torch.argmax(label, dim=2)  # Convert one-hot to indices

            predicted_digits = [[idx_to_token[idx.item()] for idx in pred] for pred in predicted_digits]
            true_digits = [[idx_to_token[idx.item()] for idx in true] for true in true_indices]
            
            if batch_idx % 100 == 0:  # Print every 100 batches
                for j in range(min(3, len(predicted_digits))):  # Show first 3 examples
                    print(f"Predicted: {predicted_digits[j]} | True: {true_digits[j]}")
            
            
             # Calculate loss for each position in the sequence
            for i in range(4):  # For the 4 digits
                batch_loss += criterion(output[:, i, :], label[:, i])


            progress_bar.set_postfix({"batch_loss": batch_loss.item() /  batch_size})

            epoch_loss += batch_loss.item()
            total_loss += batch_loss.item()
            batch_loss.backward()
            optimizer.step()
        
        epoch_loss = epoch_loss / len(dataloader.dataset)
        # total_loss += epoch_loss
        print(f"Epoch {epoch + 1}/{epochs}, Loss: {epoch_loss:.4f}")
        print(f"Total {epoch + 1}/{epochs}, Loss: {total_loss / (epoch + 1):.4f}")

        
train()
