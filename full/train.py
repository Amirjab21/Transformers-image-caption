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
import wandb

from dataset import ImageDataset
from Transformer import Transformer
from Decoder import Decoder
from Encoder import Encoder, FeedForwardLayer, Attention_Layer, PositionalEncoding, IdentityEmbedding, Cross_Attention_Layer

def validate(model, num_samples=10):
    model.eval()
    correct = 0
    total = num_samples * 4  # 4 digits per sample
    
    with torch.no_grad():
        for _ in range(num_samples):
            image, true_labels = generate_random_image(train_dataset)
            patches_array = split_image_to_patches(image)
            patches_tensor = [torch.tensor(patch.flatten(), dtype=torch.float32) for patch in patches_array]
            image_tensor = torch.stack(patches_tensor).unsqueeze(0)
            
            current_sequence = torch.zeros((1, 5, 13))
            current_sequence[0, 0, token_to_idx['<START>']] = 1
            
            predicted_indices = []
            for pos in range(5):
                output = model.forward(image_tensor, current_sequence)
                output_probabilities = torch.softmax(output, dim=2)
                predicted_digit = torch.argmax(output_probabilities[0, pos])
                predicted_indices.append(predicted_digit.item())
                
                if pos < 3:
                    current_sequence[0, pos + 1, predicted_digit.item()] = 1
            
            correct += sum(p == t for p, t in zip(predicted_indices, true_labels))
    
    accuracy = correct / total

    wandb.init(project="mnist-transformer", config={
        "learning_rate": learning_rate,
        "epochs": epochs,
        "batch_size": batch_size,
        "embedding_dim": embedding_dim,
        "num_heads": num_heads
    })

    wandb.log({"accuracy": accuracy})

    print(f"Validation Accuracy: {accuracy:.2%}")
    return accuracy

idx_to_token = {
    0: '0', 1: '1', 2: '2', 3: '3', 4: '4',
    5: '5', 6: '6', 7: '7', 8: '8', 9: '9',
    10: '<START>', 11: "<PAD>", 12: '<END>'
}

token_to_idx = {
    '0': 0, '1': 1, '2': 2, '3': 3, '4': 4,
    '5': 5, '6': 6, '7': 7, '8': 8, '9': 9,
    '<START>': 10, "<PAD>": 11, '<END>': 12
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
batch_size = 512
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
tgt_vocab_size = 13
dim_model_decoder = 24
fflayer_decoder = FeedForwardLayer(tgt_vocab_size, tgt_vocab_size)
self_attention_layer_decoder = Attention_Layer(tgt_vocab_size, num_heads)
cross_attention_layer_decoder = Cross_Attention_Layer(embedding_dim, tgt_vocab_size, dim_model_decoder, num_heads)
positional_layer_decoder = PositionalEncoding(input_dimension_decoder,tgt_vocab_size)
embedding_layer_decoder = IdentityEmbedding()


decoder = Decoder(input_dimension_decoder,tgt_vocab_size, dim_model_decoder, n_loops, fflayer_decoder, self_attention_layer_decoder, cross_attention_layer_decoder, positional_layer_decoder, embedding_layer_decoder, token_to_idx['<PAD>'])
transformer = Transformer(embedding_dim, encoder, decoder, tgt_vocab_size)

learning_rate = 0.001
optimizer = torch.optim.Adam(transformer.parameters(), learning_rate)
epochs = 50
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
            # print(image[0], label[0])
            
            optimizer.zero_grad()

            output = transformer.forward(image, label)


            # Apply softmax across the vocabulary dimension (dim=2)
            output_probabilities = torch.softmax(output, dim=2)
            predicted_digits = torch.argmax(output_probabilities, dim=2)  # Shape: [batch_size, 4]
            
            true_indices = torch.argmax(label[:, 1:], dim=2)  # Skip first position (START token)
            end_token = torch.full((label.size(0), 1), token_to_idx['<END>'], device=device)
            true_indices = torch.cat([true_indices, end_token], dim=1)  # Add END token
            # print(true_indices[0])

            predicted_digits = [[idx_to_token[idx.item()] for idx in pred] for pred in predicted_digits]
            true_digits = [[idx_to_token[idx.item()] for idx in true] for true in true_indices]
            
            if batch_idx % 100 == 0:  # Print every 100 batches
                for j in range(min(3, len(predicted_digits))):  # Show first 3 examples
                    print(f"Predicted: {predicted_digits[j]} | True: {true_digits[j]}")
            
            
             # Calculate loss for each position in the sequence
            for i in range(5):  # For the 4 digits
                batch_loss += criterion(output[:, i, :], true_indices[:, i])


            progress_bar.set_postfix({"batch_loss": batch_loss.item() })

            epoch_loss += batch_loss.item()
            total_loss += batch_loss.item()
            batch_loss.backward()
            optimizer.step()
        validate(transformer,100)
        epoch_loss = epoch_loss / len(dataloader.dataset)
        # total_loss += epoch_loss
        # print(f"Epoch {epoch + 1}/{epochs}, Loss: {epoch_loss:.4f}")
        print(f"Total {epoch + 1}/{epochs}, Loss: {total_loss / (epoch + 1):.4f}")

        
# train()

import os
# torch.save({ 'model_state_dict': transformer.state_dict()}, 'checkpoints/best_model.pt')

def load_model(model, optimizer, checkpoint_path='checkpoints/best_model.pt'):
    """Load a saved model and optimizer state"""
    if os.path.exists(checkpoint_path):
        checkpoint = torch.load(checkpoint_path, map_location=torch.device('cpu'))
        model.load_state_dict(checkpoint['model_state_dict'])
        
        # Add verification checks
        print(f"Model loaded from {checkpoint_path}")
        # Print a few parameter statistics to verify loading
        for name, param in model.named_parameters():
            if param.requires_grad:
                print(f"{name}: mean={param.data.mean():.4f}, std={param.data.std():.4f}")
                break  # Just print the first parameter as an example
        return model
    print(f"No checkpoint found at {checkpoint_path}")
    return False

# # After loading the model, you can also test it with a sample input
transformer = load_model(transformer, optimizer)

# def visualize_attention(attn_probs):
#     """
#     Visualize attention probabilities for a specific head
    
#     Args:
#         attn_probs: Tensor of shape [batch_size, num_heads, seq_len, seq_len]
#         head: Which attention head to visualize (default=0)
#     """
#     # Take first batch and specified head
#     attn_map = attn_probs.detach().cpu().numpy()
    
#     plt.figure(figsize=(10, 10))
#     plt.imshow(attn_map, cmap='viridis')
#     plt.colorbar()
#     plt.title(f'Attention Head ')
#     plt.xlabel('Key Position')
#     plt.ylabel('Query Position')
#     plt.show()

def do_test_new(model):
    image, label = generate_random_image(train_dataset)

    print(f"\nTrue labels: {label}")

    patches_array = split_image_to_patches(image)
    patches_tensor = [torch.tensor(patch.flatten(), dtype=torch.float32) for patch in patches_array]
    image_tensor = torch.stack(patches_tensor).unsqueeze(0)  # Add batch dimension [1, 16, 196]
    
    model.eval()
    current_sequence = torch.zeros((1, 5, 13))

    # Initialize all positions with PAD token (index 11)
    current_sequence[0, :, 11] = 1
    # Set START token at position 0
    current_sequence[0, 0, 10] = 1
    current_sequence[0, 0, 11] = 0  # Remove PAD token from START position

    predicted_indices = [10]
    
    # Generate one digit at a time
    for pos in range(4):
        # print(current_sequence)
        output = model.forward(image_tensor, current_sequence)
        # print(output, "ouytput")
        output_probabilities = torch.softmax(output, dim=2)
        # print(output_probabilities, output_probabilities[0, pos])
        predicted_digits = torch.argmax(output_probabilities[0, pos])
        # print(predicted_digits, output_probabilities[0, pos])
        predicted_indices.append(predicted_digits.item())
        # print(predicted_digits.item())
        
        # Update sequence for next iteration (if not last position)
        if pos < 3:
            current_sequence = torch.zeros((1, 5, 13))
            # Set PAD token everywhere first
            current_sequence[0, :, 11] = 1
            # Set START token
            current_sequence[0, 0, 10] = 1
            current_sequence[0, 0, 11] = 0
            # Add predicted digits so far
            for i, idx in enumerate(predicted_indices[1:], start=1):
                current_sequence[0, i, idx] = 1
                current_sequence[0, i, 11] = 0  # Remove PAD token where we put a prediction
    
    # Convert to readable digits using idx_to_token
    predicted_digits = [idx_to_token[idx] for idx in predicted_indices]
    
    print(f"Predicted digits: {predicted_digits}")

do_test_new(transformer)
    