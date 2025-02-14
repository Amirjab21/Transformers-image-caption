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
import os

from dataset import ImageDataset
from Transformer import Transformer
from Decoder import Decoder
from Encoder import Encoder, FeedForwardLayer, PositionalEncoding, IdentityEmbedding, EncoderLayer
from Decoder import DecoderLayer
from attention import Cross_Attention_Layer, Attention_Layer

def generate_random_image(train_dataset):
    images = []
    labels = []
    for _ in range(4):
        idx = torch.randint(len(train_dataset), size=(1,)).item()
        img, label = train_dataset[idx]
        images.append(img.squeeze().numpy())
        labels.append(label)
    top_row = np.concatenate([images[0], images[1]], axis=1)
    bottom_row = np.concatenate([images[2], images[3]], axis=1)
    final_image = np.concatenate([top_row, bottom_row], axis=0)
    return final_image, labels


def validate(model, num_samples=10):
    model.eval()
    correct = 0
    total = num_samples
    
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
            
            if predicted_indices[:4] == true_labels:
                correct += 1
    
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



def turn_to_one_hot(label_array, idx_to_token):
    token_to_idx = {v: k for k, v in idx_to_token.items()}

    # Your existing one-hot encoding code
    label_tensor = torch.tensor(label_array)
    one_hot = torch.nn.functional.one_hot(label_tensor, num_classes=13)

    # Create start token
    start_token = torch.zeros(13)
    start_token[10] = 1
    start_token = start_token.unsqueeze(0)
    one_hot = one_hot.squeeze(1)
    # Add start token to sequence
    sequence_with_start = torch.vstack([start_token, one_hot])
    return sequence_with_start

def generate_random_image_batch(batch_size=512, batch=None):
    images = []
    labels = []
    # print(batch, "batch")
    for _ in range(4):
        idx = torch.randint(len(batch), size=(1,)).item()
        img, label = batch[idx]['image'], batch[idx]['label']
        
        images.append(img)
        labels.append(label)
        
        # Concatenate images into a 2x2 grid
    top_row = np.concatenate([images[0], images[1]], axis=1)
    bottom_row = np.concatenate([images[2], images[3]], axis=1)
    final_image = np.concatenate([top_row, bottom_row], axis=0)
    return final_image, labels

def collate_fn(batch):
    all_images = []
    all_labels = []
    # print(batch[0], "batch")
    for _ in range(len(batch)):
        image, labels = generate_random_image_batch(batch_size,batch)
        
        # Process each image individually and combine results
        patches_array = split_image_to_patches(image)
        patches_tensor = [torch.tensor(patch.flatten(), dtype=torch.float32) for patch in patches_array]
        image_tensor = torch.stack(patches_tensor)
        
        label_tensor = turn_to_one_hot(labels, idx_to_token)
        label_tensor = torch.tensor(label_tensor, dtype=torch.long)
        
        all_images.append(image_tensor)
        all_labels.append(label_tensor)
    
    return {
        'image': torch.stack(all_images),
        'label': torch.stack(all_labels)
    }

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
        idx = torch.randint(len(train_dataset), size=(1,)).item()
        image, label = train_dataset[idx]
        dataset.append((image.squeeze().numpy(), [label]))
    
    return dataset

train_dataset = load_dataset()
dataset = create_dataset(30000, train_dataset)

dataset_ready = ImageDataset(dataset)
batch_size = 512
dataloader = torch.utils.data.DataLoader(
    dataset_ready,
    batch_size=batch_size,
    shuffle=True,
    collate_fn=collate_fn
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

encoder_layer = EncoderLayer(embedding_dim, positional_layer_encoder, attention_layer_encoder, fflayer_encoder, input_dimension_images)
encoder = Encoder(input_dimension_images, embedding_dim, n_loops, encoder_layer)

input_dimension_decoder = 5
tgt_vocab_size = 13
dim_model_decoder = 24
fflayer_decoder = FeedForwardLayer(tgt_vocab_size, tgt_vocab_size)
self_attention_layer_decoder = Attention_Layer(tgt_vocab_size, num_heads)
cross_attention_layer_decoder = Cross_Attention_Layer(embedding_dim, tgt_vocab_size, dim_model_decoder, num_heads)
positional_layer_decoder = PositionalEncoding(input_dimension_decoder,tgt_vocab_size)
embedding_layer_decoder = IdentityEmbedding()

decoder_layer = DecoderLayer(input_dimension_decoder,tgt_vocab_size, dim_model_decoder, n_loops, fflayer_decoder, self_attention_layer_decoder, cross_attention_layer_decoder, positional_layer_decoder, embedding_layer_decoder, token_to_idx['<PAD>'])
decoder = Decoder(tgt_vocab_size, token_to_idx['<PAD>'], embedding_layer_decoder, decoder_layer, n_loops)
transformer = Transformer(embedding_dim, encoder, decoder, tgt_vocab_size)

learning_rate = 0.001
optimizer = torch.optim.Adam(transformer.parameters(), learning_rate)
epochs = 25
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
            
            
             
            for i in range(5):  # For the 4 digits
                batch_loss += criterion(output[:, i, :], true_indices[:, i])
            # batch_loss = criterion(output.view(-1, output.size(-1)), true_indices.view(-1)) ALTERNATIVE LOSS FUNCTION


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

def visualize_attention(attn_probs):
    """
    Visualize attention probabilities for a specific head
    
    Args:
        attn_probs: Tensor of shape [batch_size, num_heads, seq_len, seq_len]
        head: Which attention head to visualize (default=0)
    """
    # Take first batch and specified head
    attn_map = attn_probs.detach().cpu().numpy()
    
    plt.figure(figsize=(10, 10))
    plt.imshow(attn_map, cmap='viridis')
    plt.colorbar()
    plt.title(f'Attention Head ')
    plt.xlabel('Key Position')
    plt.ylabel('Query Position')
    plt.show()

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

# do_test_new(transformer)
from PIL import Image

def predict_from_png(model, image_path):
    """
    Takes a PNG image file, processes it, and returns predictions using the transformer model
    
    Args:
        model: The trained transformer model
        image_path: Path to the PNG image file
    
    Returns:
        predicted_digits: List of predicted digits
    """
    # Load and preprocess the image
    transform = transforms.Compose([
        transforms.Grayscale(),  # Convert to grayscale
        transforms.Resize((28, 28)),  # Resize to MNIST size
        transforms.ToTensor(),
        transforms.Normalize((0.1307,), (0.3081,))  # MNIST normalization
    ])
    
    # Load image
    image = Image.open(image_path)
    image = transform(image).squeeze().numpy()
    
    # Visualize the initial processed 28x28 image
    plt.figure(figsize=(10, 5))
    plt.subplot(1, 2, 1)
    plt.imshow(image, cmap='gray')
    plt.title('Single Processed Image (28x28)')
    
    # Create 2x2 grid
    full_image = np.zeros((56, 56))
    full_image[0:28, 0:28] = image
    full_image[0:28, 28:56] = image
    full_image[28:56, 0:28] = image
    full_image[28:56, 28:56] = image
    
    # Visualize the 2x2 grid
    plt.subplot(1, 2, 2)
    plt.imshow(full_image, cmap='gray')
    plt.title('2x2 Grid (56x56)')
    plt.show()
    
    # Visualize the 16 patches (optional)
    patches_array = split_image_to_patches(full_image)
    fig, axes = plt.subplots(4, 4, figsize=(10, 10))
    for i in range(4):
        for j in range(4):
            idx = i * 4 + j
            axes[i, j].imshow(patches_array[idx], cmap='gray')
            axes[i, j].axis('off')
    plt.suptitle('16 Patches (14x14 each)')
    plt.show()
    
    # Process image
    patches_array = split_image_to_patches(full_image)
    patches_tensor = [torch.tensor(patch.flatten(), dtype=torch.float32) for patch in patches_array]
    image_tensor = torch.stack(patches_tensor).unsqueeze(0)  # Add batch dimension [1, 16, 196]
    

    # Prepare model and sequence
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

# print(predict_from_png(transformer, "first.png"))