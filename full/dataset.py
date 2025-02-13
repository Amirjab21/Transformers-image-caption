import torch

idx_to_token = {
    0: '0', 1: '1', 2: '2', 3: '3', 4: '4',
    5: '5', 6: '6', 7: '7', 8: '8', 9: '9',
    10: '<START>', 11: "<PAD>", 12: '<END>'
}

def turn_to_one_hot(label_array, idx_to_token):
    token_to_idx = {v: k for k, v in idx_to_token.items()}

    # Your existing one-hot encoding code
    label_tensor = torch.tensor(label_array)
    one_hot = torch.nn.functional.one_hot(label_tensor, num_classes=13)

    # Create start token
    start_token = torch.zeros(13)
    start_token[10] = 1

    # Add start token to sequence
    sequence_with_start = torch.vstack([start_token, one_hot])
    return sequence_with_start

class ImageDataset(torch.utils.data.Dataset):
    def __init__(self, image_label_pairs):
        self.image_label_pairs = image_label_pairs
    
    def __len__(self):
        return len(self.image_label_pairs)
    
    def __getitem__(self, idx):
        image, label = self.image_label_pairs[idx]
        
        patches_array = self.split_image_to_patches(image)
        patches_tensor = [torch.tensor(patch.flatten(), dtype=torch.float32) for patch in patches_array] #array of 16 with each tensor[196]
        image_tensor = torch.stack(patches_tensor) # tensor [16,196]
        label_tensor = turn_to_one_hot(label, idx_to_token)
        #turn label array of integers into tensor:
        # label_tensor = torch.tensor(label, dtype=torch.long)
        
        
        return {
            'image': image_tensor,
            'label': label_tensor,
        }
    
    def split_image_to_patches(self, image):
        blocks = []  # Initialize empty list to store blocks

        for i in range(4):
            for j in range(4):
                # Extract 14x14 block
                block = image[i*14:(i+1)*14, j*14:(j+1)*14]
                blocks.append(block)  # Add block to our list
        return blocks