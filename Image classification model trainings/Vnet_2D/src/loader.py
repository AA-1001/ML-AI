import os
import torch
from torch.utils.data import Dataset, DataLoader
from torchvision.io import read_image

# Custom Dataset Class
class UltrasoundDataset(Dataset):
    def __init__(self, img_dir, mask_dir):
        self.img_dir = img_dir
        self.mask_dir = mask_dir
        self.image_paths = sorted([os.path.join(img_dir, img) for img in os.listdir(img_dir) if img.endswith('.png')])
        self.mask_paths = sorted([os.path.join(mask_dir, img) for img in os.listdir(mask_dir) if img.endswith('.png')])

        if len(self.image_paths) != len(self.mask_paths):
            raise ValueError("The number of images and masks does not match!")

    def __len__(self):
        return len(self.image_paths)

    def __getitem__(self, idx):
        img_path = self.image_paths[idx]
        mask_path = self.mask_paths[idx]
        
        image = read_image(img_path).float() / 255.0  # Normalize image to range [0, 1]
        mask = read_image(mask_path).float() / 255.0  # Normalize mask to range [0, 1]
        
        if image.shape[0] > 1:
            image = image[0].unsqueeze(0)
        if mask.shape[0] > 1:
            mask = mask[0].unsqueeze(0)

        return image, mask

# Function to load datasets
def load_data(train_data_dir, train_mask_dir, val_data_dir, val_mask_dir):
    train_dataset = UltrasoundDataset(img_dir=train_data_dir, mask_dir=train_mask_dir)
    train_loader = DataLoader(train_dataset, batch_size=32, shuffle=True)

    val_dataset = UltrasoundDataset(img_dir=val_data_dir, mask_dir=val_mask_dir)
    val_loader = DataLoader(val_dataset, batch_size=32, shuffle=False)

    return train_loader, val_loader
