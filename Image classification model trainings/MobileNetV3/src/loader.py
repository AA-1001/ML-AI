import os
from torch.utils.data import Dataset, DataLoader
from torchvision import transforms
from PIL import Image

# Custom Dataset Class
class CustomImageDataset(Dataset):
    def __init__(self, img_dir, mask_dir):
        self.img_dir = img_dir
        self.mask_dir = mask_dir
        # Get all image file paths
        self.image_paths = [os.path.join(img_dir, img) for img in os.listdir(img_dir) if img.endswith('.png')]
        # Get corresponding mask paths
        self.mask_paths = [os.path.join(mask_dir, img.replace('.png', '_mask.png')) for img in os.listdir(img_dir) if img.endswith('.png')]

    def __len__(self):
        return len(self.image_paths)

    def __getitem__(self, idx):
        img_path = self.image_paths[idx]
        mask_path = self.mask_paths[idx]
        
        image = Image.open(img_path).convert("RGB")  
        mask = Image.open(mask_path).convert("L")    
        
        image = transforms.ToTensor()(image)  # Convert image to tensor
        mask = transforms.ToTensor()(mask)    # Convert mask to tensor
        
        return image, mask  # Return image and mask as a tuple

# Function to load datasets
def load_data(train_data_dir, train_mask_dir, val_data_dir, val_mask_dir, batch_size=8):
    train_dataset = CustomImageDataset(img_dir=train_data_dir, mask_dir=train_mask_dir)
    val_dataset = CustomImageDataset(img_dir=val_data_dir, mask_dir=val_mask_dir)

    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
    val_loader = DataLoader(val_dataset, batch_size=batch_size, shuffle=False)

    return train_loader, val_loader
