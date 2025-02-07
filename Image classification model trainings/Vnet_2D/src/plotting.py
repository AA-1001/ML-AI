import matplotlib.pyplot as plt
import numpy as np
# Function to display an image and its mask side by side

def plot_image_and_mask(image, mask):
    fig, ax = plt.subplots(1, 2, figsize=(10, 5))

    # Convert image and mask from tensor to numpy
    image = image.squeeze().cpu().numpy()  # Remove batch dimension, convert to NumPy
    mask = mask.squeeze().cpu().numpy()    # Same for mask

    # If the image is grayscale, it has shape (H, W), no need to transpose
    if len(image.shape) == 2:  # Grayscale image
        ax[0].imshow(image, cmap='gray')
    else:
        # If it's RGB, image should be (C, H, W) -> transpose to (H, W, C)
        image = np.transpose(image, (1, 2, 0))  # Convert to HxWxC format for RGB
        ax[0].imshow(image)

    ax[0].set_title('Input Image')
    ax[0].axis('off')

    # Display mask (assuming it's a 2D grayscale image)
    ax[1].imshow(mask, cmap='gray')
    ax[1].set_title('Ground Truth Mask')
    ax[1].axis('off')

    plt.show()


import torch
import matplotlib.pyplot as plt
from PIL import Image
import numpy as np
from torchvision import transforms
from ellipse_functions import fit_ellipse_from_mask, create_ellipse_mask
import os
# Function to predict mask and visualize results
def visualize_prediction(image_path, mask_path, model, device):
    # Load and transform the image
    image = Image.open(image_path).convert("L")  # Convert image to grayscale
    image_tensor = transforms.ToTensor()(image).unsqueeze(0).to(device)  # Add batch dimension and send to device
    
    # Load and transform the mask (ground truth)
    original_mask = Image.open(mask_path).convert("L")  # Convert mask to grayscale
    original_mask_tensor = transforms.ToTensor()(original_mask).unsqueeze(0).to(device)  # Add batch dimension

    # Predict the mask using the model
    with torch.no_grad():
        output = model(image_tensor)  # Forward pass
        output = torch.sigmoid(output)  # Apply sigmoid to get values between 0 and 1
        predicted_mask = (output > 0.5).float()  # Binarize output (threshold at 0.5)
        predicted_mask = predicted_mask.squeeze(0).cpu().numpy()  # Remove batch dimension and convert to NumPy

    # Ensure predicted_mask is 2D (single channel)
    predicted_mask_2d = np.squeeze(predicted_mask)  # Should be in shape (height, width)

    # Fit an ellipse to the predicted mask
    ellipse = fit_ellipse_from_mask(predicted_mask_2d)

    # Create a smooth mask based on the fitted ellipse
    if ellipse is not None:
        ellipse_mask = create_ellipse_mask(predicted_mask_2d.shape, ellipse)
    else:
        ellipse_mask = np.zeros(predicted_mask_2d.shape, dtype=np.uint8)  # Return empty mask if no ellipse

    # Visualize the original image, ground truth, and predicted mask with ellipse
    plt.figure(figsize=(15, 5))

    # Display input image
    plt.subplot(1, 3, 1)
    plt.imshow(image, cmap='gray')
    plt.title('Input Image')
    plt.axis('off')

    # Display original ground truth mask
    plt.subplot(1, 3, 2)
    plt.imshow(original_mask, cmap='gray')
    plt.title('Ground Truth Mask')
    plt.axis('off')

    # Display ellipse fitted mask
    plt.subplot(1, 3, 3)
    plt.imshow(ellipse_mask, cmap='gray')
    plt.title('Predicted Ellipse Mask')
    plt.axis('off')

    plt.show()

# Example usage
# visualize_prediction("path_to_image.png", "path_to_mask.png", model, device)
