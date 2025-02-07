import matplotlib.pyplot as plt
import numpy as np
# Function to display an image and its mask side by side
def plot_image_and_mask(image, mask):
    fig, ax = plt.subplots(1, 2, figsize=(10, 5))
    
    # Convert image and mask from tensor to numpy
    image = np.transpose(image.squeeze().numpy(), (1, 2, 0))  # Convert to HxWxC format
    mask = mask.squeeze().numpy()  # Convert to HxW format (grayscale)
    
    # Display image
    ax[0].imshow(image)
    ax[0].set_title('Original Image')
    ax[0].axis('off')

    # Display mask
    ax[1].imshow(mask, cmap='gray')
    ax[1].set_title('Ground Truth Mask')
    ax[1].axis('off')

    plt.show()


import torch
import matplotlib.pyplot as plt
from PIL import Image
import numpy as np
from torchvision import transforms
from ellipse_fitting import fit_ellipse_from_mask, create_ellipse_mask
import os

# Function to predict mask and visualize results
def visualize_prediction(image_path, mask_path, model, device):
    # Load and transform the image
    image = Image.open(image_path).convert("RGB")
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

    # Convert predicted mask to 2D
    predicted_mask_2d = np.squeeze(predicted_mask)

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
    plt.imshow(image)
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
