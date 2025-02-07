import numpy as np
import torch
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score

# Function to calculate DICE score
def dice_score(y_true, y_pred, smooth=1e-5):
    intersection = (y_true * y_pred).sum()  # Calculate intersection
    return (2. * intersection + smooth) / (y_true.sum() + y_pred.sum() + smooth)  # Calculate DICE score

# Function to calculate various metrics
def calculate_metrics(val_loader, model):
    # Initialize device and model
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model.to(device)  # Ensure model is on the correct device
    model.eval()  # Set model to evaluation mode

    # Variables to store total metrics
    total_dice_score = 0.0
    num_batches = 0
    all_preds = []
    all_masks = []

    # Inference loop to calculate metrics on the validation set
    with torch.no_grad():  # Disable gradient computation for evaluation
        for images, masks in val_loader:
            images = images.to(device)  # Move images to the same device as the model
            masks = masks.to(device).float()  # Assuming masks are binary (0, 1)

            # Get predictions from the model
            preds = model(images)
            preds = torch.sigmoid(preds)  # Apply sigmoid to get probabilities

            # Store predictions and masks for metric calculations
            all_preds.append(preds.cpu().numpy())  # Move predictions to CPU and convert to numpy
            all_masks.append(masks.cpu().numpy())  # Move masks to CPU and convert to numpy
            
            # Convert predictions to binary masks
            preds_binary = (preds > 0.5).float()  # Binarize predictions

            # Calculate DICE score for the current batch
            batch_dice_score = dice_score(masks, preds_binary)  # Call DICE score function
            total_dice_score += batch_dice_score.item()  # Accumulate DICE score
            num_batches += 1  # Increment number of batches

    # Concatenate all predictions and masks from batches
    all_preds = np.concatenate(all_preds)
    all_masks = np.concatenate(all_masks)

    # Calculate the average DICE score across all batches
    average_dice_score = total_dice_score / num_batches
    print(f"Average DICE Score on the validation set: {average_dice_score:.4f}")

    # Flatten masks and predictions for metric calculations
    all_preds_flat = all_preds.flatten()
    all_masks_flat = all_masks.flatten()

    # Calculate other metrics
    mae = mean_absolute_error(all_masks_flat, all_preds_flat)
    mse = mean_squared_error(all_masks_flat, all_preds_flat)
    rmse = np.sqrt(mse)

    # Mean Percentage Error (MPE)
    all_masks_flat_safe = np.where(all_masks_flat == 0, 1e-5, all_masks_flat)  # Avoid division by zero
    mpe = np.mean(np.abs((all_masks_flat_safe - all_preds_flat) / all_masks_flat_safe)) * 100

    # R-squared (R²)
    r2 = r2_score(all_masks_flat, all_preds_flat)

    # Print metrics
    print(f"Mean Absolute Error (MAE): {mae:.4f}")
    print(f"Mean Squared Error (MSE): {mse:.4f}")
    print(f"Root Mean Squared Error (RMSE): {rmse:.4f}")
    print(f"Mean Percentage Error (MPE): {mpe:.4f}%")
    print(f"R-squared (R²): {r2:.4f}")
