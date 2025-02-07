import numpy as np
import torch
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
import os

# Function to calculate DICE score
def dice_score(y_true, y_pred, smooth=1e-5):
    intersection = (y_true * y_pred).sum()  # Calculate intersection
    return (2. * intersection + smooth) / (y_true.sum() + y_pred.sum() + smooth)  # Calculate DICE score

# Function to calculate trainable parameters
def count_trainable_params(model):
    return sum(p.numel() for p in model.parameters() if p.requires_grad)

# Function to calculate model size in MBs
def model_size_in_mb(model_path):
    size_bytes = os.path.getsize(model_path)
    size_mb = size_bytes / (1024 * 1024)  # Convert to MB
    return size_mb

# Function to calculate MAE, MSE, RMSE, MPE, and R-squared metrics
def calculate_metrics(all_masks_flat, all_preds_flat):
    mae = mean_absolute_error(all_masks_flat, all_preds_flat)
    mse = mean_squared_error(all_masks_flat, all_preds_flat)
    rmse = np.sqrt(mse)

    # Mean Percentage Error (MPE)
    all_masks_flat_safe = np.where(all_masks_flat == 0, 1e-5, all_masks_flat)  # Avoid division by zero
    mpe = np.mean(np.abs((all_masks_flat_safe - all_preds_flat) / all_masks_flat_safe)) * 100

    # R-squared (R²)
    r2 = r2_score(all_masks_flat, all_preds_flat)

    return mae, mse, rmse, mpe, r2
