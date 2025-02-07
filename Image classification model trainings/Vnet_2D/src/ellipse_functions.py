import numpy as np
import cv2

# Function to fit an ellipse to the predicted mask
def fit_ellipse_from_mask(predicted_mask):
    # Convert the predicted mask to a binary mask (0 or 255)
    binary_mask = (predicted_mask.astype(np.uint8)) * 255
    
    # Find contours in the binary mask
    contours, _ = cv2.findContours(binary_mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    
    # If no contours are found, return None
    if len(contours) == 0:
        print("No contours found in the mask.")
        return None
    
    # Find the largest contour based on area
    largest_contour = max(contours, key=cv2.contourArea)
    
    # Fit an ellipse to the largest contour if there are enough points
    if len(largest_contour) >= 5:  # Minimum points required for fitting an ellipse
        ellipse = cv2.fitEllipse(largest_contour)
        return ellipse
    else:
        print("Not enough points to fit an ellipse.")
        return None

# Function to create a smooth mask based on the fitted ellipse
def create_ellipse_mask(image_shape, ellipse_params):
    # Create a blank mask
    mask = np.zeros(image_shape, dtype=np.uint8)
    
    # Draw the ellipse on the mask
    center, axes, angle = ellipse_params
    cv2.ellipse(mask, (int(center[0]), int(center[1])),
                (int(axes[0] / 2), int(axes[1] / 2)),
                angle, 0, 360, (255), -1)  # Fill the ellipse
    return mask
