import torch
import torch.nn as nn
import torch.optim as optim
from tqdm import tqdm
from model import MobileNetV3Segmentation

class TrainingClass:
    def __init__(self, learning_rate=1e-4):
        # Use GPU if available
        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        print(f"Using device: {self.device}")

        self.model = MobileNetV3Segmentation().to(self.device)  # Move model to GPU
        self.criterion = nn.BCEWithLogitsLoss()  # Define loss function for binary classification
        self.optimizer = optim.Adam(self.model.parameters(), lr=learning_rate)  # Optimizer

    def train(self, train_loader, val_loader, num_epochs=20):
        self.model.train()  # Set model to training mode
        for epoch in range(num_epochs):
            running_loss = 0.0
            for inputs, targets in tqdm(train_loader, desc=f'Training Epoch {epoch + 1}/{num_epochs}'):
                # Move inputs and targets to the GPU
                inputs = inputs.to(self.device)
                targets = targets.to(self.device)

                self.optimizer.zero_grad()  # Clear previous gradients
                outputs = self.model(inputs)  # Forward pass

                targets = targets.float() 
                loss = self.criterion(outputs, targets)  # Compute loss
                loss.backward()  # Backward pass
                self.optimizer.step()  # Optimize

                running_loss += loss.item()

            # Print average loss for the epoch
            print(f'Epoch [{epoch + 1}/{num_epochs}], Loss: {running_loss / len(train_loader):.4f}')

            # Run validation after each epoch
            self.validate(val_loader)

    def validate(self, val_loader):
        self.model.eval()  # Set model to evaluation mode
        with torch.no_grad():
            val_loss = 0.0
            for inputs, targets in val_loader:
                # Move inputs and targets to the GPU
                inputs = inputs.to(self.device)
                targets = targets.to(self.device)

                outputs = self.model(inputs)  # Forward pass
                targets = targets.float()  # Ensure targets are float
                loss = self.criterion(outputs, targets)  # Compute loss
                val_loss += loss.item()

            print(f'Validation Loss: {val_loss / len(val_loader):.4f}')

    def save_model(self, model_save_path):
        torch.save(self.model.state_dict(), model_save_path)
        print(f'Model saved to {model_save_path}')
