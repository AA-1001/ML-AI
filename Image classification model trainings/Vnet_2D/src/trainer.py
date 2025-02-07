import torch
import torch.optim as optim
from tqdm import tqdm
from model import VNet, DiceLoss

class VNetTraining:
    def __init__(self, train_loader, val_loader, learning_rate=1e-4):
        self.model = VNet(input_channels=1, num_classes=1).to(device)  # Move model to device
        self.criterion = DiceLoss()  # Use Dice Loss for segmentation
        self.optimizer = optim.NAdam(self.model.parameters(), lr=learning_rate)  # NAdam optimizer
        self.train_loader = train_loader
        self.val_loader = val_loader

    def train(self, num_epochs=10):
        self.model.train()
        for epoch in range(num_epochs):
            running_loss = 0.0
            for inputs, targets in tqdm(self.train_loader, desc=f'Training Epoch {epoch + 1}/{num_epochs}'):
                inputs, targets = inputs.to(device), targets.to(device).float()

                self.optimizer.zero_grad()
                outputs = self.model(inputs)
                loss = self.criterion(outputs, targets)
                loss.backward()
                self.optimizer.step()

                running_loss += loss.item()

            print(f'Epoch [{epoch + 1}/{num_epochs}], Loss: {running_loss / len(self.train_loader):.4f}')
            self.validate()

    def validate(self):
        self.model.eval()
        val_loss = 0.0
        with torch.no_grad():
            for inputs, targets in self.val_loader:
                inputs, targets = inputs.to(device), targets.to(device).float()
                
                outputs = self.model(inputs)
                loss = self.criterion(outputs, targets)

                val_loss += loss.item()

        print(f'Validation Loss: {val_loss / len(self.val_loader):.4f}')

    def save_model(self):
        model_save_path = './model'
        torch.save(self.model.state_dict(), model_save_path)
        print(f'Model saved to {model_save_path}')

# Set device
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
