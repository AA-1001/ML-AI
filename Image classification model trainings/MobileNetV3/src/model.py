import torch.nn as nn
from torchvision import models

# Define a segmentation model using MobileNetV3
class MobileNetV3Segmentation(nn.Module):
    def __init__(self):
        super(MobileNetV3Segmentation, self).__init__()
        self.base_model = models.mobilenet_v3_large(weights='DEFAULT')  # Load pre-trained MobileNetV3
        self.base_model.classifier = nn.Sequential()  # Remove the original classifier

        # Add a convolutional layer to produce the segmentation output
        self.segmentation_head = nn.Conv2d(960, 1, kernel_size=1)  # Change input channels to 960
        # Upsample layer to increase output size to 224x224
        self.upsample = nn.Upsample(size=(224, 224), mode='bilinear', align_corners=False)

    def forward(self, x):
        x = self.base_model.features(x)  # Use only the feature extractor part
        x = self.segmentation_head(x)  # Apply the segmentation head
        x = self.upsample(x)  # Upsample to match input image size
        return x
