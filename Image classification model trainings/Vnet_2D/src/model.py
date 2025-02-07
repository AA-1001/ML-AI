import torch
import torch.nn as nn

class DiceLoss(nn.Module):
    def __init__(self, smooth=1e-5):
        super(DiceLoss, self).__init__()
        self.smooth = smooth

    def forward(self, y_true, y_pred):
        y_true_f = y_true.view(-1)
        y_pred_f = y_pred.view(-1)
        intersection = torch.sum(y_true_f * y_pred_f)
        denom = torch.sum(y_true_f) + torch.sum(y_pred_f)
        dice = (2. * intersection + self.smooth) / (denom + self.smooth)
        return 1 - dice

class ResBlock(nn.Module):
    def __init__(self, in_channels):
        super(ResBlock, self).__init__()
        self.conv1 = nn.Conv2d(in_channels, 16, kernel_size=5, padding=2)
        self.bn1 = nn.BatchNorm2d(16)
        self.prelu1 = nn.PReLU()

        self.conv2 = nn.Conv2d(16, 16, kernel_size=5, padding=2)
        self.bn2 = nn.BatchNorm2d(16)
        self.prelu2 = nn.PReLU()

        self.residual_conv = nn.Conv2d(in_channels, 16, kernel_size=1)

    def forward(self, x):
        residual = self.residual_conv(x)
        out = self.conv1(x)
        out = self.bn1(out)
        out = self.prelu1(out)

        out = self.conv2(out)
        out = self.bn2(out)
        out = self.prelu2(out)

        out += residual
        return out

class DownBlock(nn.Module):
    def __init__(self, in_channels):
        super(DownBlock, self).__init__()
        self.resblock = ResBlock(in_channels)
        self.downsample = nn.Conv2d(16, 32, kernel_size=2, stride=2)

    def forward(self, x):
        x = self.resblock(x)
        down = self.downsample(x)
        return down, x

class UpBlock(nn.Module):
    def __init__(self, in_channels):
        super(UpBlock, self).__init__()
        self.upconv = nn.ConvTranspose2d(in_channels, 16, kernel_size=2, stride=2)
        self.resblock = ResBlock(32)

    def forward(self, x, skip):
        x = self.upconv(x)
        x = torch.cat([x, skip], dim=1)
        x = self.resblock(x)
        return x

class VNet(nn.Module):
    def __init__(self, input_channels=1, num_classes=1, is_training=True, stage_num=5):
        super(VNet, self).__init__()
        self.keep_prob = 1.0 if is_training else 0.0

        self.input_conv = nn.Sequential(
            nn.Conv2d(input_channels, 16, kernel_size=5, padding=2),
            nn.BatchNorm2d(16),
            nn.PReLU()
        )

        self.down_blocks = nn.ModuleList()
        self.up_blocks = nn.ModuleList()

        for _ in range(stage_num):
            in_channels = 16 if _ == 0 else 32
            self.down_blocks.append(DownBlock(in_channels))

        for _ in range(stage_num):
            in_channels = 32 if _ == 0 else 16
            self.up_blocks.append(UpBlock(in_channels))

        self.final_conv = nn.Conv2d(16, num_classes, kernel_size=1)

    def forward(self, x):
        skips = []
        x = self.input_conv(x)
        for down in self.down_blocks:
            x, skip = down(x)
            skips.append(skip)

        for up, skip in zip(self.up_blocks, reversed(skips)):
            x = up(x, skip)

        out = self.final_conv(x)
        if self.final_conv.out_channels == 1:
            out = torch.sigmoid(out)
        else:
            out = torch.softmax(out, dim=1)

        return out
