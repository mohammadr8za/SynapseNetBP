import torch
import torch.nn as nn

class VNet(nn.Module):
    def __init__(self, in_channels= 1, out_channels=1):
        super(VNet, self).__init__()

        # Downsample layers
        self.down1 = DoubleConv(in_channels, 16)
        self.pool1 = nn.MaxPool1d(kernel_size=2)
        self.down2 = DoubleConv(16, 32)
        self.pool2 = nn.MaxPool1d(kernel_size=2)
        self.down3 = DoubleConv(32, 64)
        self.pool3 = nn.MaxPool1d(kernel_size=2)

        # Bottleneck layer
        self.bottleneck = DoubleConv(64, 128)

        # Upsample layers
        self.up3 = nn.ConvTranspose1d(128, 64, kernel_size=2, stride=2)
        self.upconv3 = DoubleConv(128, 64)
        self.up2 = nn.ConvTranspose1d(64, 32, kernel_size=2, stride=2)
        self.upconv2 = DoubleConv(64, 32)
        self.up1 = nn.ConvTranspose1d(32, 16, kernel_size=2, stride=2)
        self.upconv1 = DoubleConv(32, 16)

        # Output layer
        self.out = nn.Conv1d(16, out_channels, kernel_size=1)

    def forward(self, x):
        # Downsample
        down1 = self.down1(x)
        pool1 = self.pool1(down1)
        down2 = self.down2(pool1)
        pool2 = self.pool2(down2)
        down3 = self.down3(pool2)
        pool3 = self.pool3(down3)

        # Bottleneck
        bottleneck = self.bottleneck(pool3)

        # Upsample
        up3 = self.up3(bottleneck)
        cat3 = torch.cat([down3, up3], dim=1)
        upconv3 = self.upconv3(cat3)
        up2 = self.up2(upconv3)
        cat2 = torch.cat([down2, up2], dim=1)
        upconv2 = self.upconv2(cat2)
        up1 = self.up1(upconv2)
        cat1 = torch.cat([down1, up1], dim=1)
        upconv1 = self.upconv1(cat1)

        # Output
        out = self.out(upconv1)

        return out.squeeze()

class DoubleConv(nn.Module):
    def __init__(self, in_channels, out_channels):
        super(DoubleConv, self).__init__()
        self.conv1 = nn.Conv1d(in_channels, out_channels, kernel_size=3, padding=1)
        self.bn1 = nn.BatchNorm1d(out_channels)
        self.relu1 = nn.ReLU(inplace=True)
        self.conv2 = nn.Conv1d(out_channels, out_channels, kernel_size=3, padding=1)
        self.bn2 = nn.BatchNorm1d(out_channels)
        self.relu2 = nn.ReLU(inplace=True)

    def forward(self, x):
        x = self.conv1(x)
        x = self.bn1(x)
        x = self.relu1(x)
        x = self.conv2(x)
        x = self.bn2(x)
        x = self.relu2(x)
        return x




if __name__ == "__main__":
    import torch

    ppg_signal_tensor = torch.randn(1, 1, 624)
    model = VNet(in_channels=1, out_channels=1)
    abp_signal_tensor = model(ppg_signal_tensor)
    abp_signal = abp_signal_tensor.squeeze(0).detach().numpy()


    a = 5