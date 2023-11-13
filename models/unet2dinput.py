import torch
import torch.nn as nn
import torch.nn.functional as F


class UNet2DInput(nn.Module):
    def __init__(self, dropout=0.1):
        super(UNet2DInput, self).__init__()
        self.dropout = dropout

        # Define the encoder
        self.conv1 = nn.Conv2d(2, 256, kernel_size=3, stride=1, padding=1)
        self.conv2 = nn.Conv1d(256, 512, kernel_size=3, stride=1, padding=1, padding_mode= "replicate")
        self.conv3 = nn.Conv1d(512, 512, kernel_size=3, stride=1, padding=1, padding_mode= "replicate")
        self.conv4 = nn.Conv1d(512, 1024, kernel_size=3, stride=1, padding=1, padding_mode= "replicate")
        self.conv5 = nn.Conv1d(1024, 2048, kernel_size=3, stride=1, padding=1, padding_mode= "replicate")

        # Define the decoder
        self.upconv1 = nn.ConvTranspose1d(2048, 1024, kernel_size=3, stride=1, padding=1)
        self.conv6 = nn.Conv1d(2048, 1024, kernel_size=3, stride=1, padding=1, padding_mode= "replicate")
        self.upconv2 = nn.ConvTranspose1d(1024, 512,  kernel_size=3, stride=1, padding=1)
        self.conv7 = nn.Conv1d(1024, 512, kernel_size=3, stride=1, padding=1, padding_mode= "replicate")
        self.upconv3 = nn.ConvTranspose1d(512, 512,  kernel_size=3, stride=1, padding=1)
        self.conv8 = nn.Conv1d(1024, 512, kernel_size=3, stride=1, padding=1, padding_mode= "replicate")
        self.upconv4 = nn.ConvTranspose1d(512, 256,  kernel_size=3, stride=1, padding=1)
        self.conv9 = nn.Conv1d(512, 256, kernel_size=3, stride=1, padding=1, padding_mode= "replicate")
        self.conv10 = nn.Conv1d(256, 1,  kernel_size=3, stride=1, padding=1)
        self.dropout = nn.Dropout(p=dropout)

    def forward(self, x):
        # Encoder part of the network
        x = x.unsqueeze(2)
        x1 = F.relu(self.conv1((x)))
        x2 = F.relu(self.conv2((x1.squeeze())))
        x3 = F.relu(self.conv3((x2)))
        x4 = F.relu(self.conv4((x3)))
        x5 = F.relu(self.conv5((x4)))
        x5 = self.dropout(x5)

        # Decoder part of the network

        x = F.relu(self.upconv1(x5))
        x = torch.cat([x, x4], dim=1)
        x = F.relu(self.conv6(x))
        x = F.relu(self.upconv2(x))
        x = torch.cat([x, x3], dim=1)
        x = F.relu(self.conv7(x))
        x = F.relu(self.upconv3(x))
        x = torch.cat([x, x2], dim=1)
        x = F.relu(self.conv8(x))
        x = F.relu(self.upconv4(x))
        x = torch.cat([x, x1.squeeze()], dim=1)
        x = F.relu(self.conv9(x))
        x = self.conv10(x)

        return x.squeeze(1)


class UNet2DInput2(nn.Module):
    def __init__(self, dropout=0.1):
        super(UNet2DInput2, self).__init__()
        self.dropout = dropout

        # Define the encoder
        self.conv1 = nn.Conv2d(2, 64, kernel_size=3, stride=1, padding=1)
        self.conv2 = nn.Conv1d(64, 128, kernel_size=3, stride=1, padding=1)
        self.conv3 = nn.Conv1d(128, 256, kernel_size=3, stride=1, padding=1)
        self.conv4 = nn.Conv1d(256, 512, kernel_size=3, stride=1, padding=1)
        self.conv5 = nn.Conv1d(512, 1024, kernel_size=3, stride=1, padding=1)

        # Define the decoder
        self.upconv1 = nn.ConvTranspose1d(1024, 512, kernel_size=3, stride=1, padding=1)
        self.conv6 = nn.Conv1d(1024, 512, kernel_size=3, stride=1, padding=1)
        self.upconv2 = nn.ConvTranspose1d(512, 256,  kernel_size=3, stride=1, padding=1)
        self.conv7 = nn.Conv1d(512, 256, kernel_size=3, stride=1, padding=1)
        self.upconv3 = nn.ConvTranspose1d(256, 128,  kernel_size=3, stride=1, padding=1)
        self.conv8 = nn.Conv1d(256, 128, kernel_size=3, stride=1, padding=1)
        self.upconv4 = nn.ConvTranspose1d(128, 64,  kernel_size=3, stride=1, padding=1)
        self.conv9 = nn.Conv1d(128, 64, kernel_size=3, stride=1, padding=1)
        self.conv10 = nn.Conv1d(64, 1,  kernel_size=3, stride=1, padding=1)
        self.dropout = nn.Dropout(p=dropout)

    def forward(self, x):
        # Encoder part of the network
        # x = x.unsqueeze(2)
        x1 = F.relu(self.conv1((x)))
        x2 = F.relu(self.conv2((x1.squeeze())))
        x3 = F.relu(self.conv3((x2)))
        x4 = F.relu(self.conv4((x3)))
        x5 = F.relu(self.conv5((x4)))
        x5 = self.dropout(x5)

        # Decoder part of the network

        x = F.relu(self.upconv1(x5))
        x = torch.cat([x, x4], dim=1)
        x = F.relu(self.conv6(x))
        x = F.relu(self.upconv2(x))
        x = torch.cat([x, x3], dim=1)
        x = F.relu(self.conv7(x))
        x = F.relu(self.upconv3(x))
        x = torch.cat([x, x2], dim=1)
        x = F.relu(self.conv8(x))
        x = F.relu(self.upconv4(x))
        x = torch.cat([x, x1.squeeze()], dim=1)
        x = F.relu(self.conv9(x))
        x = self.conv10(x)

        return x.squeeze(1)



if __name__ == "__main__":

    sample = torch.randn(size=(1, 1, 512))
    model = UNet2DInput2()
    output = model(sample)