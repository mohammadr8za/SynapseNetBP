import torch
import torch.nn as nn
import torch.nn.functional as F


class AttentionBlock(nn.Module):
    def __init__(self, in_channels):
        super(AttentionBlock, self).__init__()
        self.conv1 = nn.Conv1d(in_channels, in_channels // 2, kernel_size=1)
        self.conv2 = nn.Conv1d(in_channels, in_channels // 2, kernel_size=1)
        self.conv3 = nn.Conv1d(in_channels // 2, 1, kernel_size=1)

    def forward(self, x):
        query = self.conv1(x)
        key = self.conv2(x)
        energy = torch.matmul(query.transpose(2, 1), key)
        attention_weights = F.softmax(energy, dim=2)
        attended = torch.matmul(attention_weights, x)
        output = self.conv3(attended)
        return output


class UNetPPGtoABP(nn.Module):
    def __init__(self, dropout=0.1):
        super(UNetPPGtoABP, self).__init__()
        self.dropout = dropout

        # Define the encoder
        self.conv1 = nn.Conv1d(1, 64, kernel_size=3, stride=1, padding=1)
        self.attention1 = AttentionBlock(64)
        self.conv2 = nn.Conv1d(64, 128, kernel_size=3, stride=1, padding=1)
        self.attention2 = AttentionBlock(128)
        self.conv3 = nn.Conv1d(128, 256, kernel_size=3, stride=1, padding=1)
        self.attention3 = AttentionBlock(256)
        self.conv4 = nn.Conv1d(256, 512, kernel_size=3, stride=1, padding=1)
        self.attention4 = AttentionBlock(512)
        self.conv5 = nn.Conv1d(512, 1024, kernel_size=3, stride=1, padding=1)

        # Define the decoder
        self.upconv1 = nn.ConvTranspose1d(1024, 512, kernel_size=2, stride=2)
        self.conv6 = nn.Conv1d(1024, 512, kernel_size=3, stride=1, padding=1)
        self.upconv2 = nn.ConvTranspose1d(512, 256, kernel_size=2, stride=2)
        self.conv7 = nn.Conv1d(512, 256, kernel_size=3, stride=1, padding=1)
        self.upconv3 = nn.ConvTranspose1d(256, 128, kernel_size=2, stride=2)
        self.conv8 = nn.Conv1d(256, 128, kernel_size=3, stride=1, padding=1)
        self.upconv4 = nn.ConvTranspose1d(128, 64, kernel_size=2, stride=2)
        self.conv9 = nn.Conv1d(128, 64, kernel_size=3, stride=1, padding=1)
        self.conv10 = nn.Conv1d(64, 1, kernel_size=1, stride=1)
        self.dropout = nn.Dropout(p=dropout)

    def forward(self, x):
        # Encoder part of the network
        x1 = F.relu(self.conv1(x))
        x1 = self.attention1(x1)

        x2 = F.relu(self.conv2(F.max_pool1d(x1, kernel_size=2, stride=2)))
        x2 = self.attention2(x2)

        x3 = F.relu(self.conv3(F.max_pool1d(x2, kernel_size=2, stride=2)))
        x3 = self.attention3(x3)

        x4 = F.relu(self.conv4(F.max_pool1d(x3, kernel_size=2, stride=2)))
        x4 = self.attention4(x4)

        x5 = F.relu(self.conv5(F.max_pool1d(x4, kernel_size=2, stride=2)))
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
        x = torch.cat([x, x1], dim=1)
        x = F.relu(self.conv9(x))
        x = self.conv10(x)

        return x.squeeze(1)