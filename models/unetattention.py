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
        query = self.conv1(x.unsqueeze(0))  # Add extra batch dimension
        key = self.conv2(x.unsqueeze(0))  # Add extra batch dimension
        energy = torch.matmul(query.transpose(2, 1), key)
        attention_weights = F.softmax(energy, dim=2)
        attended = torch.matmul(attention_weights, x.unsqueeze(0))  # Add extra batch dimension
        output = self.conv3(attended).squeeze(0)  # Remove extra batch dimension
        return output


class UNetAttention(nn.Module):
    def __init__(self, dropout=0.1):
        super(UNetAttention, self).__init__()
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
        x1 = F.relu(self.conv1(x.unsqueeze(0)))  # Add extra batch dimension
        x1 = self.attention1(x1).squeeze(0)  # Remove extra batch dimension

        x2 = F.relu(self.conv2(F.max_pool1d(x1.unsqueeze(0), kernel_size=2, stride=2)))  # Add extra batch dimension
        x2 = self.attention2(x2).squeeze(0)  # Remove extra batch dimension

        x3 = F.relu(self.conv3(F.max_pool1d(x2.unsqueeze(0), kernel_size=2, stride=2)))  # Add extra batch dimension
        x3 = self.attention3(x3).squeeze(0)  # Remove extra batch dimension

        x4 = F.relu(self.conv4(F.max_pool1d(x3.unsqueeze(0), kernel_size=2, stride=2)))  # Add extra batch dimension
        x4 = self.attention4(x4).squeeze(0)  # Remove extra batch dimension

        x5 = F.relu(self.conv5(F.max_pool1d(x4.unsqueeze(0), kernel_size=2, stride=2))).squeeze(
            0)  # Removeextra batch dimension

        # Decoder part of the network
        x6 = self.dropout(F.relu(self.upconv1(x5)))
        x6 = torch.cat((x6, x4), dim=1)
        x6 = self.dropout(F.relu(self.conv6(x6)))

        x7 = self.dropout(F.relu(self.upconv2(x6)))
        x7 = torch.cat((x7, x3), dim=1)
        x7 = self.dropout(F.relu(self.conv7(x7)))

        x8 = self.dropout(F.relu(self.upconv3(x7)))
        x8 = torch.cat((x8, x2), dim=1)
        x8 = self.dropout(F.relu(self.conv8(x8)))

        x9 = self.dropout(F.relu(self.upconv4(x8)))
        x9 = torch.cat((x9, x1), dim=1)
        x9 = self.dropout(F.relu(self.conv9(x9)))

        output = self.conv10(x9).squeeze(1)  # Adjust the dimensions

        return output