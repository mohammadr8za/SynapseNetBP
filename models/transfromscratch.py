import torch
import torch.nn as nn


class DenoisingNetwork(nn.Module):
    def __init__(self, input_dim=512, hidden_dim=512, num_heads=8, num_layers=8, dropout=.1):
        super(DenoisingNetwork, self).__init__()

        self.embedding = nn.Linear(input_dim, hidden_dim)
        self.dropout = nn.Dropout(dropout)
        self.encoder_layers = nn.ModuleList([
            TransformerEncoderLayer(hidden_dim, num_heads, hidden_dim * 4, dropout)
            for _ in range(num_layers)
        ])
        self.decoder_layers = nn.ModuleList([
            TransformerDecoderLayer(hidden_dim, num_heads, hidden_dim * 4, dropout)
            for _ in range(num_layers)
        ])
        self.fc = nn.Linear(hidden_dim, input_dim)

    def forward(self, x):
        x = self.embedding(x)
        x = self.dropout(x)

        # Encoder
        for layer in self.encoder_layers:
            x = layer(x)

        # Decoder
        for layer in self.decoder_layers:
            x = layer(x)

        x = self.fc(x)
        return x


class TransformerEncoderLayer(nn.Module):
    def __init__(self, d_model, nhead, dim_feedforward, dropout):
        super(TransformerEncoderLayer, self).__init__()

        self.self_attn = nn.MultiheadAttention(d_model, nhead)
        self.linear1 = nn.Linear(d_model, dim_feedforward)
        self.dropout1 = nn.Dropout(dropout)
        self.linear2 = nn.Linear(dim_feedforward, d_model)
        self.norm1 = nn.LayerNorm(d_model)
        self.norm2 = nn.LayerNorm(d_model)
        self.dropout2 = nn.Dropout(dropout)

    def forward(self, x):
        attn_input = x
        x, _ = self.self_attn(x, x, x)
        x = x + attn_input
        x = self.norm1(x)

        fc_input = x
        x = self.linear1(x)
        x = self.dropout1(x)
        x = torch.relu(x)
        x = self.linear2(x)
        x = self.dropout2(x)
        x = x + fc_input
        x = self.norm2(x)

        return x


class TransformerDecoderLayer(nn.Module):
    def __init__(self, d_model, nhead, dim_feedforward, dropout):
        super(TransformerDecoderLayer, self).__init__()

        self.self_attn = nn.MultiheadAttention(d_model, nhead)
        self.multihead_attn = nn.MultiheadAttention(d_model, nhead)
        self.linear1 = nn.Linear(d_model, dim_feedforward)
        self.dropout1 = nn.Dropout(dropout)
        self.linear2 = nn.Linear(dim_feedforward, d_model)
        self.norm1 = nn.LayerNorm(d_model)
        self.norm2 = nn.LayerNorm(d_model)
        self.norm3 = nn.LayerNorm(d_model)
        self.dropout2 = nn.Dropout(dropout)
        self.dropout3 = nn.Dropout(dropout)

    def forward(self, x):
        attn_input = x
        x, _ = self.self_attn(x, x, x)
        x = x + attn_input
        x = self.norm1(x)

        attn2_input = x
        x, _ = self.multihead_attn(x, x, x)
        x = x + attn2_input
        x = self.norm2(x)

        fc_input = x
        x = self.linear1(x)
        x = self.dropout1(x)
        x = torch.relu(x)
        x = self.linear2(x)
        x = self.dropout2(x)
        x = x + fc_input
        x = self.norm3(x)

        return x