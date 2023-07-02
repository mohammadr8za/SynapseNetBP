import torch
import torch.nn as nn


# transformer network class
class TransformerBlock(nn.Module):
    def __init__(self, embed_dim=512, num_heads=8, dropout=0.1):
        super().__init__()
        self.embed_dim = embed_dim
        self.num_heads = num_heads
        self.head_dim = embed_dim // num_heads
        assert embed_dim % num_heads == 0, "embed_dim must be divisible by num_heads"

        self.self_attn = nn.MultiheadAttention(embed_dim, num_heads)
        self.dropout1 = nn.Dropout(dropout)
        self.norm1 = nn.LayerNorm(embed_dim)


        self.fc1 = nn.Linear(embed_dim, 4 * embed_dim)
        self.activation = nn.ReLU()
        self.fc2 = nn.Linear(4 * embed_dim, embed_dim)
        self.dropout2 = nn.Dropout(dropout)
        self.norm2 = nn.LayerNorm(embed_dim)

    def forward(self, x):
        attn_input = x
        x, _ = self.self_attn(x, x, x)
        x = self.dropout1(x)
        x = self.norm1(x + attn_input)

        fc_input = x
        x = self.fc1(x)
        x = self.activation(x)
        x = self.fc2(x)
        x = self.dropout2(x)
        x = self.norm2(x + fc_input)

        return x.squeeze(1)

input_shape = 512
net = TransformerBlock()
x = torch.rand(10, 512)
if __name__ == "__main__":
    out_put = net(x)
    print(out_put.shape)
