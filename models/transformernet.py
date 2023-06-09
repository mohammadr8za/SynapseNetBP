import torch
import torch.nn as nn

class Transformer(nn.Module):
    def __init__(self, input_shape, num_transformer_block=40, num_attention_heads=80, ff_dim=512):
        super(Transformer, self).__init__()
        self.input_shape = input_shape
        self.num_transformer_block = num_transformer_block
        self.num_attention_heads = num_attention_heads
        self.ff_dim = ff_dim

        #input_layer
        self.input_layer = nn.Linear(input_shape, input_shape)

        #transformer_block
        transformer_blocks = []
        for i in range(num_transformer_block):
            self.transformer = nn.Sequential(*transformer_blocks)
        #outputlayer
        self.output_layer = nn.Linear(input_shape, input_shape)

    def forward(self,x):
        x = self.input_layer(x)
        x = x.permute(1, 0)
        x = self.transformer(x)
        x = x.permute(1, 0)
        x = self.output_layer(x)
        return x

# input_shape = 625
# net = Transformer(input_shape)
# x = torch.rand(10, 625)
# if __name__ == "__main__":
#     out_put = net(x)
#     print(out_put.shape)
