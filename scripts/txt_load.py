import torch
import struct
import numpy as np

if torch.cuda.is_available():
    device = "cuda:0"
else:
    device = "cpu"
    print(f"Using device {device}")


def TxtLoad(path):
    data_array = np.loadtxt(path)
    return data_array


# data_path = r'D:\data\PPGBP\mimic\abp\1-0-0.txt'
# if __name__ == '__main__':
#     data = TxtLoad(data_path)
#     print(data.shape)
