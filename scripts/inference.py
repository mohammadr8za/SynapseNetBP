import os
import time
import torch
import numpy as np
from dataset import BPDataset
from dataset import BPDatasetRam
from torch.utils.data import DataLoader
from models import BPNeuralNet
import torch.nn as nn
from sklearn.metrics import r2_score
from transformernet import Transformer
from unet import UNetPPGtoABP
# from vnet import VNet1D
# import  dataset
from  snrloss import snr_loss
os.environ['CUDA_LAUNCH_BLOCKING'] = "1"



test_data_annotation_path = r"D:\data\PPGBP\mimic\Data_test_Annotation.csv"

Batch_size = 1
fs = 125
win_time = 5  # seconds
LEARNING_RATE = .0001
input_shape = fs * win_time
torch.manual_seed(1234)
torch.cuda.manual_seed(1234)


if torch.cuda.is_available():

    device = "cuda:0"
    # device = "cpu"
else:
    device = "cpu"
    print(f"Using device {device}")


bp_data_test = BPDatasetRam(test_data_annotation_path, device)
bp_data_test._load_data_to_RAM()
data_loader_test = DataLoader(bp_data_test, Batch_size, shuffle=True)
stat_dict = torch.load("D:\PythonProjects\PPGBP\Models\chekpoint\BPmodelepoch199.pth")
# model = Transformer(input_shape)
model = UNetPPGtoABP()
model.load_state_dict(stat_dict['net'])
model.eval()
model.to(device)


def test(eopch):

    y_true = []
    y_pred = []
    # time.sleep(5)
    for batch_idx, (inputs, targets) in enumerate(data_loader_test):
        inputs, targets = inputs.to(device), targets.to(device)


        outputs = model(inputs.unsqueeze(1))


        y_true += targets.cpu().numpy().tolist()
        y_pred += outputs.cpu().detach().numpy().tolist()


if __name__ == "__main__":
    start, end = 0, 100
    for epoch in range (start, end):
        print(f"*************eopch:{epoch}***************")
        test((epoch))

