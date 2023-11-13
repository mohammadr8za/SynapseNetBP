import os
import pandas as pd
# from sklearn.metrics import r2_score
import numpy as np
import torch
from dataset import BPDatasetRam, BPDataset
from torch.utils.data import DataLoader
from models.unet import UNetPPGtoABP
from models.unet2dinput import UNet2DInput, UNet2DInput2
from models.vnet import VNet
from models.transformernet import TransformerBlock
from models.unet3 import UNetPPGtoABP3
from make_annotation import MakeMainAnnotation
from snr_metric import Snr
from sklearn.metrics import mean_squared_error
from sklearn.metrics import mean_absolute_error
from sklearn.metrics import r2_score
from os.path import join
import matplotlib.pyplot as plt
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
from os.path import join
from pathlib import Path
from scipy.signal import find_peaks





os.environ['CUDA_LAUNCH_BLOCKING'] = "1"
#TODO: chnage this dir to the correct dir of test txt files
inference_data_path = r"D:\PPG2ABP\data_for_test"
# inference_data_annotation_path = r"D:\PPG2ABP\data_for_test\inference.csv"
inference_data_annotation_path = r"D:\PPG2ABP\data_for_test\inference.csv"

Batch_size = 1
fs = 125
win_time = 5  # seconds
LEARNING_RATE = .0001
input_shape = fs * win_time
torch.manual_seed(1234)
torch.cuda.manual_seed(1234)

if torch.cuda.is_available():

    device = "cpu"
    # device = "cpu"
else:
    device = "cpu"
    print(f"Using device {device}")


# Make annotaion file for inference data
# MakeMainAnnotation(inference_data_path, mode="inference")

# Load data and create a data loader
bp_data_inference = BPDataset(inference_data_annotation_path, device)
# bp_data_inference._load_data_to_RAM()
data_loader_inference = DataLoader(bp_data_inference, 50, shuffle=False)


stat_dict = torch.load(r"G:\PPG2ABP_TRAIN\PPG2ABP\scripts\checkpoint\s\dualnet\train_final\drop_0.085\TransformerBlock\loss_MSELoss\denoising_best\batch_16\ConstantLR\epoch5.pth")
translation_stat_dict = torch.load(r"G:\PPG2ABP_TRAIN\PPG2ABP\scripts\checkpoint\s\dualnet\final_denoised_ppg\drop_0.085\TransformerBlock\loss_MSELoss\lr_0.0001\batch_32\ConstantLR\epoch74.pth")

#DENOISNG
# model = Transformer(input_shape)
unet_model = UNet2DInput()
unet_model.load_state_dict(stat_dict['unet'])
unet_model.eval()
unet_model.to(device)

trans_model = TransformerBlock()
trans_model.load_state_dict(stat_dict['transformer'])
trans_model.eval()
trans_model.to(device)

def blood_pressure_single_abp(abp_chunk, plot=False):

    # Systolic Blood Pressure (SBP)
    sys_peaks, _ = find_peaks(abp_chunk, prominence=10, width=10)

    sbp = int(np.median(abp_chunk[sys_peaks]))

    # Diastolic Blood Pressure (DBP)
    abp_chunk_reverse = (-abp_chunk) + np.max(abp_chunk + 10)
    dias_peaks, _ = find_peaks(abp_chunk_reverse, prominence=10, width=10)

    dbp = int(np.median(abp_chunk[dias_peaks]))

    return sbp, dbp






#TRANSLATION
# model = Transformer(input_shape)
translation_unet_model = translation_stat_dict['unet_model_architecht']
translation_unet_model.load_state_dict(translation_stat_dict['unet'])
translation_unet_model.eval()
translation_unet_model.to(device)

translation_transformer_model = translation_stat_dict['transformer_model_architecht']
translation_transformer_model.load_state_dict(translation_stat_dict['transformer'])
translation_transformer_model.eval()
translation_transformer_model.to(device)

def inference():


    i = 0
    trans_model.eval()
    unet_model.eval()
    for batch_idx, (inputs, targets) in enumerate(data_loader_inference):
        inputs, targets = inputs.to(device) , targets.to(device)*20

        outputs1 = trans_model(inputs.unsqueeze(1))
        inputs2 = torch.permute(torch.hstack((inputs.unsqueeze(1), outputs1.detach().unsqueeze(1))), (0, 1, 2))

        outputs2 = unet_model(inputs2)
        outputs2 = outputs2
        outputs3 = translation_transformer_model(outputs2)
        inputs3  = torch.permute(torch.hstack((outputs2.unsqueeze(1), outputs3.detach().unsqueeze(1))), (0, 1, 2))
        outputs4 = translation_unet_model(inputs3) * 20

        # sbp, dbp =blood_pressure_single_abp(outputs4[0].detach().cpu())
        i +=i
        print(f"save number:{i}")





fig_syorage = r"D:\PPG2ABP\data_for_test\results"
if __name__ == "__main__":
    inference()

