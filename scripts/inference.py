import os

import pandas as pd
# from sklearn.metrics import r2_score
import numpy as np
import torch
from dataset import BPDatasetRam
from torch.utils.data import DataLoader
from unet import UNetPPGtoABP
from make_annotation import MakeMainAnnotation
from torch.nn import MSELoss
from torch.nn import L1Loss
from snrloss import Snr
from sklearn.metrics import mean_squared_error
from sklearn.metrics import mean_absolute_error
from sklearn.metrics import r2_score





os.environ['CUDA_LAUNCH_BLOCKING'] = "1"
#TODO: chnage this dir to the correct dir of test txt files
inference_data_path = r"/home/mohammad/Documents/Project/BP/data/PPGBP/mimic"
inference_data_annotation_path = r"/home/mohammad/Documents/Project/BP/data/PPGBP/mimic/inference.csv"

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


# Make annotaion file for inference data
MakeMainAnnotation(inference_data_path, mode="inference")

# Load data and create a data loader
bp_data_inference = BPDatasetRam(inference_data_annotation_path, device)
bp_data_inference._load_data_to_RAM()
data_loader_inference = DataLoader(bp_data_inference, Batch_size, shuffle=True)
stat_dict = torch.load(r"/home/mohammad/Documents/Project/PPG2ABP/scripts/chekpoint/BPmodelepoch1.pth")
# model = Transformer(input_shape)
model = UNetPPGtoABP()
model.load_state_dict(stat_dict['net'])
model.eval()
model.to(device)


def inference():
    r2_each_sample_seperate = np.empty((1,))
    mse_loss_each_sample_saperate = np.empty((1,))
    mae_loss_each_sample_saperate = np.empty((1,))
    snr_before_each_sample_seperated = np.empty((1,))
    snr_after_each_sample_seperated = np.empty((1,))
    snr_improve_rate =  np.empty((1,))

    y_true = []
    y_pred = []
    x = []
    # time.sleep(5)
    for batch_idx, (inputs, targets) in enumerate(data_loader_inference):
        inputs, targets = inputs.to(device), targets.to(device)
        outputs = model(inputs.unsqueeze(1))
        y_true += targets.cpu().numpy().tolist()
        y_pred += outputs.cpu().detach().numpy().tolist()
        x += inputs.cpu().numpy().tolist()
        #TODO: check SNR formula
        snr_before = Snr(inputs.cpu(), targets.cpu())
        snr_after = Snr(outputs.cpu().detach(), targets.cpu())
        np.append(r2_each_sample_seperate, r2_score(outputs.cpu().detach(), targets.cpu()))
        np.append(mse_loss_each_sample_saperate , mean_squared_error(outputs.cpu().detach(), targets.cpu()))
        np.append(mae_loss_each_sample_saperate ,mean_absolute_error(outputs.cpu().detach(), targets.cpu()))
        np.append(snr_before_each_sample_seperated , snr_before)
        np.append(snr_after_each_sample_seperated , snr_after)
        np.append(snr_improve_rate, (snr_after - snr_before)/(snr_after))

    y_true = np.array(y_true)
    y_pred = np.array(y_pred)
    r2_total_samples =  r2_score(y_true, y_pred)
    mse_total_sample = mean_squared_error(y_true, y_pred)
    mae_total_sample = mean_absolute_error(y_true, y_pred)
    snr_total_sample_before = Snr(x, y_pred)
    snr_total_sample_after = Snr(y_true, y_pred)

    each_sample_separate_info = {"r2_each_sample_seperate":r2_each_sample_seperate,"mse_loss_each_sample_saperate":mse_loss_each_sample_saperate,
                                 "mae_loss_each_sample_saperate":mae_loss_each_sample_saperate, "snr_before_each_sample_seperated": snr_before_each_sample_seperated,
                                  "snr_after_each_sample_seperated": snr_after_each_sample_seperated, "snr_improve_rate": snr_improve_rate}

    total_samples_info = {"r2_total_samples": r2_total_samples, "mse_total_sample": mse_total_sample, "mae_total_sample": mae_total_sample,
                          "snr_total_sample_before": snr_total_sample_before, "snr_total_sample_after": snr_total_sample_after}


    pd.DataFrame.to_csv(pd.DataFrame(each_sample_separate_info), "each_sample_separate_info.csv")
    pd.DataFrame.to_csv(pd.DataFrame(total_samples_info), "total_samples_info.csv")


if __name__ == "__main__":
    inference()

