import os
import pandas as pd
# from sklearn.metrics import r2_score
import numpy as np
import torch
from dataset import BPDatasetRam
from torch.utils.data import DataLoader
from models.unet import UNetPPGtoABP
from models.unet2dinput import UNet2DInput
from models.vnet import VNet
from models.transformernet import TransformerBlock
from make_annotation import MakeMainAnnotation
from snr_metric import Snr
from sklearn.metrics import mean_squared_error
from sklearn.metrics import mean_absolute_error
from sklearn.metrics import r2_score
from os.path import join




os.environ['CUDA_LAUNCH_BLOCKING'] = "1"
#TODO: chnage this dir to the correct dir of test txt files
inference_data_path = r"D:\PPG2ABP\data_for_test"
inference_data_annotation_path = r"D:\PPG2ABP\data_for_test/inference.csv"

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
# MakeMainAnnotation(inference_data_path, mode="inference")

# Load data and create a data loader
bp_data_inference = BPDatasetRam(inference_data_annotation_path, device, num_data=50)
bp_data_inference._load_data_to_RAM()
data_loader_inference = DataLoader(bp_data_inference, 1, shuffle=False)


stat_dict = torch.load(r"G:\PPG2ABP_TRAIN\PPG2ABP\scripts\checkpoint\s\dualnet\train_final\drop_0.08\UNetPPGtoABP\loss_MSELoss\lr_0.0005\batch_32\ConstantLR\epoch14.pth")

# model = Transformer(input_shape)
unet_model = UNet2DInput()
unet_model.load_state_dict(stat_dict['unet'])
unet_model.eval()
unet_model.to(device)

trans_model = UNetPPGtoABP()
trans_model.load_state_dict(stat_dict['transformer'])
trans_model.eval()
trans_model.to(device)

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

    each_sample_separate_info = {"r2_each_sample_seperate": [],
                                 "mse_loss_each_sample_saperate": [],
                                 "mae_loss_each_sample_saperate": [],
                                 "snr_before_each_sample_seperated": [],
                                 "snr_after_each_sample_seperated": [],
                                 "snr_improve_rate": []}

    total_samples_info = {"r2_total_samples": [], "mse_total_sample": [],
                          "mae_total_sample": [],
                          "snr_total_sample_before": [],
                          "snr_total_sample_after": []}

    # time.sleep(5)
    i = 0
    for batch_idx, (inputs, targets) in enumerate(data_loader_inference):
        inputs, targets = inputs.to(device), targets.to(device)
        outputs1 = trans_model(inputs.unsqueeze(1))
        inputs2 = torch.permute(torch.hstack((inputs.unsqueeze(1), outputs1.detach().unsqueeze(1))), (0, 1, 2))
        outputs2 = unet_model(inputs2)
        ############################################
        info = {"noisy signal": [], "reconstructed signal": []}
        info["noisy signal"] = np.array(inputs[0].cpu())
        info["reconstructed signal"] = np.array(outputs2[0].detach().cpu())
        info = pd.DataFrame(info)
        pd.DataFrame.to_csv(info,
                            r"G:\PPG2ABP_TRAIN\train_results\Denoise_net_final_train\plots\unet_unet_denoising.csv")



        np.savetxt(r"D:\PPG2ABP\data_for_training_split_shuffle\ppg_after_denoising", outputs2[0].detach().cpu())
        i +=i
        print(f"save number:{i}")
    #
    #     y_true += targets.cpu().numpy().tolist()
    #     y_pred += outputs2.cpu().detach().numpy().tolist()
    #     x += inputs.cpu().numpy().tolist()
    #     #TODO: check SNR formula
    #     snr_before = Snr(inputs.cpu(), targets.cpu())
    #     snr_after = Snr(outputs2.cpu().detach(), targets.cpu())
    #     np.append(r2_each_sample_seperate, r2_score(outputs2.cpu().detach(), targets.cpu()))
    #     np.append(mse_loss_each_sample_saperate , mean_squared_error(outputs2.cpu().detach(), targets.cpu()))
    #     np.append(mae_loss_each_sample_saperate ,mean_absolute_error(outputs2.cpu().detach(), targets.cpu()))
    #     np.append(snr_before_each_sample_seperated , snr_before)
    #     np.append(snr_after_each_sample_seperated , snr_after)
    #     np.append(snr_improve_rate, (snr_after - snr_before)/(snr_after))
    #
    #
    #     each_sample_separate_info["r2_each_sample_seperate"].append(r2_each_sample_seperate)
    #     each_sample_separate_info["mse_loss_each_sample_saperate"].append(mse_loss_each_sample_saperate)
    #     each_sample_separate_info["mae_loss_each_sample_saperate"].append(mae_loss_each_sample_saperate)
    #     each_sample_separate_info["snr_before_each_sample_seperated"].append(snr_before_each_sample_seperated)
    #     each_sample_separate_info["snr_after_each_sample_seperated"].append(snr_after_each_sample_seperated)
    #     each_sample_separate_info["snr_improve_rate"].append(snr_improve_rate)
    #
    # y_true = np.array(y_true)
    # y_pred = np.array(y_pred)
    # r2_total_samples =  r2_score(y_true, y_pred)
    # mse_total_sample = mean_squared_error(y_true, y_pred)
    # mae_total_sample = mean_absolute_error(y_true, y_pred)
    # snr_total_sample_before = Snr(x, y_pred)
    # snr_total_sample_after = Snr(y_true, y_pred)
    # total_samples_info["r2_total_samples"].append(r2_total_samples)
    #
    # total_samples_info["mse_total_sample"].append(mse_total_sample)
    #
    # total_samples_info["mae_total_sample"].append(mae_total_sample)
    # total_samples_info["snr_total_sample_before"].append(snr_total_sample_before)
    # total_samples_info["snr_total_sample_after"].append(snr_total_sample_after)
    #
    #
    #
    #
    # pd.DataFrame.to_csv(pd.DataFrame(each_sample_separate_info), "./chekpoint/ceach_sample_separate_info.csv")
    # pd.DataFrame.to_csv(pd.DataFrame(total_samples_info), "./chekpoint/total_samples_info.csv")


if __name__ == "__main__":
    inference()

