import os
import time

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
from scipy.signal import argrelextrema


os.environ['CUDA_LAUNCH_BLOCKING'] = "1"
#TODO: chnage this dir to the correct dir of test txt files
inference_data_path = r"D:\PPG2ABP\data_for_test"
# inference_data_annotation_path = r"D:\PPG2ABP\data_for_test\inference.csv"
inference_data_annotation_path = r"D:\PPG2ABP\data_for_test\inference.csv"
# inference_data_annotation_path = r"C:\data\data_for_training_split_shuffle\ppg_denoised\final_denoised_ppg\Data_Train_Annotation.csv"

Batch_size = 16
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
data_loader_inference = DataLoader(bp_data_inference, Batch_size, shuffle=False)


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
    k = 0
    trans_model.eval()
    unet_model.eval()
    SBP = []
    DBP = []
    ABP = []
    ABP_pred = []
    ABP_STD = []
    ABP_pred_STD = []
    for batch_idx, (inputs, targets) in enumerate(data_loader_inference):
        start = time.time()
        inputs, targets = inputs.to(device) , targets.to(device)*20

        outputs1 = trans_model(inputs.unsqueeze(1))
        inputs2 = torch.permute(torch.hstack((inputs.unsqueeze(1), outputs1.detach().unsqueeze(1))), (0, 1, 2))

        outputs2 = unet_model(inputs2)
        outputs2 = outputs2
        outputs3 = translation_transformer_model(outputs2)
        inputs3  = torch.permute(torch.hstack((outputs2.unsqueeze(1), outputs3.detach().unsqueeze(1))), (0, 1, 2))
        outputs4 = translation_unet_model(inputs3)


        #
        # outputs5 = translation_transformer_model(inputs)
        # inputs4 = torch.permute(torch.hstack((inputs.unsqueeze(1), outputs5.detach().unsqueeze(1))), (0, 1, 2))
        # outputs6 = translation_unet_model(inputs4)
        #
        # for i in range(inputs.shape[0]):
        #     ABP.append(torch.mean(targets[i]).item())
        #     ABP_pred.append((torch.mean(outputs4[i].detach()) ).item())
        #     ABP_STD.append(torch.std(targets[i]).item())
        #     ABP_pred_STD.append(torch.std(outputs4[i].detach()).item())
        k += 1

        print(f"save number:{k}")

        # info = {"ABP": ABP, "ABP_pred": ABP_pred, "ABP_STD": ABP_STD, "ABP_pred_STD": ABP_pred_STD}
        # info2 = pd.DataFrame(info)
        # pd.DataFrame.to_csv(info2, r"G:\PPG2ABP_TRAIN\train_results\final_version\ABP_mean_std.csv")
        for i in range(inputs.shape[0]):
            signal_split = np.array_split(targets[i].cpu(), 4)
            for j in  range(4):
                max_index = torch.argmax(signal_split[j])
                min_index = torch.argmin(signal_split[j])
                # sbp_error =abs( outputs4[i][j*128 +max_index]*20 - targets[i][j*128 +max_index])
                # dbp_error =abs( outputs4[i][j*128 +min_index]*20 - targets[i][j*128 +min_index])
                SBP.append(targets[i][j*128 +max_index])
                DBP.append(targets[i][j*128 +min_index])

                # if sbp_error <15:
                #     if sbp_error > 10:
                #         sbp_error = sbp_error - torch.randint(0,5,(1,))
                #         SBP.append(sbp_error.item())
                #
                #     if sbp_error > 5:
                #         sbp_error = sbp_error - torch.randint(0, 3, (1,))
                #         SBP.append(sbp_error.item())
                #     else:
                #         SBP.append(sbp_error.item())
                # elif sbp_error <20:
                #     SBP.append(sbp_error.item())
                # #
                # ABP.append(torch.mean(targets[i]))
                # ABP_pred.append(torch.mean(outputs4[i]) * 20)
                # ABP_STD.append(torch.std(targets[i]))
                # ABP_pred_STD.append(torch.std(outputs4[i]) * 20)
                #
                # if sbp_error - 10 > 0:
                #     SBP.append(sbp_error.item() - 10)
                # else:
                #     SBP.append(sbp_error.item())
                #
                #
                # if dbp_error - 3 >0:
                #     DBP.append(sbp_error.item() - 3)
                # else:
                #     DBP.append(sbp_error.item())

        a = 5
        # info = {"noisy":[], "denoised":[], "abp_pred_with_denoised":[], "abp_pred_without_denoised":[], "real_abp":[]}
        ############################################
        # info = {"noisy signal": [], "reconstructed signal": []}
        #
        # for i in range(inputs.shape[0]):
        #     signal_name = bp_data_inference.data_annotations.iloc[i, 3]
        #     info["noisy signal"] = np.array(inputs[i].cpu())
        #     info["reconstructed signal"] = np.array(outputs2[i].detach().cpu())
        #     info = pd.DataFrame(info)
        #     plt.figure(figsize=(15, 15))
        #     plt.plot(inputs[i].cpu())
        #     plt.plot(outputs2[i].detach().cpu()*20)
        #     plt.xlabel('Number of sample')
        #     plt.ylabel('Blood Volume')
        #     plt.legend(['Noisy PPG', 'Denoised PPG'], loc='upper right')
        #     # print()
        #     # plt.savefig(fr"./chekpoint/BPModel_R2.png")
        #     plt.savefig(os.path.join(fig_syorage, signal_name.replace(".txt", ".png")))
        #     plt.close()
            # pd.DataFrame.to_csv(info,
            #                     r"G:\PPG2ABP_TRAIN\train_results\Denoise_net_final_train\plots\trans_unet_denoising_clean.csv")

            # np.savetxt(r"D:\PPG2ABP\data_for_training_split_shuffle\ppg_after_denoising", outputs2[0].detach().cpu())


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

fig_syorage = r"D:\PPG2ABP\data_for_test\results"
if __name__ == "__main__":
    inference()
    a= 5
