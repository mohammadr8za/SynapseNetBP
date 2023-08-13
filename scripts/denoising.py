import os
import numpy as np
import torch
from models.unet import UNetPPGtoABP
from models.unet2dinput import UNet2DInput
from os.path import join
os.environ['CUDA_LAUNCH_BLOCKING'] = "1"



torch.manual_seed(1234)
torch.cuda.manual_seed(1234)

if torch.cuda.is_available():

    device = "cuda:0"
    # device = "cpu"
else:
    device = "cpu"
    print(f"Using device {device}")

def make_model(PRETRAIN_MODEL_PATH):
    stat_dict = torch.load(PRETRAIN_MODEL_PATH)

    # model = Transformer(input_shape)
    unet_model = UNet2DInput()
    unet_model.load_state_dict(stat_dict['unet'])
    unet_model.eval()
    unet_model.to(device)

    trans_model = UNetPPGtoABP()
    trans_model.load_state_dict(stat_dict['transformer'])
    trans_model.eval()
    trans_model.to(device)

    return trans_model, unet_model
def inference(trans_model, unet_model, noisy_dataset_path, denoised_ppg_path):
    i = 0
    noisy_data_list = os.listdir(noisy_dataset_path)
    for i in noisy_data_list:
        inputs = np.loadtxt(join(noisy_dataset_path, i))
        inputs = inputs.to(device)
        outputs1 = trans_model(inputs.unsqueeze(1))
        inputs2 = torch.permute(torch.hstack((inputs.unsqueeze(1), outputs1.detach().unsqueeze(1))), (0, 1, 2))
        outputs2 = unet_model(inputs2)
        np.savetxt(join(denoised_ppg_path, i), outputs2[0].detach().cpu())
        i +=i
        print(f"save number:{i}")





PRETRAIN_MODEL_PATH = r"G:\PPG2ABP_TRAIN\PPG2ABP\scripts\checkpoint\s\dualnet\train_final\drop_0.08\UNetPPGtoABP\loss_MSELoss\lr_0.0005\batch_32\ConstantLR\epoch14.pth"
NOISY_PPG_PATH = r"D:\PPG2ABP\data_for_test"
DENOISED_PPG_PATH = r""


if __name__ == "__main__":
    transformer, unet = make_model(PRETRAIN_MODEL_PATH)
    inference(transformer, unet, NOISY_PPG_PATH, DENOISED_PPG_PATH)

