import os
import torch
from dataset import BPDatasetRam
from torch.utils.data import DataLoader
from unet import UNetPPGtoABP
from make_annotation import MakeMainAnnotation

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
data_loader_test = DataLoader(bp_data_inference, Batch_size, shuffle=True)
stat_dict = torch.load(r"/home/mohammad/Documents/Project/PPG2ABP/scripts/chekpoint/BPmodelepoch1.pth")
# model = Transformer(input_shape)
model = UNetPPGtoABP()
model.load_state_dict(stat_dict['net'])
model.eval()
model.to(device)


def inference(eopch):

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
        inference((epoch))

