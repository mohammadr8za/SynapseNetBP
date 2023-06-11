import os
import time
import torch
import numpy as np
from dataset import BPDataset
from dataset import BPDatasetRam
from torch.utils.data import DataLoader
from models import mlp
import torch.nn as nn
from sklearn.metrics import r2_score
from transformernet import Transformer
from unet import UNetPPGtoABP
# from vnet import VNet1D
# import  dataset
from  snrloss import Snr
import matplotlib.pyplot as plt
os.environ['CUDA_LAUNCH_BLOCKING'] = "1"





# Parameters
train_data_annotation_path = r"/home/mohammad/Documents/Project/BP/data/PPGBP/mimic/Data_Train_Annotation.csv"
test_data_annotation_path = r"/home/mohammad/Documents/Project/BP/data/PPGBP/mimic/Data_test_Annotation.csv"
valid_data_annotation_path = r"/home/mohammad/Documents/Project/BP/data/PPGBP/mimic/Data_valid_Annotation.csv"
Batch_size = 4
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

# bp_data_train = BPDataset(train_data_annotation_path, device)
bp_data_train = BPDatasetRam(train_data_annotation_path, device)
bp_data_train._load_data_to_RAM()

bp_data_test = BPDatasetRam(test_data_annotation_path, device)
bp_data_test._load_data_to_RAM()

bp_data_valid = BPDatasetRam(valid_data_annotation_path, device)
bp_data_valid._load_data_to_RAM()

# data_loader_train = DataLoader(bp_data_train, Batch_size, shuffle=True)
# data_loader_train = DataLoader(bp_data_train, Batch_size, shuffle=True, num_workers=2, pin_memory=False)
# data_loader_test = DataLoader(bp_data_test, Batch_size, shuffle=True, num_workers=2, pin_memory=False)
# data_loader_valid = DataLoader(bp_data_valid, Batch_size, shuffle=True, num_workers=2, pin_memory=False)

data_loader_train = DataLoader(bp_data_train, Batch_size, shuffle=True)
data_loader_test = DataLoader(bp_data_test, Batch_size, shuffle=True)
data_loader_valid = DataLoader(bp_data_valid, Batch_size, shuffle=True)



# Define the model

# model = BPNeuralNet(win_time * fs, win_time * fs) #MLP
# model.to(device)

# model = Transformer(input_shape)
model = UNetPPGtoABP()
# model = VNet1D()
model.to(device)

# Define loss funtion + optimiser
loss_fn = nn.MSELoss()  # Mean Absolute Error
# loss_fn = nn.L1Loss()  # Mean Absolute Error
optimiser = torch.optim.Adam(model.parameters(), lr=LEARNING_RATE)


class AverageMeter(object):
    """Computes and stores the average and current value"""


    def __init__(self):
        self.reset()


    def reset(self):
        self.val = 0


        self.avg = 0
        self.sum = 0
        self.count =0

    def update(self, val, n=1):
        self.val = val
        self.sum += val * n
        self.count += n
        self.avg = self.sum / self.count
def train(eopch):



    print(f"*************training eopch:{epoch}***************")
    model.train()
    loss_total = AverageMeter()
    y_true = []
    y_pred = []
    # time.sleep(5)
    for batch_idx, (inputs, targets) in enumerate(data_loader_train):
        inputs, targets = inputs.to(device), targets.to(device)
        optimiser.zero_grad()
        outputs = model(inputs.unsqueeze(1))
        loss = loss_fn(outputs, targets)
        loss.backward()
        optimiser.step()
        loss_total.update(loss)
        y_true += targets.cpu().numpy().tolist()
        y_pred += outputs.cpu().detach().numpy().tolist()
        # print('batch \ totoal_data:',f'{(batch_idx/(len(data_loader_train))):.2}', f'loss: {loss_total.avg.item()}')
        # print(f"R2 : {r2_score(y_true, y_pred)} ")
        # if batch_idx == 100:
        #     break
    r2 = r2_score(y_true, y_pred)
    print(f"TRAIN R2 : {r2} ")

    return loss_total.avg.item(), r2

def valid(epoch, checkpoint):
    print(f"************* validation eopch:{epoch}***************")
    model.eval()
    loss_total = AverageMeter()
    y_true = []
    y_pred = []
    with torch.no_grad():
        for batch_idx, (inputs, targets) in enumerate(data_loader_valid):
            inputs, targets = inputs.to(device), targets.to(device)
            outputs = model(inputs.unsqueeze(1))
            loss = loss_fn(outputs, targets)
            loss_total.update(loss)
            y_true += targets.cpu().numpy().tolist()
            y_pred += outputs.cpu().detach().numpy().tolist()
            # if batch_idx==2:
            #     break
    r2 = r2_score(y_true, y_pred)
    print(f"VALID R2 : {r2} ")
    # Save checkpoint

    checkpoint.save(model=model, acc=r2, filename=f"BPmodel",loss=loss_total.avg.item(),
                    loss_type=loss_fn, batch_size=Batch_size,
                    optimizer=optimiser, lr_value=LEARNING_RATE, epoch=epoch)

    return loss_total.avg.item(), r2

class Checkpoint(object):
    def __init__(self):

        self.best_acc = 0.
        self.folder = 'chekpoint'
        os.makedirs(self.folder, exist_ok=True)


    def save(self, model, acc, loss, lr_value, batch_size, loss_type, optimizer, filename, epoch):

        # if acc > self.best_acc:
            print('Saving checkpoint...')


            state = {
                'net': model.state_dict(),
                'acc': acc,
                'epoch': epoch,
                'loss' : loss,
                'loss_type': loss_type,
                'lr_value': lr_value,
                'batch_size': batch_size,
                'optimizer': optimizer,
                'model_architecht': model
         }
            path = os.path.join(os.path.abspath(self.folder), filename + f'epoch{epoch}.pth')
            torch.save(state, path)
            self.best_acc = acc


def result():
    plt.figure(figsize=(15, 15))
    plt.plot(train_accuracy)
    plt.plot(valid_accuracy)
    plt.xlabel('epoch')
    plt.ylabel('Accuracy')
    plt.legend(['TRAINING', 'VALIDATION'], loc='upper right')
    # print()
    plt.savefig(fr"./chekpoint/BPModel_R2.png")
    plt.close()

    plt.figure(figsize=(15, 15))
    plt.plot(train_loss)
    plt.plot(valid_loss)
    plt.xlabel('epoch')
    plt.ylabel('Accuracy')
    plt.legend(['TRAINING', 'VALIDATION'], loc='upper right')
    # print()
    plt.savefig(fr"./chekpoint/BPModel_Loss.png")
    plt.close()

    print(f"************* Test eopch:{epoch}***************")
    print("Getting predictions from test set...")
    model.eval()
    loss_total = AverageMeter()
    y_true = []
    y_pred = []
    for batch_idx, (inputs, targets) in enumerate(data_loader_test):
        inputs, targets = inputs.to(device), targets.to(device)
    with torch.no_grad():
        for batch_idx, (inputs, targets) in enumerate(data_loader_test):
            inputs, targets = inputs.to(device), targets.to(device)
            outputs = model(inputs.unsqueeze(1))
            loss = loss_fn(outputs, targets)
            loss_total.update(loss)
            y_true += targets.cpu().numpy().tolist()
            y_pred += outputs.cpu().detach().numpy().tolist()
            # if batch_idx==2:
            #     break
    r2 = r2_score(y_true, y_pred)
    print(f"TEST R2 : {r2} ")


if __name__ == "__main__":
    checkpoint = Checkpoint()
    start, end = 1, 200
    train_loss = []
    valid_loss = []
    train_accuracy = []
    valid_accuracy = []
    for epoch in range (start, end):
        TrainLoss, TrainAccuracy = train(epoch)
        ValidLoss, ValidAccuracy = valid(epoch, checkpoint)
        train_loss.append(TrainLoss)
        train_accuracy.append(TrainAccuracy)
        valid_loss.append(ValidLoss)
        valid_accuracy.append(ValidAccuracy)
        result()