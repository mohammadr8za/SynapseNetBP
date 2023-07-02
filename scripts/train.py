import os
import torch
from dataset import BPDatasetRam
from torch.utils.data import DataLoader
import torch.nn as nn
from sklearn.metrics import r2_score
from models.unet import UNetPPGtoABP
from models.vnet import VNet
from models.transformernet import TransformerBlock
# from vnet import VNet1D
# import  dataset
import matplotlib.pyplot as plt
import argparse
from torch.utils.tensorboard import SummaryWriter
from torch.nn import MSELoss, L1Loss
from os import listdir
from torch.optim.lr_scheduler import  StepLR, CosineAnnealingLR, OneCycleLR, ReduceLROnPlateau


os.environ['CUDA_LAUNCH_BLOCKING'] = "1"



# Parameters
main_data_path = r"D:\PPG2ABP\data_for_train"
DATASETS_PATH = r"D:\PPG2ABP\data_for_training_split_shuffle\ppg_noisy"
#TODO: get list of data sets for train, like noisy_scale_100 and etc.
data_folder_list = listdir(DATASETS_PATH)
Batch_size = 4
fs = 125
win_time = 5  # seconds
LEARNING_RATE = .0001
input_shape = fs * win_time
torch.manual_seed(1234)
torch.cuda.manual_seed(1234)
configs = {"models":[  UNetPPGtoABP()], "loss_func":[MSELoss(), L1Loss()], "lr":[.0001, .001],
           "optimizer":["adam", "adagrad"],"batch_size":[4, 16, 64, 128], "drop_out":[.1],
           "lr_scheduler":["Cosinanlealing", "ReduceLR", "StepR"]}

if torch.cuda.is_available():

    device = "cuda:0"
    # device = "cpu"
else:
    device = "cpu"
    print(f"Using device {device}")

def load_data(train_data_annotation_path ,valid_data_annotation_path ):

    bp_data_train = BPDatasetRam(train_data_annotation_path, device)
    bp_data_train._load_data_to_RAM()

    bp_data_valid = BPDatasetRam(valid_data_annotation_path, device)
    bp_data_valid._load_data_to_RAM()

    return bp_data_train, bp_data_valid





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
def train(model, loss_fn, optimiser, data_loader_train,epoch, scheduler):



    print(f"*************training eopch:{epoch}***************")
    model.train()
    loss_total = AverageMeter()
    y_true = []
    y_pred = []

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
        scheduler.step()
        # print('batch \ totoal_data:',f'{(batch_idx/(len(data_loader_train))):.2}', f'loss: {loss_total.avg.item()}')
        # print(f"R2 : {r2_score(y_true, y_pred)} ")
        # if batch_idx == 100:
        #     break
    r2 = r2_score(y_true, y_pred)
    print(f"TRAIN R2 : {r2} ")


    return loss_total.avg.item(), r2

def valid(model, loss_fn, data_loader_valid, optimiser,epoch, checkpoint, data_name, check_dir):
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

    checkpoint.save(model=model, acc=r2, filename=f"BPmodel",loss=loss_total.avg.item(),
                    loss_type=loss_fn, batch_size=Batch_size,
                    optimizer=optimiser, lr_value=LEARNING_RATE, epoch=epoch,data_name= data_name, check_dir=check_dir)

    return loss_total.avg.item(), r2

class Checkpoint(object):
    def __init__(self):

        self.best_acc = 0.
        self.folder = 'chekpoint'
        os.makedirs(self.folder, exist_ok=True)



    def save(self, model, acc, loss, lr_value, batch_size, loss_type, optimizer, filename, epoch, data_name, check_dir):
            # os.makedirs(os.path.join(self.folder, data_name))
        # if acc > self.best_acc:
            print('Saving checkpoint...')


            state = {
                'net': model.state_dict(),
                # 'acc': acc,
                # 'epoch': epoch,
                # 'loss' : loss,
                # 'loss_type': loss_type,
                # 'lr_value': lr_value,
                # 'batch_size': batch_size,
                # 'optimizer': optimizer,
                'model_architecht': model
         }
            path = os.path.join(check_dir , f'epoch{epoch}.pth')
            torch.save(state, path)
            self.best_acc = acc


def result(train_loss, valid_loss, train_accuracy, valid_accuracy, epoch, data_name, check_dir):
    plt.figure(figsize=(15, 15))
    plt.plot(train_accuracy)
    plt.plot(valid_accuracy)
    plt.xlabel('epoch')
    plt.ylabel('Accuracy')
    plt.legend(['TRAINING', 'VALIDATION'], loc='upper right')
    # print()
    # plt.savefig(fr"./chekpoint/BPModel_R2.png")
    plt.savefig(os.path.join(check_dir, "BPModel_R2.png"))
    plt.close()

    plt.figure(figsize=(15, 15))
    plt.plot(train_loss)
    plt.plot(valid_loss)
    plt.xlabel('epoch')
    plt.ylabel('Accuracy')
    plt.legend(['TRAINING', 'VALIDATION'], loc='upper right')
    # print()
    plt.savefig(os.path.join(check_dir, "BPModel_Loss.png"))
    # plt.savefig(fr"./chekpoint/BPModel_Loss.png")
    plt.close()



def parse_args():
    parser = argparse.ArgumentParser(description='Train a deep model')
    parser.add_argument('--data_dir', type=str, default='data', help='Path to training data directory')
    parser.add_argument('--start_epoch', type=int, default=0, help='Start epoch number')
    parser.add_argument('--end_epoch', type=int, default=20, help='End epoch number')
    parser.add_argument('--learning_rate', type=float, default=0.001, help='Learning rate for optimizer')
    parser.add_argument('--batch_size', type=int, default=32, help='Batch size for training')
    return parser.parse_args()
def main():

    for i in data_folder_list:
        data_train_path = os.path.join(DATASETS_PATH, i, "Data_Train_Annotation.csv")
        data_valid_path = os.path.join(DATASETS_PATH, i, "Data_valid_Annotation.csv")
        bp_data_train, bp_data_valid = load_data(data_train_path, data_valid_path)
        os.makedirs(os.path.join("chekpoint", i), exist_ok=True)
        start, end = 0, 20
        for drop in configs["drop_out"]:
            for model_type in configs["models"]:
                model = model_type
                model.to(device)
                for loss_func in configs["loss_func"]:
                    loss_fn = loss_func
                    for lr_rate in configs["lr"]:
                        optimiser = torch.optim.Adam(model.parameters(), lr=lr_rate)
                        for scheduler_type in configs["lr_scheduler"]:
                            if scheduler_type == "StepR":
                                scheduler = StepLR(optimiser, step_size=10, gamma=0.1)
                            elif scheduler_type == "Cosinanlealing":
                                scheduler = CosineAnnealingLR(optimiser, T_max=end, eta_min=0)
                            elif scheduler_type == "ReduceLR":
                                scheduler = ReduceLROnPlateau(optimiser, mode='min', factor=0.1, patience=5, verbose=True)
                            for batch_size in configs["batch_size"]:

                                data_loader_train = DataLoader(bp_data_train, batch_size, shuffle=True)
                                data_loader_valid = DataLoader(bp_data_valid, batch_size, shuffle=True)
                                args = parse_args()
                                checkpoint = Checkpoint()
                                checkpoint_path = os.path.join("checkpoint", i, f"drop_{drop}", f"{str(model._get_name())}",
                                                               f"loss_{loss_fn._get_name() }",f"lr_{lr_rate}",f"batch_{batch_size}",
                                                               scheduler_type)
                                os.makedirs(checkpoint_path, exist_ok=True)
                                os.makedirs(os.path.join(checkpoint_path, "run"), exist_ok=True)

                                log_dir = os.path.join(checkpoint_path, "run")

                                writer = SummaryWriter(log_dir)

                                train_loss = []
                                valid_loss = []
                                train_accuracy = []
                                valid_accuracy = []
                                for epoch in range(start, end):
                                    TrainLoss, TrainAccuracy = train(model, loss_fn, optimiser, data_loader_train,
                                                                     epoch,scheduler_type= scheduler)
                                    ValidLoss, ValidAccuracy = valid(model, loss_fn, data_loader_valid, optimiser,
                                                                     epoch, checkpoint, i,
                                                                     checkpoint_path)
                                    train_loss.append(TrainLoss)
                                    train_accuracy.append(TrainAccuracy)
                                    valid_loss.append(ValidLoss)
                                    valid_accuracy.append(ValidAccuracy)
                                    result(train_loss, valid_loss, train_accuracy, valid_accuracy, epoch, i, checkpoint_path)
                                    writer.add_scalars(main_tag="accuracy",
                                                       tag_scalar_dict={"train_accuracy": TrainAccuracy,
                                                                        "valid_accuracy": ValidAccuracy},
                                                       global_step=epoch)
                                    writer.add_scalars(main_tag="loss",
                                                       tag_scalar_dict={"train_loss": TrainLoss,
                                                                        "valid_loss": ValidLoss},
                                                       global_step=epoch)
                                writer.close()

if __name__ == "__main__":
    main()