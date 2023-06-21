import os
import torch
from dataset import BPDatasetRam
from torch.utils.data import DataLoader
import torch.nn as nn
from sklearn.metrics import r2_score
from models.unet import UNetPPGtoABP
# from vnet import VNet1D
# import  dataset
import matplotlib.pyplot as plt
import argparse

os.environ['CUDA_LAUNCH_BLOCKING'] = "1"



# Parameters
main_data_path = r"D:\PPG2ABP\data_for_train"

#TODO: get list of data sets for train, like noisy_scale_100 and etc.
data_folder_list = []
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

def load_data(train_data_annotation_path ,valid_data_annotation_path ):
    # bp_data_train = BPDataset(train_data_annotation_path, device)
    bp_data_train = BPDatasetRam(train_data_annotation_path, device)
    bp_data_train._load_data_to_RAM()

    # bp_data_test = BPDatasetRam(test_data_annotation_path, device)
    # bp_data_test._load_data_to_RAM()

    bp_data_valid = BPDatasetRam(valid_data_annotation_path, device)
    bp_data_valid._load_data_to_RAM()

    # data_loader_train = DataLoader(bp_data_train, Batch_size, shuffle=True)
    # data_loader_train = DataLoader(bp_data_train, Batch_size, shuffle=True, num_workers=2, pin_memory=False)
    # data_loader_test = DataLoader(bp_data_test, Batch_size, shuffle=True, num_workers=2, pin_memory=False)
    # data_loader_valid = DataLoader(bp_data_valid, Batch_size, shuffle=True, num_workers=2, pin_memory=False)

    data_loader_train = DataLoader(bp_data_train, Batch_size, shuffle=True)
    # data_loader_test = DataLoader(bp_data_test, Batch_size, shuffle=True)
    data_loader_valid = DataLoader(bp_data_valid, Batch_size, shuffle=True)

    return data_loader_train, data_loader_valid

# Define the model

# model = BPNeuralNet(win_time * fs, win_time * fs) #MLP
# model.to(device)

# model = Transformer(input_shape)
# state_dict = torch.load(r"D:\PythonProjects\Git\PPG2ABP\scripts\chekpoint\BPmodelepoch12.pth")
model = UNetPPGtoABP()
# model.load_state_dict(state_dict["net"])
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
def train(epoch, data_loader_train):



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

def valid(epoch, checkpoint, data_loader_valid, data_name):
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
                    optimizer=optimiser, lr_value=LEARNING_RATE, epoch=epoch,data_name= data_name)

    return loss_total.avg.item(), r2

class Checkpoint(object):
    def __init__(self):

        self.best_acc = 0.
        self.folder = 'chekpoint'
        os.makedirs(self.folder, exist_ok=True)



    def save(self, model, acc, loss, lr_value, batch_size, loss_type, optimizer, filename, epoch, data_name):
            os.makedirs(os.path.join(self.folder, data_name))
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
            path = os.path.join(os.path.abspath(self.folder), filename + data_name+  f'epoch{epoch}.pth')
            torch.save(state, path)
            self.best_acc = acc


def result(train_loss, valid_loss, train_accuracy, valid_accuracy, epoch, data_name):
    plt.figure(figsize=(15, 15))
    plt.plot(train_accuracy)
    plt.plot(valid_accuracy)
    plt.xlabel('epoch')
    plt.ylabel('Accuracy')
    plt.legend(['TRAINING', 'VALIDATION'], loc='upper right')
    # print()
    # plt.savefig(fr"./chekpoint/BPModel_R2.png")
    plt.savefig(os.path.join("./chekpoint"), data_name, "BPModel_R2.png")
    plt.close()

    plt.figure(figsize=(15, 15))
    plt.plot(train_loss)
    plt.plot(valid_loss)
    plt.xlabel('epoch')
    plt.ylabel('Accuracy')
    plt.legend(['TRAINING', 'VALIDATION'], loc='upper right')
    # print()
    plt.savefig(os.path.join("./chekpoint"), data_name, "BPModel_Loss.png")
    # plt.savefig(fr"./chekpoint/BPModel_Loss.png")
    plt.close()

    print(f"************* Test eopch:{epoch}***************")
    print("Getting predictions from test set...")
    model.eval()
    loss_total = AverageMeter()
    y_true = []
    y_pred = []
    # for batch_idx, (inputs, targets) in enumerate(data_loader_test):
    #     inputs, targets = inputs.to(device), targets.to(device)
    # with torch.no_grad():
    #     for batch_idx, (inputs, targets) in enumerate(data_loader_test):
    #         inputs, targets = inputs.to(device), targets.to(device)
    #         outputs = model(inputs.unsqueeze(1))
    #         loss = loss_fn(outputs, targets)
    #         loss_total.update(loss)
    #         y_true += targets.cpu().numpy().tolist()
    #         y_pred += outputs.cpu().detach().numpy().tolist()
    #         # if batch_idx==2:
    #         #     break
    # r2 = r2_score(y_true, y_pred)
    # print(f"TEST R2 : {r2} ")
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
        data_train_path = os.path.join(main_data_path, i, "Data_Train_Annotation.csv")
        data_valid_path = os.path.join(main_data_path, i, "Data_valid_Annotation.csv")


        data_loader_train, data_loader_valid = load_data(data_train_path, data_valid_path)
        args = parse_args()
        checkpoint = Checkpoint()
        start, end = 0, 20
        train_loss = []
        valid_loss = []
        train_accuracy = []
        valid_accuracy = []
        for epoch in range(start, end):
            TrainLoss, TrainAccuracy = train(epoch, data_loader_train)
            ValidLoss, ValidAccuracy = valid(epoch, checkpoint, data_loader_valid, i )
            train_loss.append(TrainLoss)
            train_accuracy.append(TrainAccuracy)
            valid_loss.append(ValidLoss)
            valid_accuracy.append(ValidAccuracy)
            result(train_loss, valid_loss, train_accuracy, valid_accuracy, epoch, i)

if __name__ == "__main__":
    main()