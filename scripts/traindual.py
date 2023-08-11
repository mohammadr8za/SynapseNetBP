import os
import torch
from dataset import BPDatasetRam
from torch.utils.data import DataLoader
from sklearn.metrics import r2_score
from models.unet import UNetPPGtoABP
from models.unet2dinput import UNet2DInput
from models.transformernet import TransformerBlock
from models.vnet import VNet
import matplotlib.pyplot as plt
import argparse
from torch.utils.tensorboard import SummaryWriter
from torch.nn import MSELoss, L1Loss
from os import listdir
from torch.optim.lr_scheduler import  StepLR, CosineAnnealingLR, ConstantLR
import numpy as np

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

###################
alpha, beta = 1/2, 1/2
###################

#برای شروع 36 حالت رو بررسی کنیم و بعدا با توجه به نتایح مجدد آمورس میدبم
configs = {"models":[  UNetPPGtoABP()], "loss_func":[MSELoss()], "lr":[0.0005],
           "optimizer":["adam", "adagrad"],"batch_size":[32], "drop_out":[0.08],
           "lr_scheduler":["ConstantLR", "StepR"]}

if torch.cuda.is_available():

    device = "cuda:0"
    # device = "cpu"
else:
    device = "cpu"
    print(f"Using device {device}")

def load_data(train_data_annotation_path ,valid_data_annotation_path ):

    bp_data_train = BPDatasetRam(train_data_annotation_path, device, num_data=50)
    bp_data_train._load_data_to_RAM()

    bp_data_valid = BPDatasetRam(valid_data_annotation_path, device, num_data=50)
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
def train(model1, model2, loss_fn, optimiser, data_loader_train,epoch, scheduler_fn, train_mode, loss_fn2, optimiser2):


    if train_mode == "s":
        print(f"*************training eopch:{epoch}***************")
        model1.train()
        model2.train()
        loss_total = AverageMeter()
        loss_total_unet = AverageMeter()
        loss_total_trans = AverageMeter()
        y_true = []
        y_pred = []

        for batch_idx, (inputs, targets) in enumerate(data_loader_train):
            inputs, targets = inputs.to(device), targets.to(device)
            optimiser.zero_grad()
            outputs1 = model1(inputs.unsqueeze(1))
            inputs2 = torch.permute(torch.hstack((inputs.unsqueeze(1), outputs1.detach().unsqueeze(1))), (0, 1, 2))
            outputs2 = model2(inputs2)

            #TODO:
            lossT = loss_fn(outputs1, targets)
            lossU = loss_fn(outputs2, targets)

            loss = alpha * lossT + beta * lossU
            loss.backward()
            optimiser.step()
            # optimiser2.step()
            loss_total_unet.update(lossU)
            loss_total_trans.update(lossT)
            loss_total.update(loss)
            y_true += targets.cpu().numpy().tolist()
            y_pred += outputs2.cpu().detach().numpy().tolist()
            scheduler_fn.step()
            # print('batch \ totoal_data:',f'{(batch_idx/(len(data_loader_train))):.2}', f'loss: {loss_total.avg.item()}')
            # print(f"R2 : {r2_score(y_true, y_pred)} ")
            # if batch_idx == 100:
            #     break
        r2 = r2_score(y_true, y_pred)
        print(f"TRAIN R2 : {r2} ")


        return loss_total.avg.item(), r2, loss_total_trans.avg.item(), loss_total_unet.avg.item()
    elif train_mode == "ss":
        print(f"*************self supervised training eopch:{epoch}***************")
        model1.train()
        loss_total = AverageMeter()
        y_true = []
        y_pred = []

        for batch_idx, (inputs, targets) in enumerate(data_loader_train):
            # if inputs.shape[0] != batch_size or targets.shape[0] != batch_size:
            #     continue
            inputs, targets = inputs.to(device), targets.to(device)
            optimiser.zero_grad()
            random_index_for_mask = np.random.randint(0, 400)
            inputs = targets
            inputs[0][random_index_for_mask: random_index_for_mask + 100] = 0
            outputs = model1(inputs.unsqueeze(1))
            loss = loss_fn(outputs, targets)
            loss.backward()
            optimiser.step()
            loss_total.update(loss)
            y_true += targets.cpu().numpy().tolist()
            y_pred += outputs.cpu().detach().numpy().tolist()
            # scheduler_fn.step()
            # print('batch \ totoal_data:',f'{(batch_idx/(len(data_loader_train))):.2}', f'loss: {loss_total.avg.item()}')
            # print(f"R2 : {r2_score(y_true, y_pred)} ")
            # if batch_idx == 100:
            #     break
        r2 = r2_score(y_true, y_pred)
        print(f"TRAIN R2 : {r2} ")

        return loss_total.avg.item(), r2
def valid(model1, model2, loss_fn, data_loader_valid, optimiser,epoch, checkpoint, data_name, check_dir, train_mode, loss_fn2):
    if train_mode == "s":
        print(f"************* validation eopch:{epoch}***************")
        model1.eval()
        model2.eval()
        loss_total = AverageMeter()
        loss_transformer = AverageMeter()
        loss_unet = AverageMeter()
        y_true = []
        y_pred = []
        with torch.no_grad():
            for batch_idx, (inputs, targets) in enumerate(data_loader_valid):
                inputs, targets = inputs.to(device), targets.to(device)
                outputs1 = model1(inputs.unsqueeze(1))
                inputs2 = torch.permute(torch.hstack((inputs.unsqueeze(1), outputs1.detach().unsqueeze(1))), (0, 1, 2))
                outputs2 = model2(inputs2)

                lossT = loss_fn(outputs1, targets)
                lossU = loss_fn(outputs2, targets)

                loss = alpha * lossT + beta * lossU
                loss_transformer.update(lossT)
                loss_unet.update(lossU)
                loss_total.update(loss)

                y_true += targets.cpu().numpy().tolist()
                y_pred += outputs2.cpu().detach().numpy().tolist()
                # if batch_idx==2:
                #     break
        r2 = r2_score(y_true, y_pred)
        print(f"VALID R2 : {r2} ")

        checkpoint.save(model1=model1, model2=model2, acc=r2, filename=f"BPmodel",loss=loss_total.avg.item(),
                        loss_type=loss_fn, batch_size=Batch_size,
                        optimizer=optimiser, lr_value=LEARNING_RATE, epoch=epoch,data_name= data_name, check_dir=check_dir)

        return loss_total.avg.item(), r2, loss_transformer.avg.item(), loss_unet.avg.item()
    elif train_mode == "ss":
        print(f"************* self supervised validation eopch:{epoch}***************")
        model1.eval()
        loss_total = AverageMeter()
        y_true_valid = []
        y_pred_valid = []
        with torch.no_grad():
            for batch_idx, (inputs, targets) in enumerate(data_loader_valid):
                # if inputs.shape[0] != batch_size or targets.shape[0] != batch_size:
                #     continue
                inputs, targets = inputs.to(device), targets.to(device)
                random_index_for_mask = np.random.randint(0, 400)
                inputs = targets
                inputs[0][random_index_for_mask: random_index_for_mask + 100] = 0
                outputs = model1(inputs.unsqueeze(1))
                loss = loss_fn(outputs, targets)
                loss_total.update(loss)
                y_true_valid += targets.cpu().numpy().tolist()
                y_pred_valid += outputs.cpu().detach().numpy().tolist()
                # if batch_idx==2:
                #     break
        r2 = r2_score(y_true_valid, y_pred_valid)
        print(f"VALID R2 : {r2} ")
        checkpoint.save(model=model1, acc=r2, filename=f"BPmodel", loss=loss_total.avg.item(),
                        loss_type=loss_fn, batch_size=Batch_size,
                        optimizer=optimiser, lr_value=LEARNING_RATE, epoch=epoch, data_name=data_name,
                        check_dir=check_dir)

        return loss_total.avg.item(), r2


class Checkpoint(object):
    def __init__(self):

        self.best_acc = 0.
        self.folder = 'chekpoint'
        os.makedirs(self.folder, exist_ok=True)



    def save(self, model1, model2, acc, loss, lr_value, batch_size, loss_type, optimizer, filename, epoch, data_name, check_dir):
            # os.makedirs(os.path.join(self.folder, data_name))
        # if acc > self.best_acc:
            print('Saving checkpoint...')


            state = {
                'transformer': model1.state_dict(),
                'unet': model2.state_dict(),
                # 'acc': acc,
                # 'epoch': epoch,
                # 'loss' : loss,
                # 'loss_type': loss_type,
                # 'lr_value': lr_value,
                # 'batch_size': batch_size,
                # 'optimizer': optimizer,
                'transformer_model_architecht': model1,
                'unet_model_architecht': model2
         }
            path = os.path.join(check_dir , f'epoch{epoch}.pth')
            torch.save(state, path)
            self.best_acc = acc


def result(train_loss, valid_loss, train_accuracy, valid_accuracy,transtrainloss, transvalidloss,
           unettrainloss, unetvalidloss, epoch, data_name, check_dir):

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

    plt.figure(figsize=(15, 15))
    plt.plot(transtrainloss)
    plt.plot(transvalidloss)
    plt.xlabel('epoch')
    plt.ylabel('Loss')
    plt.legend(['TRAINING', 'VALIDATION'], loc='upper right')
    # print()
    # plt.savefig(fr"./chekpoint/BPModel_R2.png")
    plt.savefig(os.path.join(check_dir, "trans_train_valid_loss.png"))
    plt.close()

    plt.figure(figsize=(15, 15))
    plt.plot(unettrainloss)
    plt.plot(unetvalidloss)
    plt.xlabel('epoch')
    plt.ylabel('Loss')
    plt.legend(['TRAINING', 'VALIDATION'], loc='upper right')
    # print()
    # plt.savefig(fr"./chekpoint/BPModel_R2.png")
    plt.savefig(os.path.join(check_dir, "unet_train_valid_loss.png"))
    plt.close()


def parse_args():
    parser = argparse.ArgumentParser(description='Train a deep model')
    parser.add_argument('--data_dir', type=str, default='data', help='Path to training data directory')
    parser.add_argument('--start_epoch', type=int, default=0, help='Start epoch number')
    parser.add_argument('--end_epoch', type=int, default=20, help='End epoch number')
    parser.add_argument('--learning_rate', type=float, default=0.001, help='Learning rate for optimizer')
    parser.add_argument('--batch_size', type=int, default=32, help='Batch size for training')
    return parser.parse_args()
def main(TRAIN_MODE):
    counter = 0
    for i in data_folder_list:
        i="train_final"
        data_train_path = os.path.join(DATASETS_PATH, i, "Data_Train_Annotation.csv")
        data_valid_path = os.path.join(DATASETS_PATH, i, "Data_valid_Annotation.csv")
        bp_data_train, bp_data_valid = load_data(data_train_path, data_valid_path)
        os.makedirs(os.path.join("chekpoint", i), exist_ok=True)
        start, end = 0, 200
        for drop in configs["drop_out"]:
            for model_type in configs["models"]:
                model = model_type
                stat_dict = torch.load(PRETRAIN_MODEL)
                model.load_state_dict(stat_dict["net"])
                model.to(device)
                model2 = UNet2DInput()
                # stat_dict2 = torch.load(PRETRAIN_MODEL2)
                # model2.load_state_dict2(stat_dict["net"])
                model2.to(device)
                for loss_func in configs["loss_func"]:
                    loss_fn = loss_func
                    loss_fn2 = loss_func
                    for lr_rate in configs["lr"]:
                        optimiser = torch.optim.Adam([{"params":model.parameters(),"lr":lr_rate},
                                                      {"params":model2.parameters(),"lr":lr_rate}])
                        optimiser2 = torch.optim.Adam(model2.parameters(), lr=lr_rate)
                        for scheduler_type in configs["lr_scheduler"]:
                            if scheduler_type == "StepR":
                                scheduler_lr = StepLR(optimiser, step_size=25, gamma=0.5)
                            elif scheduler_type == "Cosinanlealing":
                                scheduler_lr = CosineAnnealingLR(optimiser, T_max=15, eta_min=0)
                            elif scheduler_type == "ConstantLR":
                                scheduler_lr = ConstantLR(optimiser, factor=0.5, total_iters=45)
                            for batch_size in configs["batch_size"]:

                                data_loader_train = DataLoader(bp_data_train, batch_size, shuffle=True)
                                data_loader_valid = DataLoader(bp_data_valid, batch_size, shuffle=True)
                                args = parse_args()
                                checkpoint = Checkpoint()
                                checkpoint_path = os.path.join("checkpoint",TRAIN_MODE,"dualnet", i, f"drop_{drop}", f"{str(model._get_name())}",
                                                               f"loss_{loss_fn._get_name() }",f"lr_{lr_rate}",f"batch_{batch_size}",
                                                               scheduler_type)
                                os.makedirs(checkpoint_path, exist_ok=True)
                                os.makedirs(os.path.join(checkpoint_path, "run"), exist_ok=True)

                                log_dir = os.path.join(checkpoint_path, "run")

                                writer = SummaryWriter(log_dir)
                                counter += 1
                                # if counter < 6:
                                #     continue
                                print(f"***********************COUNTER: {counter}***************************")
                                trans_train_loss = []
                                trans_valid_loss = []
                                trans_train_accuracy = []
                                trans_valid_accuracy = []

                                unet_train_loss = []
                                unet_valid_loss = []
                                unet_train_accuracy = []
                                unet_valid_accuracy = []

                                train_loss = []
                                valid_loss = []
                                train_accuracy = []
                                valid_accuracy = []
                                for epoch in range(start, end):
                                    TrainLoss, TrainAccuracy, TrainLossT, TrainLossU = train(model, model2, loss_fn, optimiser, data_loader_train,
                                                                     epoch,scheduler_fn= scheduler_lr, train_mode=TRAIN_MODE,
                                                                     optimiser2= optimiser2, loss_fn2=loss_fn2)
                                    ValidLoss, ValidAccuracy, ValidLossT, ValidLossU = valid(model,model2, loss_fn, data_loader_valid, optimiser,
                                                                     epoch, checkpoint, i,
                                                                     checkpoint_path, train_mode= TRAIN_MODE,
                                                                     loss_fn2=loss_fn2)
                                    trans_train_loss.append(TrainLossT)
                                    trans_valid_loss.append(ValidLossT)
                                    unet_train_loss.append(TrainLossU)
                                    unet_valid_loss.append(ValidLossT)
                                    train_loss.append(TrainLoss)

                                    train_accuracy.append(TrainAccuracy)
                                    valid_loss.append(ValidLoss)
                                    valid_accuracy.append(ValidAccuracy)
                                    result(train_loss, valid_loss, train_accuracy, valid_accuracy,trans_train_loss,
                                           trans_valid_loss, unet_train_loss, unet_valid_loss, epoch, i, checkpoint_path)
                                    writer.add_scalars(main_tag="accuracy",
                                                       tag_scalar_dict={"train_accuracy": TrainAccuracy,
                                                                        "valid_accuracy": ValidAccuracy},
                                                       global_step=epoch)
                                    writer.add_scalars(main_tag="loss",
                                                       tag_scalar_dict={"train_loss": TrainLoss,
                                                                        "valid_loss": ValidLoss},
                                                       global_step=epoch)
                                writer.close()
PRETRAIN_MODEL = r"G:\PPG2ABP_TRAIN\train_results\Denoise_net_final_train\self-supervised\final-stage\unet\epoch20.pth"
PRETRAIN_MODEL2 = r"TODO"
TRAIN_MODE = "s"
if __name__ == "__main__":
    main(TRAIN_MODE)