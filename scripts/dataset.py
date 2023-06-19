import os
import torch
from torch.utils.data import Dataset
import pandas as pd
from sklearn.model_selection import train_test_split
from scripts import txt_load
import numpy as np
from tqdm import tqdm


if torch.cuda.is_available():
    device = "cuda:0"
else:
    device = "cpu"
    print(f"Using device {device}")

data_annotation_path = r"D:\data\PPGBP\mimic\Data_Train_Annotation.csv"

class BPDataset(Dataset):

    def __init__(self,
                 data_annotations_file,
                 device):
        self.data_annotations = pd.read_csv(data_annotations_file)
        self.data_dir = self.data_annotations["dir"]
        self.device = device

    def __len__(self):
        return len(self.data_annotations)

    def __getitem__(self, index):
        abp_signal_path = self._get_signal_path(index)
        label_path = self._get_label_path(index)
        signal = torch.tensor(txt_load.TxtLoad(abp_signal_path))
        label = torch.tensor( txt_load.TxtLoad(label_path))
        signal = signal.to(self.device)
        label = label.to(self.device)
        # print(signal.shape)
        # print(label.shape)
        return signal.float()/10, label.float()/200



    def _get_signal_path(self, index):
        path = os.path.join(self.data_dir[index], self.data_annotations.iloc[index, 2],
                            self.data_annotations.iloc[index, 3])
        return path
    def _get_label_path(self, index):
        path = os.path.join(self.data_dir[index], self.data_annotations.iloc[index, 4],
                            self.data_annotations.iloc[index, 5])
        return path
class BPDatasetRam(Dataset):

    def __init__(self,
                 data_annotations_file,
                 device):
        self.data_annotations = pd.read_csv(data_annotations_file)
        self.data_dir = self.data_annotations["dir"]
        self.device = device
        self.total_data = np.empty((512))
        self.total_label = np.empty((512))


    def __len__(self):
        return len(self.data_annotations)

    def __getitem__(self, index):
        signal = self.total_data[index+1]
        label = self.total_label[index+1]
        signal = torch.tensor(signal)
        label = torch.tensor(label)
        #TODO: یونت و وی نت چون توان 2 بالا پایین میکنند اینجارو باید مضرب2 بذاریم
        return  (signal.float()/10), (label.float()/10)
        # return  (signal.float()/200), (label.float()/10)

    def _load_data_to_RAM(self):
        for index in tqdm(range(len(self.data_annotations)), desc= "Loading all data to RAM", ncols=80):

            abp_signal_path = self._get_signal_path(index)
            label_path = self._get_label_path(index)
            signal = torch.tensor(txt_load.TxtLoad(abp_signal_path))
            label = torch.tensor(txt_load.TxtLoad(label_path))
            # signal = signal.to(self.device)
            # label = label.to(self.device)
            if signal.shape[0] == 512 and label.shape[0]== 512:
                self.total_data = np.vstack((self.total_data, signal))
                self.total_label = np.vstack((self.total_label, label))
        print("All data succesfully loaded in RAM, let's TRAIN")


    def _get_signal_path(self, index):
        path = os.path.join(self.data_dir[index], self.data_annotations.iloc[index, 2],
                            self.data_annotations.iloc[index, 3])
        return path
    def _get_label_path(self, index):
        path = os.path.join(self.data_dir[index], self.data_annotations.iloc[index, 4],
                            self.data_annotations.iloc[index, 5])
        return path


class train_test_valid:
    def __init__(self, data):
        self.bp = data

    def get_data_set(self):
        rsd_train, rsd_test1 = train_test_split(self.bp, test_size=.2, shuffle=True)
        rsd_test, rsd_valid = train_test_split(rsd_test1, test_size=.8, shuffle=True)

        return rsd_train, rsd_test, rsd_valid




if __name__ == "__main__":
    bp_dataset = BPDataset(data_annotation_path, device)
    print(bp_dataset.__getitem__(1))