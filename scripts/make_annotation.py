import os.path
from os import listdir
from os.path import join
import pandas as pd
from dataset import train_test_valid
import numpy as np


data_path = r"D:\data\PPGBP\mimic\pick_clean"

def MakeMainAnnotation(path,mode):
    main_path = listdir(path)
    if mode == "main":
        abp_files = listdir(join(path, main_path[1]))
        ppg_files = listdir(join(path, main_path[3]))

        signals_info = {"dir": [], "signal_fold": [], "signal_name": [], "label_fold":[],"lalbel_name":[]}
        for i in range(min(len(ppg_files), len(abp_files))):
            if os.path.exists(join(path,main_path[1], ppg_files[i])) and os.path.exists(join(path, main_path[3], abp_files[i])):
                signals_info["dir"].append(path)
                signals_info["signal_fold"].append(main_path[1])
                signals_info["label_fold"].append(main_path[3])
                signals_info["signal_name"].append(ppg_files[i])
                signals_info["lalbel_name"].append(ppg_files[i])


        signals_info = pd.DataFrame(signals_info)
        pd.DataFrame.to_csv(signals_info, r"D:\data\PPGBP\mimic\Main_Data_Train_Annotation.csv")
   #TODO: elif mode == 'inference':

def MakeSubAnnotation(mode):

    if mode == 'main':
        Data_Annotation = pd.read_csv(r"D:\data\PPGBP\mimic\Main_Data_Train_Annotation.csv")

        Data_Annotation.drop('Unnamed: 0', axis=1, inplace= True)
        number_of_data =len(Data_Annotation)
        x_index = np.arange(0, number_of_data)
        train_index, test_index, valid_index = train_test_valid(x_index).get_data_set()

        train_annotation = Data_Annotation.loc[train_index.tolist(), :]
        test_annotation = Data_Annotation.loc[test_index.tolist(), :]
        valid_annotation = Data_Annotation.loc[valid_index.tolist(), :]

    #TODO: elif mode == 'inference':

    if mode == "main":
        pd.DataFrame.to_csv(train_annotation,
                            r"D:\data\PPGBP\mimic\Data_Train_Annotation.csv")
        pd.DataFrame.to_csv(test_annotation,
                            r"D:\data\PPGBP\mimic\Data_test_Annotation.csv")
        pd.DataFrame.to_csv(valid_annotation,
                            r"D:\data\PPGBP\mimic\Data_valid_Annotation.csv")
    #TODO: elif mode == "inference":

def MakeMainAnnotationClean(path,mode):
    main_path = listdir(path)

    if mode == "main":
        abp_files = listdir(join(path, main_path[1]))
        ppg_files = listdir(join(path, main_path[3]))

        signals_info = {"dir": [], "signal_fold": [], "signal_name": [], "label_fold":[],"lalbel_name":[]}
        for i in range(min(len(ppg_files), len(abp_files))):
            if os.path.exists(join(path,main_path[0], ppg_files[i].replace("png", "txt") )) and os.path.exists(join(path, main_path[2], abp_files[i].replace("png", "txt") )):
                signals_info["dir"].append(path)
                signals_info["signal_fold"].append(main_path[0])
                signals_info["label_fold"].append(main_path[2])
                signals_info["signal_name"].append(ppg_files[i].replace("png", "txt"))
                signals_info["lalbel_name"].append(ppg_files[i].replace("png", "txt"))


        signals_info = pd.DataFrame(signals_info)
        pd.DataFrame.to_csv(signals_info, r"C:\data\PPGBP\mimic\pick_clean/Main_Data_Train_Annotation.csv")
   #TODO: elif mode == 'inference':

def MakeSubAnnotationClean(mode):

    if mode == 'main':
        Data_Annotation = pd.read_csv(r"C:\data\PPGBP\mimic\pick_clean\Main_Data_Train_Annotation.csv")

        Data_Annotation.drop('Unnamed: 0', axis=1, inplace= True)
        number_of_data =len(Data_Annotation)
        x_index = np.arange(0, number_of_data)
        train_index, test_index, valid_index = train_test_valid(x_index).get_data_set()

        train_annotation = Data_Annotation.loc[train_index.tolist(), :]
        test_annotation = Data_Annotation.loc[test_index.tolist(), :]
        valid_annotation = Data_Annotation.loc[valid_index.tolist(), :]

    #TODO: elif mode == 'inference':

    if mode == "main":
        pd.DataFrame.to_csv(train_annotation,
                            r"C:\data\PPGBP\mimic\pick_clean\Data_Train_Annotation.csv")
        pd.DataFrame.to_csv(test_annotation,
                            r"C:\data\PPGBP\mimic\pick_clean\Data_test_Annotation.csv")
        pd.DataFrame.to_csv(valid_annotation,
                            r"C:\data\PPGBP\mimic\pick_clean\Data_valid_Annotation.csv")
    #TODO: elif mode == "inference":

def MakeMainAnnotationNoisy(path,mode):
    main_path = listdir(path)
    if mode == "main":
        abp_files = listdir(join(path, main_path[5]))
        ppg_files = listdir(join(path, main_path[3], "sectorial_loss"))

        signals_info = {"dir": [], "signal_fold": [], "signal_name": [], "label_fold":[],"lalbel_name":[]}
        for i in range(min(len(ppg_files), len(abp_files))):
            if os.path.exists(join(join(path, main_path[3], "sectorial_loss"))) and os.path.exists(join(path, main_path[5], abp_files[i])):
                signals_info["dir"].append(path)
                signals_info["signal_fold"].append(join(main_path[3], 'sectorial_loss'))
                signals_info["label_fold"].append(main_path[5])
                signals_info["signal_name"].append(ppg_files[i])
                signals_info["lalbel_name"].append(abp_files[i])


        signals_info = pd.DataFrame(signals_info)
        pd.DataFrame.to_csv(signals_info, r"D:\data\PPGBP\mimic\Main_Data_Train_Annotation.csv")
   #TODO: elif mode == 'inference':

if __name__ == "__main__":
    # MakeMainAnnotation(data_path,mode='main')
    MakeMainAnnotationNoisy(data_path, mode='main')
    MakeSubAnnotation(mode="main")

    # MakeMainAnnotationClean(data_path,mode='main')
    # MakeSubAnnotationClean(mode="main")