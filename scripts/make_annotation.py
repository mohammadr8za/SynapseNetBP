import os.path
from os import listdir
from os.path import join
import pandas as pd
from dataset import train_test_valid
import numpy as np


data_path = r"D:\data\PPGBP\mimic\pick_clean"
#TODO: chnage this dir to the correct dir of test txt files
inference_data_path = r"D:\PPG2ABP\data_for_test"
def MakeMainAnnotation(path,mode):

    if mode == "main":
        main_path = listdir(path)
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
    elif mode == 'inference':
        assert os.path.exists(join(inference_data_path, 'PPG')) and os.path.exists(join(inference_data_path, 'ABP')),\
            "You must put ppg and abp files in two folder with name PPG and ABP and give main dir of two folders to " \
            "Make annotation function"
        ppg_files = listdir(join(path,"PPG"))
        abp_files = listdir(join(path,"ABP"))


        signals_info = {"dir": [], "signal_fold": [], "signal_name": [], "label_fold": [], "lalbel_name": []}
        for i in range(min(len(ppg_files), len(abp_files))):

                signals_info["dir"].append(path)
                signals_info["signal_fold"].append('PPG')
                signals_info["label_fold"].append("ABP")
                signals_info["signal_name"].append(ppg_files[i])
                signals_info["lalbel_name"].append(ppg_files[i])
        signals_info = pd.DataFrame(signals_info)
        pd.DataFrame.to_csv(signals_info, os.path.join(inference_data_path, "inference.csv"))

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


def Make_Train_Valid_Annotation_directly_refinment(path):



        train_ppg_noisy_files = listdir(join(path, "TRAIN", "PPG_NOISY"))
        # train_ppg_clean_files = listdir("/home/mohammad/Documents/Project/DATA_BP_MAIN/train_data_v1/target/txt")
        valid_ppg_noisy_files = listdir(join(path, "VALID", "PPG_NOISY"))
        # valid_ppg_clean_files = listdir("/home/mohammad/Documents/Project/DATA_BP_MAIN/train_data_v1/target/txt")

        train_signals_info = {"dir": [], "signal_fold": [], "signal_name": [], "label_fold":[],"lalbel_name":[]}
        valid_signals_info = {"dir": [], "signal_fold": [], "signal_name": [], "label_fold":[],"lalbel_name":[]}

        for i in range(len(train_ppg_noisy_files)):

                train_signals_info["dir"].append(path)
                train_signals_info["signal_fold"].append(join( "TRAIN", "PPG_NOISY"))
                train_signals_info["label_fold"].append(r"D:\PPG2ABP\DenoisingNetwork\DATA\initial_ppg_abp\512samples\ppg_p1\txt")
                train_signals_info["signal_name"].append(train_ppg_noisy_files[i])
                train_signals_info["lalbel_name"].append(train_ppg_noisy_files[i])


        train_signals_info = pd.DataFrame(train_signals_info)
        pd.DataFrame.to_csv(train_signals_info, join(path, "TRAIN","Data_Train_Annotation.csv"))

        for i in range(len(valid_ppg_noisy_files)):

                valid_signals_info["dir"].append(path)
                valid_signals_info["signal_fold"].append(join( "VALID", "PPG_NOISY"))
                valid_signals_info["label_fold"].append(r"D:\PPG2ABP\DenoisingNetwork\DATA\initial_ppg_abp\512samples\ppg_p1\txt")
                valid_signals_info["signal_name"].append(valid_ppg_noisy_files[i])
                valid_signals_info["lalbel_name"].append(valid_ppg_noisy_files[i])
        valid_signals_info = pd.DataFrame(valid_signals_info)
        pd.DataFrame.to_csv(valid_signals_info, join(path, "VALID", "Data_valid_Annotation.csv"))


train_path = r"D:\PPG2ABP\data_for_train\train_data_200sample_loss"
if __name__ == "__main__":
    Make_Train_Valid_Annotation_directly_refinment(train_path)
    # MakeMainAnnotation(data_path,mode='main')
    # MakeMainAnnotationNoisy(data_path, mode='main')
    # MakeSubAnnotation(mode="main")

    # MakeMainAnnotationClean(data_path,mode='main')
    # MakeSubAnnotationClean(mode="main")