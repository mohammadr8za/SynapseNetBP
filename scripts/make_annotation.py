import os.path
from os import listdir
from os.path import join
import pandas as pd
from dataset import train_test_valid
import numpy as np




#TODO: chnage this dir to the correct dir of test txt files
inference_data_path = r"D:\PPG2ABP\data_for_test"


def  MakeMainAnnotation(data_path, label_path, mode, noise_type):

    if mode == "main":
        path = r"C:\data\data_for_training_split_shuffle"
        abp_files = listdir(data_path)
        ppg_files = listdir(label_path)
        signals_info = {"dir": [], "signal_fold": [], "signal_name": [], "label_fold":[],"lalbel_name":[]}
        for i in range(min(len(ppg_files), len(abp_files))):
            if os.path.exists(join(data_path, ppg_files[i])) and os.path.exists(join(label_path, abp_files[i])):
                signals_info["dir"].append(path)
                signals_info["signal_fold"].append(join("ppg_noisy", noise_type, "noisy_data"))
                signals_info["label_fold"].append("ppg_clean")
                signals_info["signal_name"].append(ppg_files[i])
                signals_info["lalbel_name"].append(ppg_files[i])


        signals_info = pd.DataFrame(signals_info)
        pd.DataFrame.to_csv(signals_info,join(ROOT_DATA_PATH, "ppg_noisy", noise_type, "Main_Data_Train_Annotation.csv"))
    elif mode == 'inference':
        assert os.path.exists(join(inference_data_path, 'PPG')) and os.path.exists(join(inference_data_path, 'ABP')),\
            "You must put ppg and abp files in two folder with name PPG and ABP and give main dir of two folders to " \
            "Make annotation function"
        path = r"D:\PPG2ABP\data_for_test"
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

def MakeSubAnnotation(mode, noise_type):

    if mode == 'main':
        Data_Annotation = pd.read_csv(join(ROOT_DATA_PATH, "ppg_denoised",'final_denoised_ppg', "Main_Data_Train_Annotation.csv"))

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
                            join(ROOT_DATA_PATH,  "ppg_denoised",'final_denoised_ppg', "Data_Train_Annotation.csv"))
        pd.DataFrame.to_csv(test_annotation,
                           join(ROOT_DATA_PATH,  "ppg_denoised",'final_denoised_ppg',"Data_test_Annotation.csv"))
        pd.DataFrame.to_csv(valid_annotation,
                           join(ROOT_DATA_PATH,  "ppg_denoised",'final_denoised_ppg',"Data_valid_Annotation.csv"))
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

def  MakeMainAnnotationABP(data_path, label_path, mode, noise_type):

    if mode == "main":
        path = r"C:\data\data_for_training_split_shuffle"
        abp_files = listdir(data_path)
        ppg_files = listdir(label_path)
        signals_info = {"dir": [], "signal_fold": [], "signal_name": [], "label_fold":[],"lalbel_name":[]}
        for i in range(min(len(ppg_files), len(abp_files))):
            if os.path.exists(join(data_path, ppg_files[i])) and os.path.exists(join(label_path, abp_files[i])):
                signals_info["dir"].append(path)
                signals_info["signal_fold"].append(join("ppg_denoised", noise_type, "final_ppg"))
                signals_info["label_fold"].append("final_abp")
                signals_info["signal_name"].append(ppg_files[i])
                signals_info["lalbel_name"].append(ppg_files[i])


        signals_info = pd.DataFrame(signals_info)
        pd.DataFrame.to_csv(signals_info,join(ROOT_DATA_PATH, "ppg_denoised","final_denoised_ppg", "Main_Data_Train_Annotation.csv"))
    # elif mode == 'inference':
    #     assert os.path.exists(join(inference_data_path, 'PPG')) and os.path.exists(join(inference_data_path, 'ABP')),\
    #         "You must put ppg and abp files in two folder with name PPG and ABP and give main dir of two folders to " \
    #         "Make annotation function"
    #     ppg_files = listdir(join(path,"PPG"))
    #     abp_files = listdir(join(path,"ABP"))


        # signals_info = {"dir": [], "signal_fold": [], "signal_name": [], "label_fold": [], "lalbel_name": []}
        # for i in range(min(len(ppg_files), len(abp_files))):
        #
        #         signals_info["dir"].append(path)
        #         signals_info["signal_fold"].append('PPG')
        #         signals_info["label_fold"].append("ABP")
        #         signals_info["signal_name"].append(ppg_files[i])
        #         signals_info["lalbel_name"].append(ppg_files[i])
        # signals_info = pd.DataFrame(signals_info)
        # pd.DataFrame.to_csv(signals_info, os.path.join(inference_data_path, "inference.csv"))


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

ROOT_DATA_PATH = r"C:\data\data_for_training_split_shuffle"
DATASETS_PATH = r"C:\data\data_for_training_split_shuffle\ppg_noisy\train_final\noisy_data"
train_path = r"C:\data\data_for_train\train_data_100sample_loss"
LABEL_PATH = r"C:\data\data_for_training_split_shuffle\ppg_clean"
ABP_PATH = r"C:\data\data_for_training_split_shuffle\final_abp"


inference_data_path = r"D:\PPG2ABP\data_for_test"
if __name__ == "__main__":

    # for dataset in listdir(DATASETS_PATH):
        NOISE_TYPE = "final_denoised_ppg"
        # DATA_PATH = os.path.join(ROOT_DATA_PATH,"DNET", "ppg_noisy", NOISE_TYPE,"final_ppg")
        DATA_PATH = os.path.join(ROOT_DATA_PATH, "ppg_denoised", NOISE_TYPE,"final_ppg")
        # MakeMainAnnotationABP(DATA_PATH, LABEL_PATH,'main', NOISE_TYPE)
        # MakeMainAnnotationABP(DATA_PATH, ABP_PATH,'main', NOISE_TYPE)
        MakeMainAnnotation(DATASETS_PATH, LABEL_PATH, "inference", NOISE_TYPE)
        # MakeSubAnnotation("main", NOISE_TYPE)
