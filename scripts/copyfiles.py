import os
import shutil

# set the paths to the two folders and the destination folder
folder1_path = r"D:\PPG2ABP\DenoisingNetwork\DATA\initial_ppg_abp\cleaning_512samples\fig_clean"
folder2_path = r"D:\PPG2ABP\DenoisingNetwork\DATA\initial_ppg_abp\512samples\ppg_p1\txt"
dest_folder_path = r"D:\PPG2ABP\data_for_training_split_shuffle\ppg_clean"


png_files = [f for f in os.listdir(folder1_path) if f.endswith(".png")]


for png_file in png_files:
    png_file_name = os.path.splitext(png_file)[0]
    for txt_file in os.listdir(folder2_path):
        if txt_file.startswith(png_file_name) and txt_file.endswith(".txt"):
            shutil.copy(os.path.join(folder2_path, txt_file), dest_folder_path)
            break