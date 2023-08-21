import os
import shutil

# set the paths to the two folders and the destination folder
folder1_path = r"D:\PPG2ABP\data_for_training_split_shuffle\very_clean_data_for_train\ppg_denoised\final_denoised_ppg\final_ppg"
folder2_path = r"D:\PPG2ABP\DenoisingNetwork\DATA\initial_ppg_abp\512samples\abp_p1\txt"
dest_folder_path = r"D:\PPG2ABP\data_for_training_split_shuffle\very_clean_data_for_train\final_abp"


png_files = [f for f in os.listdir(folder1_path) if f.endswith(".txt")]
abp_all = os.listdir(folder2_path)
#
#
# for png_file in png_files:
#     png_file_name = os.path.splitext(png_file)[0]
#     for txt_file in abp_all :
#         if txt_file.startswith(png_file_name) and txt_file.endswith(".txt"):
#             shutil.copy(os.path.join(folder2_path, txt_file), dest_folder_path)
#             break





# a = os.listdir(r"D:\PPG2ABP\DenoisingNetwork\DATA\initial_ppg_abp\512samples\ppg_p1\txt")
b = os.listdir(r"D:\PPG2ABP\data_for_training_split_shuffle\very_clean_data_for_train\ppg_denoised\final_denoised_ppg\final_ppg")
# aa = set(a)
# bb = set(b)
# cc = aa - bb
# ppg_not_denoised = cc
# for png_file in ppg_not_denoised:
#     png_file_name = os.path.splitext(png_file)[0]
#     for txt_file in os.listdir(folder2_path):
#         if txt_file.startswith(png_file_name) and txt_file.endswith(".txt"):
#             shutil.copy(os.path.join(folder2_path, txt_file), dest_folder_path)
#             break
# abp_all =  set(os.listdir(folder2_path))
for png_file in png_files:
      # shutil.copy(os.path.join(folder2_path, png_file.replace(".png",".txt")), dest_folder_path)
      shutil.copy(os.path.join(folder2_path, png_file), dest_folder_path)
#       try:
#        os.remove(os.path.join(folder2_path, png_file))
#       except FileNotFoundError:
#             print(f"file not found")
