import os
import shutil

# set the paths to the two folders and the destination folder
folder1_path = r"C:\path\to\folder1"
folder2_path = r"C:\path\to\folder2"
dest_folder_path = r"C:\path\to\destination\folder"

# get a list of all PNG files in folder1
png_files = [f for f in os.listdir(folder1_path) if f.endswith(".png")]

# loop through the PNG files and look for matching TXT files in folder2
for png_file in png_files:
    png_file_name = os.path.splitext(png_file)[0]
    for txt_file in os.listdir(folder2_path):
        if txt_file.startswith(png_file_name) and txt_file.endswith(".txt"):
            # matching TXT file found, copy it to the destination folder
            shutil.copy(os.path.join(folder2_path, txt_file), dest_folder_path)
            break