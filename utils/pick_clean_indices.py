import os
import random
from os import listdir
from os import walk
from os.path import join

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

path_to_data = r'D:\data\PPGBP\mimic\pick_clean\abp_clean_indices'
save_path = r'D:\data\PPGBP\mimic\pick_clean'


def data_to_excel(path, save_path):

    dict = {"data_name": []}

    for i, data in enumerate(listdir(path)):
        # Only No-Data
        dict["data_name"].append(data)

    data_frame = pd.DataFrame(dict)

    data_frame.to_csv(join(save_path, 'clean_ppg_list.xlsx'), index=False)

    # data_frame.to_excel(join(save_path, 'clean_abp.xlsx'), sheet_name='sheet1', index=False)


if __name__ == "__main__":
    # path to clean ABPs
    path_to_data = r'D:\data\PPGBP\mimic\pick_clean\abp_clean_indices'
    # where to save list of the clean ABPs
    save_path = r'D:\data\PPGBP\mimic\pick_clean'
    data_to_excel(path_to_data, save_path)   # TODO: it saves name of the clean ABPs/PPGs

    data_list = pd.read_csv(join(save_path, 'clean_abp.xlsx'))['data_name']
    for name in range(len(data_list)):
        # address of PPGs
        ppg_chunk_address = join(save_path, 'ppg', data_list[name][:-3]+'txt')
        ppg_chunk = np.loadtxt(ppg_chunk_address)
        # plt.figure()
        # plt.plot(np.arange(0, ppg_chunk.shape[0]), ppg_chunk)
        # plt.savefig(os.path.join(save_path, 'ppg_clean_indices', data_list[name]))
        # plt.close()
        # TODO: separate and save equivalent PPGs/ABPs (to the clean ABPs/PPGs) in another directory
        np.savetxt(join(save_path, 'ppg_clean_indices', data_list[name][:-3]+'txt'), ppg_chunk)

# TODO: Purpose of this code is to pick PPGs equivalent to the separated clean ABPs
# TODO: to create training pairs (ABP, PPG) [PPGs needs additional cleaning]
# TODO: then, remove invalid PPGs and pick equivalent ABPs again
#  (less noisy signals should remain as the recreation network exist)
# TODO: in another scenario, we use all PPGs as the Recreation Network exist
#  [performance may degrade compare to the above mentioned scenario,
#  however we must reach a performance similar or better than previously published papers]
