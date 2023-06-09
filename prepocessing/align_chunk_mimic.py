import os
import numpy as np

from scipy.io import loadmat
import matplotlib.pyplot as plt
from scipy.signal import correlate

#     G E N E R A  L     S E T U P
ppg_save_dir = r'D:\PPG2ABP\DenoisingNetwork\DATA\initial_ppg_abp\512samples\ppg'
abp_save_dir = r'D:\PPG2ABP\DenoisingNetwork\DATA\initial_ppg_abp\512samples\abp'
fs = 125
T = 1 / fs
chunk_in_sec = 4.096
chunk_in_sample = int(chunk_in_sec / T)
part_number = 3

if part_number == 1:
    annots_1 = loadmat('D:\PythonProjects\PPGBP\Part_1.mat')
    part_1 = annots_1['Part_1'].squeeze()
elif part_number == 2:
    annots_3 = loadmat('D:\PythonProjects\PPGBP\Part_2.mat')
    part_3 = annots_3['Part_2'].squeeze()
elif part_number == 3:
    annots_3 = loadmat('D:\PythonProjects\PPGBP\Part_3.mat')
    part_3 = annots_3['Part_3'].squeeze()
elif part_number == 4:
    annots_4 = loadmat('D:\PythonProjects\PPGBP\Part_4.mat')
    part_4 = annots_4['Part_4'].squeeze()
else:
    print('No such part exists! Choose among 1, 2, 3, and 4.')


for sig in range(len(part_1)):
    ppg_sig = part_1[sig][0]
    abp_sig = part_1[sig][1]
    # Alignment
    corr = correlate(ppg_sig, abp_sig)
    delay = corr.argmax() - ppg_sig.shape[0]
    abp_sig = np.roll(abp_sig, delay)
    # Segment into n seconds chunks
    num_chunk = int(ppg_sig.shape[0] / chunk_in_sample)
    for num in range(num_chunk):
        chunk_name = str(3) + '-' + str(sig) + '-' + str(
            num)  # [Part-Sig-Chunk]  -> 1-200-18 means chunk 18 from 200-th signal in Part 1 <-
        ppg_chunk = ppg_sig[num * chunk_in_sample: (num + 1) * chunk_in_sample]
        plt.figure()
        plt.plot(np.arange(0, ppg_chunk.shape[0]), ppg_chunk)
        plt.savefig(os.path.join(ppg_save_dir, 'fig', chunk_name) + '.png')
        plt.close()
        np.savetxt(os.path.join(ppg_save_dir, 'txt', chunk_name) + '.txt', ppg_chunk, fmt='%f')

        abp_chunk = abp_sig[num * chunk_in_sample: (num + 1) * chunk_in_sample]
        # plt.figure()
        # plt.plot(np.arange(0, abp_chunk.shape[0]), abp_chunk)
        # plt.savefig(os.path.join(abp_save_dir, chunk_name) + '.png')
        # plt.close()
        # np.savetxt(os.path.join(abp_save_dir, chunk_name) + '.txt', abp_chunk, fmt='%f')
