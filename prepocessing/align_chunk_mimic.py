import os
import numpy as np

from scipy.io import loadmat
import matplotlib.pyplot as plt
from scipy.signal import correlate

#     G E N E R A  L     S E T U P
ppg_save_dir = r'D:\data\PPGBP\mimic\ppg'
abp_save_dir = r'D:\data\PPGBP\mimic\abp'
fs = 125
T = 1 / fs
chunk_in_sec = 5
chunk_in_sample = int(5 / T) # اینجا به جای 5 ، chunk in sec گذاشتم

annots_1 = loadmat('Part_1.mat')
part_1 = annots_1['Part_1'].squeeze()

# annots_3 = loadmat('Part_3.mat')
# part_3 = annots_3['Part_3'].squeeze()
#
# annots_4 = loadmat('Part_4.mat')
# part_4 = annots_4['Part_4'].squeeze()

for sig in range(111, len(part_1)):
    ppg_sig = part_1[sig][0]
    abp_sig = part_1[sig][1]
    # Alignment
    corr = correlate(ppg_sig, abp_sig)
    delay = corr.argmax() - ppg_sig.shape[0]
    abp_sig = np.roll(abp_sig, delay)
    # Segment into n seconds chunks
    num_chunk = int(ppg_sig.shape[0] / chunk_in_sample)
    for num in range(num_chunk):
        chunk_name = str(1) + '-' + str(sig) + '-' + str(
            num)  # [Part-Sig-Chunk]  -> 1-200-18 means chunk 18 from 200-th signal in Part 1 <-
        ppg_chunk = ppg_sig[num * chunk_in_sample: (num + 1) * chunk_in_sample]
        plt.figure()
        plt.plot(np.arange(0, ppg_chunk.shape[0]), ppg_chunk)
        plt.savefig(os.path.join(ppg_save_dir, chunk_name) + '.png')
        plt.close()
        np.savetxt(os.path.join(ppg_save_dir, chunk_name) + '.txt', ppg_chunk, fmt='%f')

        abp_chunk = abp_sig[num * chunk_in_sample: (num + 1) * chunk_in_sample]
        # plt.figure()
        # plt.plot(np.arange(0, abp_chunk.shape[0]), abp_chunk)
        # plt.savefig(os.path.join(abp_save_dir, chunk_name) + '.png')
        # plt.close()
        # np.savetxt(os.path.join(abp_save_dir, chunk_name) + '.txt', abp_chunk, fmt='%f')

1+1