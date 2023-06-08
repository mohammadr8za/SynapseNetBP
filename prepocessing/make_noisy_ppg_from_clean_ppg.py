import os
from matplotlib import pyplot as plt
import numpy as np


clean_ppg_dir = r'D:\data\PPGBP\mimic\pick_clean\ppg_clean_indices'
clean_ppg_list = os.listdir(clean_ppg_dir)

ppg_txt_dir = r'D:\data\PPGBP\mimic\pick_clean\ppg'
noisy_ppg_save_dir = r'D:\data\PPGBP\mimic\pick_clean\noisy_ppg'

os.environ['KMP_DUPLICATE_LIB_OK']='True'


NUM_LOSS_SAMPLE = 50

for clean_ppg in clean_ppg_list:

    ppg_chunk = np.loadtxt(os.path.join(ppg_txt_dir, clean_ppg[:-3]+'txt'))
    plt.plot(np.arange(0, ppg_chunk.shape[0]), ppg_chunk, color='b')
    # plt.show()

    # sample loss
    random_samples_loss = np.sort(np.random.randint(low=0, high=ppg_chunk.shape[0], size=NUM_LOSS_SAMPLE))

    noisy_ppg = ppg_chunk.copy()
    noisy_ppg[random_samples_loss] = np.random.rand(NUM_LOSS_SAMPLE)
    # plt.figure()
    # plt.plot(np.arange(0, noisy_ppg.shape[0]), noisy_ppg, color='r')
    # plt.savefig(os.path.join(noisy_ppg_save_dir, 'random_loss', clean_ppg))
    # # plt.show()
    # plt.close()
    np.savetxt(os.path.join(noisy_ppg_save_dir, 'random_loss', clean_ppg[:-3]) + 'txt', noisy_ppg, fmt='%f')

    # sectorial loss
    random_samples_loss = np.random.randint(low=0, high=ppg_chunk.shape[0]-NUM_LOSS_SAMPLE, size=1)
    noisy_ppg_2 = ppg_chunk.copy()
    noisy_ppg_2[random_samples_loss.item():(random_samples_loss+NUM_LOSS_SAMPLE).item()] = \
        (np.max(ppg_chunk)-np.min(ppg_chunk))*np.random.rand(NUM_LOSS_SAMPLE)

    # noisy_ppg_2[random_samples_loss.item():(random_samples_loss + NUM_LOSS_SAMPLE).item()] = \
    #     np.mean(ppg_chunk)

    # plt.figure()
    # plt.plot(np.arange(0, noisy_ppg_2.shape[0]), noisy_ppg_2, color='r')
    # plt.savefig(os.path.join(noisy_ppg_save_dir, 'sectorial_loss', clean_ppg))
    # # plt.show()
    # plt.close()
    np.savetxt(os.path.join(noisy_ppg_save_dir, 'sectorial_loss', clean_ppg[:-3]) + 'txt', noisy_ppg_2, fmt='%f')
