import torch
import numpy as np
from os import listdir
from os.path import join



def Snr(predicted_signal, target_signal):
    noise = target_signal - predicted_signal
    signal_power = np.sum(np.power(target_signal, 2))
    noise_power =  np.sum(np.power(noise, 2))
    # snr = (10 * np.log10( noise_power)) / (10 *np.log10(signal_power))
    snr = 10 * np.log10(signal_power / noise_power)
    # Return the negative SNR as the loss (to be minimized)
    return snr


def dataset_snr(enhanced_path, noisy_path, clean_path, mode='BOTH'):

    enhanced_list = listdir(enhanced_path)
    noisy_list = listdir(noisy_path)
    clean_list = listdir(clean_path)

    if mode == "init":
        init_snr_list = []
        for index in range(len(noisy_list)):
            if noisy_list[index] == clean_list[index]:
                init_snr_list.append(Snr(np.loadtxt(join(noisy_path, noisy_list[index])),
                                         np.loadtxt(join(enhanced_path, enhanced_list[index])))
                                     )
            else:
                print("mismatch, check your data directory!")

        return np.mean(np.array(init_snr_list))

    if mode == "enhanced":
        enhanced_snr_list = []
        for index in range(len(enhanced_list)):
            if enhanced_list[index] == clean_list[index]:
                enhanced_snr_list.append(Snr(np.loadtxt(join(enhanced_path, enhanced_list[index])),
                                             np.loadtxt(join(clean_path, clean_list[index])))
                                         )
            else:
                print("mismatch, check your data directory!")

        return np.mean(np.array(enhanced_snr_list))

    if mode == 'both':
        init_snr_list, enhanced_snr_list = [], []
        for index in range(len(enhanced_list)):
            if noisy_list[index] == clean_list[index] and enhanced_list[index] == clean_list[index]:
                init_snr_list.append(Snr(np.loadtxt(join(noisy_path, noisy_list[index])),
                                         np.loadtxt(join(enhanced_path, enhanced_list[index])))
                                     )
                enhanced_snr_list.append(Snr(np.loadtxt(join(enhanced_path, enhanced_list[index])),
                                             np.loadtxt(join(clean_path, clean_list[index])))
                                         )
            else:
                print("mismatch, check your data directory!")

        return np.mean(np.array(init_snr_list)), np.mean(np.array(enhanced_snr_list))





if __name__ == "__main__":
    enhanced_path = "Y"
    noisy_path = "X"
    init_snr, enhanced_snr = dataset_snr(enhanced_path= enhanced_path, noisy_path= noisy_path)
    print(f"primary SNR: {init_snr}| enhanced SNR: {enhanced_snr}| improvement: {enhanced_snr-init_snr}")

