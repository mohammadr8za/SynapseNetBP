import os
import random

from matplotlib import pyplot as plt
import numpy as np
from os.path import join
from pathlib import Path
from os import listdir
from scipy.signal import find_peaks

os.environ['KMP_DUPLICATE_LIB_OK'] = 'True'


def noise_factor_definer(noise_level, count, len):
    if noise_level == '1':
        if count < int((1 / 2) * len):
            noise_factor = 0.75
        else:
            noise_factor = 1.25
    elif noise_level == '2':
        if count < int((1 / 2) * len):
            noise_factor = 0.5
        else:
            noise_factor = 1.5
    elif noise_level == '3':
        if count < int((1 / 2) * len):
            noise_factor = 0.25
        else:
            noise_factor = 1.75
    else:
        print('[INFO] No such level of noise is defined!, choose a level of 1, 2, or 3')

    return noise_factor


def distortion(ppg_chunk, num_peaks, rnd_peaks, count, length):

    divide = int((1 / 2) * length)

    if num_peaks > 5:
        if count <= divide:
            for peak in rnd_peaks:
                ppg_chunk[peak - 5: peak] = np.flip(ppg_chunk[peak - 10: peak - 5])
                ppg_chunk[peak: peak + 7] = np.flip(ppg_chunk[peak + 7: peak + 14])
        else:
            for peak in rnd_peaks:
                ppg_chunk[peak - 5: peak] += ppg_chunk[peak - 5: peak]
                ppg_chunk[peak: peak + 5] += ppg_chunk[peak: peak + 5]

    else:
        if count <= divide:
            for peak in rnd_peaks:
                ppg_chunk[peak - 5: peak] = np.flip(ppg_chunk[peak - 10: peak - 5])
                ppg_chunk[peak: peak + 10] = np.flip(ppg_chunk[peak + 10: peak + 20])
        else:
            for peak in rnd_peaks:
                ppg_chunk[peak - 5: peak] += ppg_chunk[peak - 5: peak]
                ppg_chunk[peak: peak + 10] += ppg_chunk[peak: peak + 10]






class make_noisy():

    def __init__(self, clean_ppg_directory, clean_ppg_txt_directory, noisy_ppg_save_directory, num_loss_sample):

        self.clean_ppg_directory = clean_ppg_directory
        self.clean_ppg_txt_directory = clean_ppg_txt_directory
        self.noisy_ppg_save_directory = noisy_ppg_save_directory
        self.num_loss_sample = num_loss_sample
        self.clean_ppg_list = listdir(clean_ppg_directory)
        self.NLS = num_loss_sample

    def scale(self, level: int, plot=True, txt=True):

        noise_level = str(level)

        save_noisy_output_directory = join(self.noisy_ppg_save_directory, 'scale', 'sample_loss_' + str(self.NLS),
                                           'noise_level_' + noise_level)

        path = Path(save_noisy_output_directory)
        path.mkdir(parents=True, exist_ok=True)
        if txt:
            txt_path = path / 'txt'
            txt_path.mkdir(parents=True, exist_ok=True)
        if plot:
            plot_path = path / 'fig'
            plot_path.mkdir(parents=True, exist_ok=True)

        for counter, clean_ppg_name in enumerate(self.clean_ppg_list):

            noise_factor = noise_factor_definer(noise_level, counter, len(self.clean_ppg_list))

            ppg_chunk = np.loadtxt(os.path.join(self.clean_ppg_txt_directory, clean_ppg_name.replace('png', 'txt')))

            # start_index: distortion_start_index_random
            start_index = np.random.randint(low=0, high=ppg_chunk.shape[0] - self.NLS, size=1)

            ppg_chunk[start_index.item():int(start_index.item() + self.NLS)] *= noise_factor

            if plot:
                plt.figure()
                plt.plot(ppg_chunk)
                plt.title(f'Noise Type: Scale| Noise Level: {noise_level}| Noise Factor: {noise_factor}')
                plt.savefig(join(plot_path, clean_ppg_name))
                # plt.show()
                plt.close()
            if txt:
                np.savetxt(join(txt_path, clean_ppg_name.replace('png', 'txt')), ppg_chunk)

    def sectoral_loss(self, plot=True, txt=True):

        save_noisy_output_directory = join(self.noisy_ppg_save_directory, 'sectoral', 'sample_loss_' + str(self.NLS))

        path = Path(save_noisy_output_directory)
        path.mkdir(parents=True, exist_ok=True)
        if txt:
            txt_path = path / 'txt'
            txt_path.mkdir(parents=True, exist_ok=True)
        if plot:
            plot_path = path / 'fig'
            plot_path.mkdir(parents=True, exist_ok=True)

        for clean_ppg_name in self.clean_ppg_list:

            ppg_chunk = np.loadtxt(os.path.join(self.clean_ppg_txt_directory, clean_ppg_name.replace('png', 'txt')))

            # start_index: distortion_start_index_random
            start_index = np.random.randint(low=0, high=ppg_chunk.shape[0] - self.NLS, size=1)

            ppg_chunk[start_index.item():int((start_index.item() + self.NLS))] = \
                (np.mean(ppg_chunk)) * 2 * np.random.rand(self.NLS)

            if plot:
                plt.figure()
                plt.plot(ppg_chunk)
                plt.title(f'Noise Type: Sectoral')
                plt.savefig(join(plot_path, clean_ppg_name))
                # plt.show()
                plt.close()
            if txt:
                np.savetxt(join(txt_path, clean_ppg_name.replace('png', 'txt')), ppg_chunk)

    def sample_loss(self, plot=True, txt=True):

        save_noisy_output_directory = join(self.noisy_ppg_save_directory, 'sample', 'sample_loss_' + str(self.NLS))

        path = Path(save_noisy_output_directory)
        path.mkdir(parents=True, exist_ok=True)
        if txt:
            txt_path = path / 'txt'
            txt_path.mkdir(parents=True, exist_ok=True)
        if plot:
            plot_path = path / 'fig'
            plot_path.mkdir(parents=True, exist_ok=True)

        for clean_ppg_name in self.clean_ppg_list:

            ppg_chunk = np.loadtxt(os.path.join(self.clean_ppg_txt_directory, clean_ppg_name.replace('png', 'txt')))

            random_samples_loss = np.sort(np.random.randint(low=0, high=ppg_chunk.shape[0], size=self.NLS))

            ppg_chunk[random_samples_loss] = np.random.rand(random_samples_loss.shape[0])

            if plot:
                plt.figure()
                plt.plot(ppg_chunk)
                plt.title(f'Noise Type: Sample Loss')
                plt.savefig(join(plot_path, clean_ppg_name))
                # plt.show()
                plt.close()
            if txt:
                np.savetxt(join(txt_path, clean_ppg_name.replace('png', 'txt')), ppg_chunk)

    def shift_distortion(self, plot=True, txt=True):

        # Directory: noise folder - distortion type - num of sample loss
        save_noisy_output_directory = join(self.noisy_ppg_save_directory, 'shift_distorted',
                                           'sample_loss_' + str(self.NLS))

        path = Path(save_noisy_output_directory)
        path.mkdir(parents=True, exist_ok=True)
        if txt:
            txt_path = path / 'txt'
            txt_path.mkdir(parents=True, exist_ok=True)
        if plot:
            plot_path = path / 'fig'
            plot_path.mkdir(parents=True, exist_ok=True)

        for count, clean_ppg_name in enumerate(self.clean_ppg_list):

            ppg_chunk = np.loadtxt(os.path.join(self.clean_ppg_txt_directory, clean_ppg_name.replace('png', 'txt')))

            start_index = np.random.randint(low=0, high=ppg_chunk.shape[0] - self.NLS, size=1)

            ppg_chunk[start_index.item():int((start_index.item() + self.NLS))] += np.power(-1, count) * np.mean(
                ppg_chunk)

            if plot:
                plt.figure()
                plt.plot(ppg_chunk)
                plt.title(f'Noise Type: Shift Distortion')
                plt.savefig(join(plot_path, clean_ppg_name))
                # plt.show()
                plt.close()
            if txt:
                np.savetxt(join(txt_path, clean_ppg_name.replace('png', 'txt')), ppg_chunk)

    def peak_distorted(self, ndp: int, plot=True, txt=True):
        # ndp: number of distorted peaks

        # save_noisy_output_directory = join(self.noisy_ppg_save_directory, 'peak_distorted',
        #                                    'sample_loss_' + str(self.NLS))
        save_noisy_output_directory = join(self.noisy_ppg_save_directory, 'peak_distorted')


        path = Path(save_noisy_output_directory)
        path.mkdir(parents=True, exist_ok=True)
        if txt:
            txt_path = path / 'txt'
            txt_path.mkdir(parents=True, exist_ok=True)
        if plot:
            plot_path = path / 'fig'
            plot_path.mkdir(parents=True, exist_ok=True)

        for count, clean_ppg_name in enumerate(self.clean_ppg_list):

            ppg_chunk = np.loadtxt(os.path.join(self.clean_ppg_txt_directory, clean_ppg_name.replace('png', 'txt')))

            peaks, _ = find_peaks(ppg_chunk, width=20, distance=20)

            if ndp <= peaks.shape[0] - 1:
                rnd_peaks = np.random.choice(peaks, size=ndp, replace=False)
            else:
                rnd_peaks = np.random.choice(peaks, size=ppg_chunk.shape[0], replace=False)

            distortion(ppg_chunk, peaks.shape[0], rnd_peaks, count, len(self.clean_ppg_list))

            if plot:
                plt.figure()
                plt.plot(ppg_chunk)
                plt.title(f'Noise Type: Peak Distortion')
                plt.savefig(join(plot_path, clean_ppg_name))
                # plt.show()
                plt.close()
            if txt:
                np.savetxt(join(txt_path, clean_ppg_name.replace('png', 'txt')), ppg_chunk)


if __name__ == '__main__':
    # Test The Program
    clean_ppg_dir = r'D:\PPG2ABP\DenoisingNetwork\DATA\initial_ppg_abp\cleaning_512samples\fig_8000'
    clean_ppg_list = os.listdir(clean_ppg_dir)

    ppg_txt_dir = r'D:\PPG2ABP\DenoisingNetwork\DATA\initial_ppg_abp\512samples\ppg_p1\txt'
    noisy_ppg_save_dir = r'D:\PPG2ABP\DenoisingNetwork\DATA\noisy_ppg_created_from_clean_ppg'

    distort = make_noisy(clean_ppg_directory=clean_ppg_dir,
                         clean_ppg_txt_directory=ppg_txt_dir,
                         noisy_ppg_save_directory=noisy_ppg_save_dir,
                         num_loss_sample=50)
    distort.peak_distorted(ndp=3)
