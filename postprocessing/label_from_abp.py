import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
from os.path import join
from pathlib import Path
from scipy.signal import find_peaks


def blood_pressure_csv(csv_file, plot=False, txt=False, save_dir=None):

    abp_list = pd.read_csv(csv_file)

    sbp_list, dbp_list = [], []
    for i in range(len(abp_list)):

        abp_txt = join(abp_list['dir'][i], abp_list['signal_folder'][i], abp_list['signal_name'][i])
        abp_chunk = np.loadtxt(abp_txt)

        if plot:
            plt.figure()
            plt.plot(np.arange(0, abp_chunk.shape[0]), abp_chunk, color='maroon')
            plt.title(chunk_name)
            plt.grid()
            plt.show()
            # Save Plot Address

        # Systolic Blood Pressure (SBP)
        sys_peaks, _ = find_peaks(abp_chunk, prominence=10, width=10)

        if plot:
            plt.figure()
            plt.plot(np.arange(0, abp_chunk.shape[0]), abp_chunk, color='maroon')
            plt.plot(sys_peaks, abp_chunk[sys_peaks], 'x', color='b')
            plt.title(chunk_name)
            plt.grid()
            plt.show()
            # Save Plot Address

        sbp = int(np.median(abp_chunk[sys_peaks]))
        sbp_list.append(sbp)

        # Diastolic Blood Pressure (DBP)
        abp_chunk_reverse = (-abp_chunk) + np.max(abp_chunk + 10)
        dias_peaks, _ = find_peaks(abp_chunk_reverse, prominence=10, width=10)

        if plot:
            plt.figure()
            plt.plot(np.arange(0, abp_chunk.shape[0]), abp_chunk, color='maroon')
            plt.plot(dias_peaks, abp_chunk[dias_peaks], 'x', color='g')
            plt.title(chunk_name)
            plt.grid()
            plt.show()

        dbp = int(np.median(abp_chunk[dias_peaks]))
        dbp_list.append(dbp)

    if txt:
        save_path = Path(save_dir) / "txt"

        if save_path.is_dir():
            print("[INFO] This path already exists!")
        else:
            print("Creating path ...")
            save_path.mkdir(parents=True, exist_ok=True)
            print("Created!")

        sbp_arr = np.array(sbp_list)
        dbp_arr = np.array(dbp_list)
        np.savetxt(join(save_path, "systolic_bp" + "_" + Path(csv_file).name[:-3] + "txt"), sbp_arr)
        np.savetxt(join(save_path, "diastolic_bp" + "_" + Path(csv_file).name[:-3]) + "txt", dbp_arr)

    return sbp_arr, dbp_arr

def blood_pressure_single_abp(abp_file, plot=False):

    abp_chunk = np.loadtxt(abp_file)
    if plot:
        plt.figure()
        plt.plot(np.arange(0, abp_chunk.shape[0]), abp_chunk, color='maroon')
        plt.title(chunk_name)
        plt.grid()
        plt.show()

    # Systolic Blood Pressure (SBP)
    sys_peaks, _ = find_peaks(abp_chunk, prominence=10, width=10)

    if plot:
        plt.figure()
        plt.plot(np.arange(0, abp_chunk.shape[0]), abp_chunk, color='maroon')
        plt.plot(sys_peaks, abp_chunk[sys_peaks], 'x', color='b')
        plt.title(chunk_name)
        plt.grid()
        plt.show()

    sbp = int(np.median(abp_chunk[sys_peaks]))

    # Diastolic Blood Pressure (DBP)
    abp_chunk_reverse = (-abp_chunk) + np.max(abp_chunk + 10)
    dias_peaks, _ = find_peaks(abp_chunk_reverse, prominence=10, width=10)

    if plot:
        plt.figure()
        plt.plot(np.arange(0, abp_chunk.shape[0]), abp_chunk, color='maroon')
        plt.plot(dias_peaks, abp_chunk[dias_peaks], 'x', color='g')
        plt.title(chunk_name)
        plt.grid()
        plt.show()

    dbp = int(np.median(abp_chunk[dias_peaks]))

    return sbp, dbp


if __name__ == "__main__":

    abp_save_dir = r'D:\data\PPGBP\mimic\abp'
    chunk_name = '1-958-9.txt'
    abp_file = join(abp_save_dir, chunk_name)

    sbp, dbp = blood_pressure_single_abp(abp_file, plot=True)
    sbp, dbp
