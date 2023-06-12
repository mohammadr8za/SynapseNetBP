import numpy as np
import os
import matplotlib.pyplot as plt

abp_save_dir = r'D:\data\PPGBP\mimic\abp'
chunk_name = '1-958-9.txt'

from scipy.signal import find_peaks

abp_chunk = np.loadtxt(os.path.join(abp_save_dir, chunk_name))
plt.figure()
plt.plot(np.arange(0, abp_chunk.shape[0]), abp_chunk, color='maroon')
plt.title(chunk_name)
plt.grid()
plt.show()

blood_pressure_list = []

sys_peaks, _ = find_peaks(abp_chunk, prominence=10, width=10)
plt.figure()
plt.plot(np.arange(0, abp_chunk.shape[0]), abp_chunk, color='maroon')
plt.plot(sys_peaks, abp_chunk[sys_peaks], 'x', color='b')
plt.title(chunk_name)
plt.grid()
plt.show()
sbp = int(np.median(abp_chunk[sys_peaks]))
blood_pressure_list.append(sbp)

abp_chunk_reverse = (-abp_chunk) + np.max(abp_chunk+10)
dias_peaks, _ = find_peaks(abp_chunk_reverse, prominence=10, width=10)
plt.figure()
plt.plot(np.arange(0, abp_chunk.shape[0]), abp_chunk, color='maroon')
plt.plot(dias_peaks, abp_chunk[dias_peaks], 'x', color='g')
plt.title(chunk_name)
plt.grid()
plt.show()
dbp = int(np.median(abp_chunk[dias_peaks]))
blood_pressure_list.append(dbp)

blood_pressure = np.array(blood_pressure_list)




1+1