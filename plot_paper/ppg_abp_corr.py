import numpy as np
import matplotlib.pyplot as plt
from os.path import join

directory = r"C:\Users\ASUS\Desktop\Paper=plots\alignmrnt_and_chunk"

raw_ppg = np.loadtxt(join(directory, "raw_ppg.txt"))
raw_abp = np.loadtxt(join(directory, "raw_abp.txt"))
shifted_abp = np.loadtxt(join(directory, "shifted_abp.txt"))

plt.figure(figsize=(8, 3))
plt.plot(raw_abp[:512], label="ABP Waveform"), plt.plot(raw_ppg[:512], label="PPG Wavefrom")
plt.axvline(x= 52, linestyle='--', color='maroon')
plt.grid()
plt.legend(loc=1)
plt.axis(False)
plt.savefig(join(directory, "unaligned.pdf"))
plt.show()



plt.figure(figsize=(8, 3))
plt.plot(shifted_abp[:512], label="ABP Waveform"), plt.plot(raw_ppg[:512], label="PPG Wavefrom")
plt.axvline(x= 52, linestyle='--', color='maroon')
plt.grid()
plt.legend(loc=1)
plt.axis(False)
plt.savefig(join(directory, "aligned.pdf"))
plt.show()



1+1



