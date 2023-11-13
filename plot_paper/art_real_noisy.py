import matplotlib.pyplot as plt
import numpy as np
from os.path import join

path_real = r'D:\PPG2ABP\DenoisingNetwork\DATA\initial_ppg_abp\512samples\ppg_p1\txt'
real_noisy_1 = np.loadtxt(join(path_real, '1-108-40.txt'))
real_noisy_2 = np.loadtxt(join(path_real, '1-385-43.txt'))
real_noisy_3 = np.loadtxt(join(path_real, '1-307-19.txt'))
real_noisy_4 = np.loadtxt(join(path_real, '1-4-33.txt'))

path_art = r'D:\PPG2ABP\data_for_training_split_shuffle\DNET\ppg_noisy_raw'
art_noisy_1 = np.loadtxt(join(path_art, r'scale\sample_loss_200\noise_level_3\txt', '1-176-109.txt'))
art_noisy_2 = np.loadtxt(join(path_art, r'sectoral\sample_loss_150\txt', '1-319-53.txt'))
art_noisy_3 = np.loadtxt(join(path_art, r'peak_distorted\ndp_1\txt', '1-230-104.txt'))
art_noisy_4 = np.loadtxt(join(path_art, r'shift_distorted\sample_loss_150\txt', '1-232-76.txt'))

# Plot DATA
fig, axes = plt.subplots(nrows=4, ncols=2)
axes[0][0].plot(art_noisy_1, color='b')
axes[0][0].set_title('Artificially Distorted PPG')
axes[0][0].axis(False)
axes[0][1].plot(real_noisy_1, color='maroon')
axes[0][1].set_title('Originally Distorted PPG')
axes[0][1].axis(False)

axes[1][0].plot(art_noisy_2, color='b')
# axes[1][0].set_title('Artificially Distorted PPG')
axes[1][0].axis(False)
axes[1][1].plot(real_noisy_2, color='maroon')
# axes[1][1].set_title('Distorted PPG from Main Dataset')
axes[1][1].axis(False)
#
axes[2][0].plot(art_noisy_3, color='b')
# axes[2][0].set_title('Artificially Distorted PPG')
axes[2][0].axis(False)
axes[2][1].plot(real_noisy_3, color='maroon')
# axes[2][1].set_title('Distorted PPG from Main Dataset')
axes[2][1].axis(False)

axes[3][0].plot(art_noisy_4, color='b')
# axes[3][0].set_title('Artificially Distorted PPG')
axes[3][0].axis(False)
axes[3][1].plot(real_noisy_4, color='maroon')
# axes[3][1].set_title('Distorted PPG from Main Dataset')
axes[3][1].axis(False)

fig.tight_layout()


directory = r"C:\Users\ASUS\Desktop\Paper=plots\alignmrnt_and_chunk"
plt.savefig(join(directory, "art_real_noisy.pdf"))

plt.show()

1




