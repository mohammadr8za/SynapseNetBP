import torch
import numpy as np




def Snr(predicted_signal, target_signal):
    noise = predicted_signal - target_signal
    signal_power = np.mean(np.array( target_signal) ** 2)
    noise_power =  np.mean(np.array( noise) ** 2)
    # snr = (10 * np.log10( noise_power)) / (10 *np.log10(signal_power))
    snr = 10 * np.log10(signal_power / noise_power)
    # Return the negative SNR as the loss (to be minimized)
    return snr