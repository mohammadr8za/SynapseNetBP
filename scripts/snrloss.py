import torch
import numpy as np




def snr_loss(predicted_signal, target_signal):
    noise = predicted_signal - target_signal
    signal_power = torch.mean(predicted_signal ** 2)
    noise_power = torch.mean(noise ** 2)
    snr = (10 *np.log10(signal_power)) /(10 * np.log10( noise_power))
    # Return the negative SNR as the loss (to be minimized)
    return snr