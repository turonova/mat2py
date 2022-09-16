import numpy as np


def will_dose_filter_single_batchfunction(image, dose, freq_array):
    a = 0.245
    b = -1.665
    c = 2.81

    ft = np.fft.fftshift(np.fft.fft2(image))

    q = np.exp((-dose) / (2 * ((a * (freq_array ** b)) + c)))

    return np.fft.ifft2(np.fft.ifftshift(ft * q))
