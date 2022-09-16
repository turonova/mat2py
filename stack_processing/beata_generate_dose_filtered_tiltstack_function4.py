import mrcfile as mrc
import numpy as np
from stack_processing import string_fill_with_zeros
from .will_dose_filter_single_batchfunction import will_dose_filter_single_batchfunction


def beata_generate_dose_filtered_tiltstack_function4(output_suffix, tomo_str, dose_array, pixelsize):
    tilt_stack = tomo_str + '.st'

    print('Reading tilt-stack for tomo', tomo_str)
    t_stack = mrc.read(tilt_stack)
    t_stack = t_stack.Value.astype(float)
    ims_x = t_stack.shape[0]
    ims_y = t_stack.shape[1]
    n_tilts = t_stack.shape[2]

    print('Calculating frequency array')

    freq_array = np.zeros((ims_x, ims_y))
    cen_x = (ims_x / 2) + 1
    cen_y = (ims_y / 2) + 1

    rstep_x = 1 / (ims_x * pixelsize)
    rstep_y = 1 / (ims_y * pixelsize)

    for x in range(ims_x):
        for y in range(ims_y):
            d = ((x - cen_x) ** 2 * rstep_x ** 2 + (y - cen_y) ** 2 * rstep_y ** 2) ** 0.5
            freq_array[x, y] = d

    data_size_gb = (ims_x * ims_y * n_tilts * 4) / 1000000000
    if data_size_gb > 7:
        split_stack = True
        filt_stack = np.zeros((ims_x, ims_y))
        tilt_digits = len(str(n_tilts))
    else:
        split_stack = False
        filt_stack = np.zeros((ims_x, ims_y, n_tilts))

    for i in range(n_tilts):
        print(f'Filter tomo {tomo_str} tilt {str(i)}')
        if split_stack:
            filt_stack = will_dose_filter_single_batchfunction(t_stack[:, :, i], dose_array[i], freq_array)
            tilt_str = string_fill_with_zeros(i, tilt_digits)
            mrc.write(filt_stack, 'name', f'{tomo_str}{output_suffix}.mrc.{tilt_str}')
        else:
            filt_stack[:, :, i] = will_dose_filter_single_batchfunction(t_stack[:, :, i], dose_array[i], freq_array)

    if not split_stack:
        print(f'Writing dose filtered tomo {tomo_str}')
        mrc.write(filt_stack, 'name', f'{tomo_str}{output_suffix}.st')
