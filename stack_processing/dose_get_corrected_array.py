import numpy as np
from .string_path_complete import string_path_complete


def dose_get_corrected_array(data_info, tomo_str, init_exposure_dose=None):
    folder_prefix = string_path_complete(data_info.path) + data_info.folder_prefix + tomo_str + '/'
    file_prefix = string_path_complete(data_info.path) + data_info.folder_prefix + tomo_str + '/' + tomo_str

    if init_exposure_dose is not None:
        init_dose = init_exposure_dose
    else:
        init_dose = 0

    if type(data_info.dose) == str:
        dose = np.loadtxt(file_prefix + data_info.dose)
    else:
        dose = data_info.dose

    if hasattr(data_info, 'corrected_dose_array') and len(data_info.corrected_dose_array) > 0 and init_exposure_dose is None:
        corrected_array = np.loadtxt(file_prefix + data_info.corrected_dose_array)
    elif hasattr(data_info, 'total_dose_array') and len(data_info.total_dose_array) > 0:
        corrected_array = np.loadtxt(file_prefix + data_info.total_dose_array)
        corrected_array = corrected_array + init_dose
    elif hasattr(data_info, 'prior_dose_array') and len(data_info.prior_dose_array) > 0:
        corrected_array = np.loadtxt(file_prefix + data_info.prior_dose_array)
        corrected_array = corrected_array + dose + init_dose
    else:
        order_tilt = np.loadtxt(folder_prefix + data_info.order_file)
        corrected_array = order_tilt[:, 0] * dose + init_dose
    return corrected_array
