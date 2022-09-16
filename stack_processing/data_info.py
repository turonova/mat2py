import numpy as np


class DataInfo:
    def __init__(self, folder_path, folder_prefix, tomo_list, tomo_digits, pixel_size, ctf_input, output_file):
        self.path = folder_path
        self.tomo_list = tomo_list
        self.folder_prefix = folder_prefix
        self.tilt_file = 'tilt.com'
        self.order_file = 'acquisition_order.txt'
        self.raw_suffix = ''
        self.dose_filt_suffix = '_dose-filt'
        self.tomo_digits = tomo_digits
        self.pixel_size = pixel_size
        self.dose = '_dose.txt'
        self.prior_dose_array = ''
        self.total_dose_array = ''
        self.corrected_dose_array = '_corrected_dose.txt'
        self.tlt_ext = '.tlt'
        self.rawtlt_ext = '.rawtlt'
        self.transform_suffix = '_fid.xf'
        self.fid_suffix = '_erase.fid'
        self.fid_binning = 8
        self.defocus_suffix = '_ctf_output.txt'
        self.defocus_format = 'ctffind4'
        self.defocus_comment_lines = [5]
        self.voltage = 300
        self.amplitude_contrast = 0.07
        self.cs = 2.7

        if ctf_input == '' or ctf_input == 'ctffind4':
            self.defocus_suffix = ['_ctf_output.txt']
            self.defocus_format = ['ctffind4']
            self.defocus_comment_lines = 5
        elif ctf_input == 'gctf':
            self.defocus_suffix = ['_gctf.star']
            self.defocus_format = ['gctf']
            self.defocus_comment_lines = 16
        elif ctf_input == 'both':
            self.defocus_suffix = ['_ctf_output.txt', '_gctf.star']
            self.defocus_format = ['ctffind4', 'gctf']
            self.defocus_comment_lines = [5, 16]

        if output_file != '':
            np.save(output_file, vars(self))
        else:
            np.save('data_info', vars(self))

    def get_field(self, field_name):
        if hasattr(self, field_name):
            return getattr(self, field_name)
        else:
            raise AttributeError(f'Field ": {field_name}  does not exist in the data_info structure!')

    def get_non_empty_field(self, field_name):
        field_name = self.get_field(field_name)
        if field_name == '':
            raise ValueError(f'Field ": {field_name} exists in the data_info but is empty!')

    def update(self, field_name, field_value):
        if field_name in ['defocus_format', 'defocus_suffix'] and type(field_value) == str:
            field_value = [field_value]
        setattr(self, field_name, field_value)
