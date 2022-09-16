import unittest
import numpy as np
import scipy.io as io
import os
import shutil

from stack_processing import DataInfo, dose_get_corrected_array


def fix_precision(array):
    # Rounds a float-number array such that each number has up to 5 digits including integers and decimals
    # Only works with numbers which absolute value is less than 100000
    for index, number in enumerate(array):
        if number < 0:
            number = round(-number, 5 - str(number).find('.'))
            array[index] = -number
        else:
            array[index] = round(number, 5 - str(number).find('.'))

    return array


class DoseGetCorrectedArrayTestCase(unittest.TestCase):
    def __init__(self, *args, **kwargs):
        super(DoseGetCorrectedArrayTestCase, self).__init__(*args, **kwargs)

        self.folder_prefix = 'TS_'
        self.tomo_digits = '155'
        self.init_exposure_dose = 3.5

        # Creates a temporary testing directory
        self.directory = f'{self.folder_prefix}{self.tomo_digits}/'
        self.path_prefix = f'{self.directory}{self.tomo_digits}'

        if not os.path.exists(self.directory):
            os.makedirs(self.directory)

        for file_name in (f'_corrected_dose_with_init_dose_{self.init_exposure_dose}.txt', f'_corrected_dose_with_init_dose_{self.init_exposure_dose}_ao.txt',
                          '_corrected_dose_with_init_dose_zero.txt', '_corrected_dose_with_init_dose_zero_ao.txt',
                          '_dose.txt', '_prior_dose.txt', '_total_dose.txt'):
            shutil.copy2(f'../example_files/{self.tomo_digits}{file_name}', self.directory)
        shutil.copy2('../example_files/acquisition_order.txt', self.directory)

        # Parses data info from the mat file
        raw_data_info = io.loadmat('../example_files/data_info_sars.mat')['data_info']
        extracted_data_info = {key: raw_data_info[key][0][0][0] for key in
                               ('path', 'folder_prefix', 'tomo_list', 'tomo_digits', 'pixel_size', 'defocus_format')}

        if len(extracted_data_info['defocus_format'][0]) == 1:
            ctf_input = extracted_data_info['defocus_format'][0][0]
        else:
            ctf_input = 'both'

        self.data_info = DataInfo(extracted_data_info['path'], extracted_data_info['folder_prefix'],
                                  extracted_data_info['tomo_list'],
                                  extracted_data_info['tomo_digits'][0], extracted_data_info['pixel_size'][0],
                                  ctf_input, '')
        self.data_info.update('path', '')
        self.data_info.update('folder_prefix', self.folder_prefix)

    def tearDown(self):
        # Removes the temporary testing folder
        if os.path.exists(self.directory):
            shutil.rmtree(self.directory)

        # Removes the saved data info
        if os.path.exists('data_info.npy'):
            os.remove('data_info.npy')

    # Case 1.1
    def test_zero_init_with_corrected_dose_array(self):
        self.data_info.update('corrected_dose_array', '_corrected_dose_with_init_dose_zero.txt')

        expected_corrected_dose = np.loadtxt(f'{self.path_prefix}_corrected_dose_with_init_dose_zero.txt')

        corrected_dose = dose_get_corrected_array(self.data_info, self.tomo_digits)
        fix_precision(corrected_dose)
        self.assertTrue(np.testing.assert_almost_equal(corrected_dose, expected_corrected_dose, 3) is None)

    # Case 2.1
    def test_zero_init_with_total_dose_array(self):
        self.data_info.update('corrected_dose_array', '')
        self.data_info.update('total_dose_array', '_total_dose.txt')

        expected_corrected_dose = np.loadtxt(f'{self.path_prefix}_corrected_dose_with_init_dose_zero.txt')

        corrected_dose = dose_get_corrected_array(self.data_info, self.tomo_digits)
        fix_precision(corrected_dose)
        self.assertTrue(np.testing.assert_almost_equal(corrected_dose, expected_corrected_dose, 3) is None)

    # Case 3.1
    def test_zero_init_with_prior_dose_array(self):
        self.data_info.update('corrected_dose_array', '')
        self.data_info.update('total_dose_array', '')
        self.data_info.update('prior_dose_array', '_prior_dose.txt')

        expected_corrected_dose = np.loadtxt(f'{self.path_prefix}_corrected_dose_with_init_dose_zero.txt')

        corrected_dose = dose_get_corrected_array(self.data_info, self.tomo_digits)
        fix_precision(corrected_dose)
        self.assertTrue(np.testing.assert_almost_equal(corrected_dose, expected_corrected_dose, 3) is None)

    # Case 4.1
    def test_zero_init_with_only_dose_and_acquisition_order_file(self):
        self.data_info.update('corrected_dose_array', '')
        self.data_info.update('total_dose_array', '')
        self.data_info.update('prior_dose_array', '')

        expected_corrected_dose = np.loadtxt(f'{self.path_prefix}_corrected_dose_with_init_dose_zero_ao.txt')

        corrected_dose = dose_get_corrected_array(self.data_info, self.tomo_digits)
        fix_precision(corrected_dose)
        self.assertTrue(np.testing.assert_almost_equal(corrected_dose, expected_corrected_dose, 3) is None)

    # Case 2.2
    def test_3_5_init_with_corrected_dose_array(self):
        self.data_info.update('corrected_dose_array', '')
        self.data_info.update('total_dose_array', '_total_dose.txt')

        expected_corrected_dose = np.loadtxt(f'{self.path_prefix}_corrected_dose_with_init_dose_{self.init_exposure_dose}.txt')

        corrected_dose = dose_get_corrected_array(self.data_info, self.tomo_digits, self.init_exposure_dose)
        fix_precision(corrected_dose)
        self.assertTrue(np.testing.assert_almost_equal(corrected_dose, expected_corrected_dose, 3) is None)

    # Case 3.2
    def test_3_5_init_with_prior_dose_array(self):
        self.data_info.update('corrected_dose_array', '')
        self.data_info.update('total_dose_array', '')
        self.data_info.update('prior_dose_array', '_prior_dose.txt')

        expected_corrected_dose = np.loadtxt(f'{self.path_prefix}_corrected_dose_with_init_dose_{self.init_exposure_dose}.txt')

        corrected_dose = dose_get_corrected_array(self.data_info, self.tomo_digits, self.init_exposure_dose)
        fix_precision(corrected_dose)
        self.assertTrue(np.testing.assert_almost_equal(corrected_dose, expected_corrected_dose, 3) is None)

    # Case 4.2
    def test_3_5_init_with_only_dose_and_acquisition_order_file(self):
        self.data_info.update('corrected_dose_array', '')
        self.data_info.update('total_dose_array', '')
        self.data_info.update('prior_dose_array', '')

        expected_corrected_dose = np.loadtxt(f'{self.path_prefix}_corrected_dose_with_init_dose_{self.init_exposure_dose}_ao.txt')

        corrected_dose = dose_get_corrected_array(self.data_info, self.tomo_digits, self.init_exposure_dose)
        fix_precision(corrected_dose)
        self.assertTrue(np.testing.assert_almost_equal(corrected_dose, expected_corrected_dose, 3) is None)


if __name__ == '__main__':
    unittest.main()
