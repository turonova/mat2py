import numpy as np
import os
import pandas as pd
import emfile


class Motl:
    # Motl module example usage
    #
    # Initialize a Motl instance from an emfile
    #   `motl = Motl.load(’path_to_em_file’)`
    # Run clean_by_otsu and write the result to a new file
    #   `motl.clean_by_otsu(4, histogram_bin=20).write_to_emfile('path_to_output_em_file')`
    # Run class_consistency on multiple Motl instances
    #   `motl_intersect, motl_bad,cl_overlap = Motl.class_consistency(Motl.load('emfile1', 'emfile2', 'emfile3'))`

    def __init__(self, motl_df):
        self.df = motl_df

    @staticmethod
    def create_empty_motl():
        empty_motl_df = pd.DataFrame(columns = ['score', 'geom1', 'geom2', 'subtomo_id', 'tomo_id', 'object_id',
                                                'subtomo_mean', 'x', 'y', 'z', 'shift_x', 'shift_y', 'shift_z', 'geom4',
                                                'geom5', 'geom6', 'phi', 'psi', 'theta', 'class'])
        return empty_motl_df

    @classmethod
    def load(cls, *args):
        # Input: Load one or more emfiles, or already initialized instances of the Motl class
        # Output: Returns one instance, or a list of instances if multiple inputs are provided

        loaded = list()
        for motl in args:
            if os.path.isfile(motl) and ('.em' in motl):  # TODO can emfile have any other suffix?
                new_motl = cls.read_from_emfile(motl)
            elif isinstance(motl, cls):
                new_motl = motl
            else:
                # TODO or will it still be possible to receive the motl in form of a pure matrix?
                raise F'Unknown input type: {motl}. Input needs to be either an emfile, or an instance of the Motl class.'
            loaded.append(new_motl)

        if len(loaded) == 1:
            loaded = loaded[0]

        return loaded

    @classmethod
    def read_from_emfile(cls, emfile_path):
        # TODO read in the EM file to get the data
        tomo_number, object_number, coordinates, angles = emfile_path
        number_of_particles = coordinates.shape[0]

        motl = cls.create_empty_motl()
        motl['subtomo_id'] = np.arange(1, number_of_particles + 1, 1)
        motl['tomo_id'] = tomo_number
        motl['object_id'] = object_number
        motl['class'] = 1

        # round coord
        motl[['x', 'y', 'z']] = np.round(coordinates.values)

        # get shifts
        motl[['shift_x', 'shift_y', 'shift_z']] = coordinates.values - np.round(coordinates.values)

        # assign angles
        motl[['phi', 'psi', 'theta']] = angles.values

        return cls(motl)

    def write_to_emfile(self, outfile_path):
        motl_array = self.df.values
        motl_array = motl_array.reshape((1, motl_array.shape[0], motl_array.shape[1]))
        emfile.write(outfile_path, motl_array, overwrite=True)

    @staticmethod
    def motl_batch_stopgap2em(motl_base_name,iteration_range):
        # to be defined
        pass

    @classmethod
    def motl_class_consistency(cls, motl1, motl2):
        # to be defined
        pass
