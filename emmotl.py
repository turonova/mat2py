import numpy as np
import pandas as pd
import emfile


class Motl:
    ALLOWED_SOURCES = ['emfile']  # allowed source files

    def __init__(self, motl_df, source):
        self.motl_df = motl_df
        if source in self.ALLOWED_SOURCES:
            self.source = source
        else:
            raise

    @classmethod
    def create_empty_motl(cls, number_of_particles):
        empty_motl_df = pd.DataFrame(data=np.zeros(number_of_particles, 20))
        empty_motl_df.columns = ['score', 'geom1', 'geom2', 'subtomo_id', 'tomo_id', 'object_id', 'subtomo_mean',
                                 'x', 'y', 'z', 'shift_x', 'shift_y', 'shift_z', 'geom4', 'geom5', 'geom6',
                                 'phi', 'psi', 'theta', 'class']
        return empty_motl_df

    @classmethod
    def read_from_emfile(cls, emfile_path):
        # TODO read in the EM file to get the data
        tomo_number, object_number, coordinates, angles = emfile_path
        number_of_particles = coordinates.shape[0]

        motl = cls.create_empty_motl(number_of_particles)
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

        return cls(motl, 'emfile')

    def write_to_emfile(self, outfile_path):
        if self.source == 'emfile':
            motl_array = self.motl_df.values
            motl_array = motl_array.reshape((1, motl_array.shape[0], motl_array.shape[1]))

            emfile.write(outfile_path, motl_array, overwrite=True)
        else:
            raise f'Sorry, the origin of the data is {self.source}, which is not compatible with the EMfile format.'

    @staticmethod
    def motl_batch_stopgap2em(motl_base_name,iteration_range):
        # to be defined
        pass

    @classmethod
    def motl_class_consistency(cls, motl1, motl2):
        # to be defined
        pass
