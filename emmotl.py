import emfile
import numpy as np
import os
import pandas as pd
import starfile
import subprocess
from scipy.spatial.transform import Rotation as rot


from exceptions import UserInputError
from math import ceil
from matplotlib import pyplot as plt
from numpy.matlib import repmat
from scipy.interpolate import UnivariateSpline


class Motl:
    # Motl module example usage
    #
    # Initialize a Motl instance from an emfile
    #   `motl = Motl.load(’path_to_em_file’)`
    # Run clean_by_otsu and write the result to a new file
    #   `motl.clean_by_otsu(4, histogram_bin=20).write_to_emfile('path_to_output_em_file')`
    # Run class_consistency on multiple Motl instances
    #   `motl_intersect, motl_bad, cl_overlap = Motl.class_consistency(Motl.load('emfile1', 'emfile2', 'emfile3'))`

    def __init__(self, motl_df, header=None):
        self.df = motl_df
        self.header = header if header else {}

    @staticmethod
    def create_empty_motl():
        empty_motl_df = pd.DataFrame(columns=['score', 'geom1', 'geom2', 'subtomo_id', 'tomo_id', 'object_id',
                                              'subtomo_mean', 'x', 'y', 'z', 'shift_x', 'shift_y', 'shift_z', 'geom4',
                                              'geom5', 'geom6', 'phi', 'psi', 'theta', 'class'], dtype=float)
        return empty_motl_df

    @staticmethod  # TODO move to different module
    def pad_with_zeros(number, total_digits):
        # Creates a string of the length specified by total_digits, containing a given number and fills the rest
        # (at the beginning) with zeros, i.e. from the input parameters 3 and 6 it creates 000003)
        #
        # Input:  number - number to be padded with zero (converts the number to integer)
        #         total_digits - final length of the output string
        # Output: string of length specified by total_digits and containing the input number at the very end

        zeros = '0' * (total_digits - len(str(int(number))))
        padded_str = zeros + str(int(number))

        return padded_str

    @staticmethod
    def get_feature(cols, feature_id):
        if isinstance(feature_id, int):
            if feature_id < len(cols):
                feature = cols[feature_id]
            else:
                raise UserInputError(
                    f'Given feature index is out of bounds. The index must be within the range 0-{len(cols) - 1}.')
        else:
            if feature_id in cols:
                feature = feature_id
            else:
                raise UserInputError('Given feature name does not correspond to any motl column.')

        return feature

    @staticmethod  # TODO move to different (sgmotl) module?
    def batch_stopgap2em(motl_base_name, iter_no):
        em_list = []
        for i in range(iter_no):
            motl_path = f'{motl_base_name}_{str(i+1)}'
            star_motl = starfile.read(f'{motl_path}.star')
            motl = Motl.stopgap_to_av3(star_motl)
            em_path = f'{motl_path}.em'
            motl.write_to_emfile(em_path)
            em_list.append(em_path)

        return em_list

    @staticmethod  # TODO move to different module?
    def otsu_threshold(bin_counts):
        # Taken from: https://www.kdnuggets.com/2018/10/basic-image-analysis-python-p4.html
        s_max = (0, 0)

        for threshold in range(len(bin_counts)):
            # update
            w_0 = sum(bin_counts[:threshold])
            w_1 = sum(bin_counts[threshold:])

            mu_0 = sum([i * bin_counts[i] for i in range(0, threshold)]) / w_0 if w_0 > 0 else 0
            mu_1 = sum([i * bin_counts[i] for i in range(threshold, len(bin_counts))]) / w_1 if w_1 > 0 else 0

            # calculate - inter class variance
            s = w_0 * w_1 * (mu_0 - mu_1) ** 2

            if s > s_max[1]:
                s_max = (threshold, s)

        return s_max[0]

    @staticmethod
    def load_dimensions(dims):
        if os.path.isfile(dims):
            dimensions = pd.read_csv(dims, sep='\s+', header=None)
        else:
            dimensions = pd.DataFrame(dims)
        dimensions.columns = ['tomo_id', 'x', 'y', 'z']
        return dimensions

    @staticmethod  # TODO add tests
    def recenter_particles(df):
        new_df = df.copy()
        shifted_x = new_df.loc[:, 'x'] + new_df.loc[:, 'shift_x']
        shifted_y = new_df.loc[:, 'y'] + new_df.loc[:, 'shift_y']
        shifted_z = new_df.loc[:, 'z'] + new_df.loc[:, 'shift_z']

        new_df.loc[:, 'x'] = round(shifted_x)
        new_df.loc[:, 'y'] = round(shifted_y)
        new_df.loc[:, 'z'] = round(shifted_z)
        new_df.loc[:, 'shift_x'] = shifted_x - new_df.loc[:, 'x']
        new_df.loc[:, 'shift_y'] = shifted_y - new_df.loc[:, 'y']
        new_df.loc[:, 'shift_z'] = shifted_z - new_df.loc[:, 'z']

        return new_df

    @staticmethod
    def get_pairwise_distance(p1, p2):
        dist_func = lambda a, b: a + b

        p1_dot = np.reshape(np.array(p1 * p1), ((p1.shape[0],1)))
        p2_dot = p2 * p2

        dist_a = dist_func(p1_dot, p2_dot)
        dist_b = 2*(p1.reshape(((p1.shape[0],1)))*p2)
        dist = dist_a - dist_b

        # Find negative values
        neg_idx = dist < 0
        # Set negative values to zeroes
        dist[neg_idx] = 0

        dist = np.sqrt(dist)
        return dist

    @classmethod
    def read_from_emfile(cls, emfile_path):
        header, parsed_emfile = emfile.read(emfile_path)
        if not len(parsed_emfile[0][0]) == 20:
            raise UserInputError(
                f'Provided file contains {len(parsed_emfile[0][0])} columns, while 20 columns are expected.')

        motl = pd.DataFrame(data=parsed_emfile[0], dtype=float,
                            columns=['score', 'geom1', 'geom2', 'subtomo_id', 'tomo_id', 'object_id',
                                     'subtomo_mean', 'x', 'y', 'z', 'shift_x', 'shift_y', 'shift_z', 'geom4',
                                     'geom5', 'geom6', 'phi', 'psi', 'theta', 'class'])
        motl['class'] = motl['class'].fillna(1.0)

        # TODO do we want to keep these? (from the original emmotl.py)
        # motl['subtomo_id'] = np.arange(1, number_of_particles + 1, 1)
        # round coord
        # motl[['x', 'y', 'z']] = np.round(coordinates.values)
        # get shifts
        # motl[['shift_x', 'shift_y', 'shift_z']] = coordinates.values - np.round(coordinates.values)

        return cls(motl, header)

    @classmethod
    def load(cls, input_motl):
        # Input: Load one or more emfiles (as a list), or already initialized instances of the Motl class
        #        E.g. `Motl.load([cryo1.em, cryo2.em, motl_instance1])`
        # Output: Returns one instance, or a list of instances if multiple inputs are provided

        loaded = list()
        motls = [input_motl] if not isinstance(input_motl, list) else input_motl
        if len(motls) == 0:
            raise UserInputError('At least one em file or a Motl instance must be provided.')
        else:
            for motl in motls:
                if isinstance(motl, cls):
                    new_motl = motl
                elif isinstance(motl, str):
                    if os.path.isfile(motl):
                        if os.path.splitext(motl)[-1] == '.em':
                            new_motl = cls.read_from_emfile(motl)
                        else:
                            raise UserInputError(f'Unknown file type: {motl}. Input needs to be either an em file '
                                                 f'(.em), or an instance of the Motl class.')
                    else:
                        raise UserInputError(f'Provided file {motl} does not exist.')
                else:
                    raise UserInputError(f'Unkown input type ({motl}).')
                # TODO or will it still be possible to receive the motl in form of a pure matrix?

                if not np.array_equal(new_motl.df.columns, cls.create_empty_motl().columns):
                    raise UserInputError(f'Provided Motl object {motl} seems to be corrupted and can not be loaded.')
                else:
                    loaded.append(new_motl)

            if len(loaded) == 1:
                loaded = loaded[0]

        return loaded

    @classmethod  # TODO move to different (sgmotl) module?
    def stopgap_to_av3(cls, star_motl):
        # Accepts input read from the star file (using the starfile.read), and outputs instance of Motl class
        # To write the resulting em motl to the wile, run write_to_emfile.
        # Example: Motl.stopgap_to_av3(starfile.read('path_to_star_file')).write_to_emfile('path_to_output_emfile')

        motl = cls.create_empty_motl()
        # TODO do we want to use 'motl_idx' as index of the dataframe or drop it?
        pairs = {'subtomo_id': 'subtomo_num', 'tomo_id': 'tomo_num', 'object_id': 'object', 'x': 'orig_x',
                 'y': 'orig_y', 'z': 'orig_z', 'score': 'score', 'shift_x': 'x_shift', 'shift_y': 'y_shift',
                 'shift_z': 'z_shift', 'phi': 'phi', 'psi': 'psi', 'theta': 'the', 'class': 'class'}
        for em_key, star_key in pairs.items():
            motl[em_key] = star_motl[star_key]
        motl['geom4'] = [0.0 if hs.lower() == 'a' else 1.0 for hs in star_motl['halfset']]

        return cls(motl)

    @classmethod
    def merge_and_renumber(cls, motl_list):
        merged_df = cls.create_empty_motl()
        feature_add = 0

        if not isinstance(motl_list, list) or len(motl_list) == 0:
            raise UserInputError(f'You must provide a list of em file paths, or Motl instances. '
                                 f'Instead, an instance of {type(motl_list).__name__} was given.')

        for m in motl_list:
            motl = cls.load(m)
            feature_min = min(motl.df.loc[:, 'object_id'])

            if feature_min <= feature_add:
                motl.df.loc[:, 'object_id'] = motl.df.loc[:, 'object_id'] + (feature_add - feature_min + 1)

            merged_df = pd.concat([merged_df, motl.df])
            feature_add = max(motl.df.loc[:, 'object_id'])

        merged_motl = cls(merged_df)
        merged_motl.renumber_particles()
        return merged_motl

    @classmethod  # TODO too slow, do the comparison more efficiently
    def get_particle_intersection(cls, motl1, motl2):
        m1, m2 = cls.load([motl1, motl2])
        m2_values = m2.df.loc[:, 'subtomo_id'].unique()
        intersected = cls.create_empty_motl()

        for value in m2_values:
            submotl = m1.df.loc[m1.df['subtomo_id'] == value]
            intersected = pd.concat([intersected, submotl])

        return cls(intersected.reset_index(drop=True))

    @classmethod  # TODO add tests, maybe write in more pythonic way
    def class_consistency(cls, *args):
        # TODO check the whole functionality against the matlab version (+ contents of the resulting objects)
        if len(args) < 2:
            raise UserInputError('At least 2 motls are needed for this analysis')

        no_cls, all_classes = 1, []
        loaded = cls.load(list(args))
        min_particles = len(loaded[0].df)

        # get number of classes
        for motl in loaded:
            min_particles = min(min_particles, len(motl.df))
            clss = motl.df.loc[:, 'class'].unique()
            no_cls = max(len(clss), no_cls)
            if no_cls == len(clss): all_classes = clss

        cls_overlap = np.zeros((len(loaded) - 1, no_cls))
        # mid_overlap = np.zeros(min_particles, all_classes)
        # mid_overlap = np.zeros(min_particles, all_classes)

        motl_intersect = cls.create_empty_motl()
        motl_bad = cls.create_empty_motl()

        for i, cl in enumerate(all_classes):
            i_motl = loaded[0]
            i_motl_df = i_motl.df.loc[i_motl.df['class'] == cl]

            for j, motl in enumerate(loaded):
                if j == 0: continue
                j_motl_df = motl.df.loc[motl.df['class'] == cl]

                cl_o = len([el for el in i_motl_df.loc[:, 'subtomo_id'] if el in j_motl_df.loc[:, 'subtomo_id']])
                cls_overlap[j-1, i] = cl_o

                motl_bad = pd.concat(
                    [motl_bad, i_motl_df.loc[i_motl_df.subtomo_id.isin(j_motl_df.loc[:, 'subtomo_id'].values)]])
                motl_bad = pd.concat(
                    [motl_bad, j_motl_df.loc[j_motl_df.subtomo_id.isin(i_motl_df.loc[:, 'subtomo_id'].values)]])

                if cl_o != 0:
                    print(f'The overlap of motl #{j} and #{j+1} is {cl_o} ({cl_o / len(i_motl_df) * 100}% of motl '
                          f'#{j} and {cl_o / len(j_motl_df) * 100}% of motl #{j+1}.')
                    i_motl_df = i_motl_df.loc[i_motl_df.subtomo_id.isin(j_motl_df.loc[:, 'subtomo_id'].values)]
                else:
                    print(f'Warning: motl # {str(j+1)} does not contain class #{str(cl)}')

            motl_intersect = pd.concat([motl_intersect, i_motl_df])

        return [motl_intersect, motl_bad, cls_overlap]

    def write_to_emfile(self, outfile_path):
        # TODO currently replaces all missing values in the whole df, maybe should be more specific to some columns
        filled_df = self.df.fillna(0.0)
        motl_array = filled_df.to_numpy()
        motl_array = motl_array.reshape((1, motl_array.shape[0], motl_array.shape[1]))
        self.header = {}  # FIXME fails on writing back the header
        emfile.write(outfile_path, motl_array, self.header, overwrite=True)

    # FIXME apply correct coordinate conversion
    def write_to_model_file(self, feature_id, output_base, point_size, binning=None):
        feature = self.get_feature(self.df.columns, feature_id)
        uniq_values = self.df.loc[:, feature].unique()
        outpath = f'{output_base}_{feature}_'

        if binning:
            bin = binning
        else:
            bin = 1

        for value in uniq_values:
            fm = self.df.loc[self.df[feature] == value].reset_index(drop=True)
            feature_str = self.pad_with_zeros(value, 3)
            output_txt = f'{outpath}{feature_str}_model.txt'
            output_mod = f'{outpath}{feature_str}.mod'

            pos_x = (fm.loc[:, 'x'] + fm.loc[:, 'shift_x']) * bin
            pos_y = (fm.loc[:, 'y'] + fm.loc[:, 'shift_y']) * bin
            pos_z = (fm.loc[:, 'z'] + fm.loc[:, 'shift_z']) * bin
            klass = fm.loc[:, 'class']
            dummy = pd.Series(np.repeat(1, len(fm)))

            pos_df = pd.concat([klass, dummy, pos_x, pos_y, pos_z], axis=1)
            pos_df = pos_df.astype(int)
            pos_df.to_csv(output_txt, sep='\t', header=False, index=False)

            # Create model files from the coordinates
            # system(['point2model -sc -sphere ' num2str(point_size) ' ' output_txt ' ' output_mod]);
            # subprocess.run(['point2model', '-sc', '-sphere', str(point_size), output_txt, output_mod])

    def remove_feature(self, feature_id, feature_values):
        # Removes particles based on their feature (i.e. tomo number)
        # Inputs: feature_id - col name or index based on which the particles will be removed (i.e. 4 for tomogram id)
        #         feature_values - list of values to be removed
        #         output_motl_name - name of the new motl; if empty the motl will not be written out
        # Usage: motl.remove_feature(4, [3, 7, 8]) - removes all particles from tomograms number 3, 7, and 8

        feature = self.get_feature(self.df.columns, feature_id)

        if not feature_values:
            raise UserInputError(
                'You must specify at least one feature value, based on witch the particles will be removed.')
        else:
            if not isinstance(feature_values, list):
                feature_values = [feature_values]
            for value in feature_values:
                self.df = self.df.loc[self.df[feature] != value]

        return self

    def update_coordinates(self):
        self.df = self.recenter_particles(self.df)
        return self

    def tomo_subset(self, tomo_numbers):  # TODO add tests
        # Updates motl to contain only particles from tomograms specified by tomo numbers
        # Input: tomo_numbers - list of selected tomogram numbers to be included
        #        renumber_particles - renumber from 1 to the size of the new motl if True

        new_motl = self.__class__.create_empty_motl()
        for i in tomo_numbers:
            df_i = self.df.loc[self.df['tomo_id'] == i]
            new_motl = pd.concat([new_motl, df_i])
        self.df = new_motl

        return self

    def renumber_particles(self):  # TODO add tests
        # new_motl(4,:)=1: size(new_motl, 2);
        self.df.loc[:, 'subtomo_id'] = list(range(1, len(self.df)+1))
        return self

    def split_by_feature(self, feature_id, write_out=False, output_prefix=None, feature_desc_id=None):
        # Split motl by uniq values of a selected feature
        # Inputs:   feature_id - column name or index of the feature based on witch the motl will be split
        #           write: save all the resulting Motl instances into separate files if True
        #           output_prefix:
        #           feature_desc_id:  # TODO how should that var look like?
        # Output: list of Motl instances, each containing only rows with one unique value of the given feature

        feature = self.get_feature(self.df.columns, feature_id)
        uniq_values = self.df.loc[:, feature].unique()
        motls = list()

        for value in uniq_values:
            submotl = self.__class__(self.df.loc[self.df[feature] == value])
            motls.append(submotl)

            if write_out:
                if feature_desc_id:
                    for d in feature_desc_id:  # FIXME should not iterate here probably
                        # out_name=[out_name '_' num2str(nm(d,1))];
                        out_name = f'{output_prefix}_{str(d)}'
                    out_name = f'{out_name}_.em'
                else:
                    out_name = f'{output_prefix}_{str(value)}.em'
                submotl.write_to_emfile(out_name)

        return motls

    def clean_by_otsu(self, feature_id, histogram_bin=None):
        # Cleans motl by Otsu threshold (based on CC values)
        # feature_id: a feature by which the subtomograms will be grouped together for cleaning;
        #             4 or 'tomo_id' to group by tomogram, 5 to clean by a particle (e.g. VLP, virion)
        # histogram_bin: how fine to split the histogram. Default is 30 for feature 5 and 40 for feature 4;
        #             for smaller number of subtomograms per feature the number should be lower

        feature = self.get_feature(self.df.columns, feature_id)
        tomos = self.df.loc[:, 'tomo_id'].unique()
        cleaned_motl = self.__class__.create_empty_motl()

        if histogram_bin:
            hbin = histogram_bin
        else:
            if feature == 'tomo_id':
                hbin = 40
            elif feature == 'object_id':
                hbin = 30
            else:
                raise UserInputError(f'The selected feature ({feature}) does not correspond either to tomo_id, nor to'
                                     f'object_id. You need to specify the histogram_bin.')

        for t in tomos:  # if feature == object_id, tomo_id needs to be used too
            tm = self.df.loc[self.df['tomo_id'] == t]
            features = tm.loc[:, feature].unique()

            for f in features:
                fm = tm.loc[tm[feature] == f]
                bin_counts, bin_centers, _ = plt.hist(fm.loc[:, 'score'])
                bn = self.otsu_threshold(bin_counts)
                cc_t = bin_centers[bn]
                fm = fm.loc[fm['score'] >= cc_t]

                cleaned_motl = pd.concat([cleaned_motl, fm])

        self.df = cleaned_motl.reset_index(drop=True)
        return self

    def remove_out_of_bounds_particles(self, dimensions, boundary_type, box_size=None, recenter_particles=True):
        dim = self.load_dimensions(dimensions)
        original_size = len(self.df)

        # Get type of bounds
        if boundary_type == 'whole':
            if box_size:
                boundary = ceil(box_size/2)
            else:
                raise UserInputError("You need to specify box_size when boundary_type is set to 'whole'.")
        elif boundary_type == 'center':
            boundary = 0
        else:
            raise UserInputError(f'Unknown type of boundaries: {boundary_type}')

        recentered = self.recenter_particles(self.df)
        idx_list = []
        for i, row in recentered.iterrows():
            tn = row['tomo_id']
            tomo_dim = dim.loc[dim['tomo_id'] == tn, 'x':'z'].reset_index(drop=True)
            c_min = [c - boundary for c in row['x':'z']]
            c_max = [c + boundary for c in row['x':'z']]
            if (all(c_min) >= 0) and \
                    (c_max[0] < tomo_dim['x'][0]) and (c_max[1] < tomo_dim['y'][0]) and (c_max[2] < tomo_dim['z'][0]):
                idx_list.append(i)

        final_motl = recentered if recenter_particles else self.df
        self.df = final_motl.iloc[idx_list].reset_index(drop=True)

        print(f'Removed {original_size - len(self.df)} particles.')
        print(f'Original size {original_size}, new_size {len(self.df)}')

        return self

    ############################
    # PARTIALLY FINISHED METHODS

    def shift_positions(self, shift):
        # Shifts positions of all subtomgoram in the motl in the direction given by subtomos' rotations

        def shift_coords(row):
            v = np.array(shift)
            euler_angles = np.array([[row['phi'], row['psi'], row['theta']]])
            orientations = rot.from_euler(seq='zxz', angles=euler_angles, degrees=True)
            rshifts = orientations.apply(v) 
            
            row['shift_x'] = row['shift_x'] + rshifts
            row['shift_y'] = row['shift_y'] + rshifts
            row['shift_z'] = row['shift_z'] + rshifts
            return row

        self.df = self.df.apply(shift_coords, axis=1)
        return self

    @staticmethod
    def spline_sampling(coords, sampling_distance):
        # Samples a spline specified by coordinates with a given sampling distance
        # Input:  coords - coordinates of the spline
        #         sampling_distance: sampling frequency in pixels
        # Output: coordinates of points on the spline

        # F=spline(1:size(coord,2),coord)
        spline = UnivariateSpline(np.arrange(0, len(coords)), coords)  # TODO ensure right interpolation

        # Keep track of steps across whole tube
        totalsteps = 0

        for n in range(len(coords)):
            if n == 0: continue
            # Calculate projected distance between each point
            xc = coords[n, 'x'] - coords[n-1, 'x']
            yc = coords[n, 'y'] - coords[n-1, 'y']
            zc = coords[n, 'z'] - coords[n-1, 'z']

            # Calculate Euclidian distance between points
            dist = np.sqrt((xc ** 2) + (yc ** 2) + (zc ** 2))

            # Number of steps between two points; steps are roughly in increments of 1 pixel
            stepnumber = round(dist / sampling_distance)
            # Length of each step
            step = 1 / stepnumber
            # Array to hold fraction of each step between points
            t = np.arrange(n-1, n, step)  # inclusive end in matlab

            # Evaluate piecewise-polynomial, i.e. spline, at steps 't'.
            # This array contains the Cartesian coordinates of each step

            # Ft(:,totalsteps+1:totalsteps+size(t,2))=ppval(F, t) TODO
            spline_t = spline(t)

            # Increment the step counter
            totalsteps += len(t)

            return spline_t

    def clean_particles_on_carbon(self, model_path, model_suffix, distance_threshold, dimensions):
        tomos_dim = self.load_dimensions(dimensions)
        tomos = self.df.loc[:, 'tomo_id'].unique()
        cleaned_motl = self.__class__.create_empty_motl()

        for t in tomos:
            tomo_str = self.pad_with_zeros(t, 3)
            tm = self.df.loc[self.df['tomo_id'] == t].reset_index(drop=True)

            tdim = tomos_dim.loc[tomos_dim[0] == t, 1:3]
            # pos=tm(8:10,:)+tm(11:13,:);  TODO check that is what it should do
            pos = pd.concat([(tm.loc[:, 'x'] + tm.loc[:, 'shift_x']),
                             (tm.loc[:, 'y'] + tm.loc[:, 'shift_y']),
                             (tm.loc[:, 'z'] + tm.loc[:, 'shift_z'])], axis=1)

            # TODO what is model_suffix supposed to be?
            mod_file_name = os.path.join(model_path, tomo_str, model_suffix)
            if os.path.isfile(f'{mod_file_name}.mod'):
                raise UserInputError(f'File to be generated ({mod_file_name}.mod) already exists in the destination. '
                                     f'Aborting the process to avoid overriding the existing file.')
            else:
                cleaned_motl = pd.concat([cleaned_motl, tm])

            subprocess.run(['model2point', '-object', f'{mod_file_name}.mod', f'{mod_file_name}.txt'])
            coord = pd.read_csv(f'{mod_file_name}.txt', sep='\t')
            carbon_edge = self.spline_sampling(coord.iloc[:, 2:4], 2)

            # TODO adjust based on carbon_edge
            all_points = []
            for z in tdim[:2:2]:
                z_points = carbon_edge
                z_points[:, 2] = z
                all_points.append(z_points)

            rm_idx = []
            for p in len(pos):
                _, npd = dsearchn(all_points, pos.iloc[p, :])
                if npd < distance_threshold:
                    rm_idx.append(p)  # TODO is this reliable? do the idx from pos correspond to idx in tm?
            tm.drop(rm_idx, inplace=True)
            cleaned_motl = pd.concat([cleaned_motl, tm])

        self.df = cleaned_motl
        return self

    def keep_multiple_positions(self, feature_id, min_no_positions, distance_threshold):
        feature = self.get_feature(self.df.columns, feature_id)
        uniq_values = self.df.loc[:, feature].unique()
        new_motl = self.create_empty_motl()

        for value in uniq_values:
            fm = self.df.loc[self.df[feature] == value]

            pos_x = fm.loc[:, 'x'] + fm.loc[:, 'shift_x']
            pos_y = fm.loc[:, 'y'] + fm.loc[:, 'shift_y']
            pos_z = fm.loc[:, 'z'] + fm.loc[:, 'shift_z']

            for i, row in fm.iterrows():
                position = [pos_x[i], pos_y[i], pos_z[i]]  # TODO fix to expected format
                remaining_positions = [pos_x.drop(i), pos_y.drop(i), pos_z.drop(i)]
                temp_dist = geometry_get_pairwise_distance(position, remaining_positions)

                # sp = size((find(temp_dist<distance_threshold)),2);  # TODO again fix, based on the result format
                sp = [x for x in temp_dist if x < distance_threshold]
                if sp:
                    row[15] = sp

            new_motl = pd.concat([new_motl, fm])

        new_motl = new_motl.loc[new_motl['geom6'] >= min_no_positions]  # TODO really should be 'geom6'?
        self.df = new_motl
        return self

    @classmethod
    def recenter_subparticle(cls, motl_list, mask_list, size_list, rotations=None):
        # motl_list = ['SU_motl_gp210_bin4_1.em','SU_motl_gp210_bin4_1.em']
        # mask_list = ['temp_mask.em','temp2_mask.em']
        # size_list = [36 36]
        # rotations = [[0 0 0], [-90 0 0]]
        # Output: Motl instance. To write the result to a file, you can run:
        #           Motl.recenter_subparticle(motl_list, mask_list, size_list).write_to_emfile(outfile_path)

        # Error tolerance - should be done differently and not hard-coded!!! TODO note from original code
        epsilon = 0.00001

        # Generete zero rotations in case they were not specified
        if not rotations:
            rotations = np.zeros((len(mask_list), 3))
        new_motl_df = cls.create_empty_motl()

        for el in range(len(mask_list)):
            mask = cls.load(mask_list[el])
            submotl = cls.load(motl_list[el])

            # TODO should we really write out new mask files here?
            # get path of the masks and output  TODO should be generalized in the emwrite method?
            # output_path = fileparts(output_motl);
            # if(isempty(output_path))
            #     output_path='./';
            # else
            #     output_path=[ output_path '/' ];
            mask_name = os.path.basename(mask_list[el])

            # set sizes of new and old masks
            mask_size = len(mask.df)
            # new_size = repmat(size_list[i], 1, 3)
            new_size = np.repeat(size_list[el], 3)
            old_center = mask_size / 2
            new_center = new_size / 2

            # find center of mask  FIXME maybe approach differently in python
            c_idx = [x for x in mask.df if x > epsilon]  # TODO map on the dataframe using apply
            # TODO not sure there is a python method for that
            i, j, k = ind2sub([len(mask.df), len(mask.df.columns)], c_idx)
            s = [min(i), min(j), min(k)]
            e = [max(i), max(j), max(k)]
            mask_center = (s + e) / 2  # TODO s+e here probably does not do the same asi in matlab

            # get shifts
            shifts = mask_center - old_center;
            shifts2 = round(mask_center - new_center)

            # write out transformed mask to check it's as expeceted  # TODO should be preserved?
            new_mask = tom_red(mask, shifts2, new_size)
            if rotations[:, el] != 0:
                new_mask = tom_rotate(new_mask, rotations[:, el])
            new_mask.write_to_emfile(f'{output_path}/{mask_name}_centered_mask.em')

            # change shifts in the motl accordingly
            # TODO can we use these methods in place of the orignal code, or are there differences?
            submotl.shift_positions(shifts, recenter_particles=True)
            # create quatertions for rotation
            if rotations[:, el] != 0:
                q1 = euler2quat(submotl['phi'], submotl['psi'], submotl['theta'])
                q2 = euler2quat(rotations[:, el])
                mult = quat_mult(q2,q1)
                new_angles = quat2euler(mult)
            # m(17:19,:)=new_angles';

            # add identifier in case of merge motls
            submotl.df['geom_6'] = el

            new_motl_df = pd.concat([new_motl_df, submotl])

        new_motl = cls(new_motl_df)
        new_motl.renumber_particles()

        return new_motl

    @classmethod
    def movement_convergence(cls, motl_base_name, iteration_range, f_std=None):
        """outcome: [ad_all, dist_rmse_all, particle_conv]"""
        c = 1
        i = 1
        m = cls.load(f'{motl_base_name}_{i}.em')
        pc = len(m.df)

        particle_stability = np.zeros((pc, 2))
        particle_stability[:, 0] = m.df.loc[:, 'subtomo_id']

        ad_all = []
        dist_rmse_all = []

        particle_conv = np.zeros((pc, iteration_range))

        if f_std:
            fstd = f_std
        else:
            fstd = 1

        for i in range(2, iteration_range+1):
            m1 = cls.load(f'{motl_base_name}_{i-1}.em')
            m2 = cls.load(f'{motl_base_name}_{i}.em')

            ad = angular_distance(m1.df.loc[:, ['phi', 'psi', 'theta']], m2.df.loc[:, ['phi', 'psi', 'theta']])
            dist_rmse = geometry_point_distance(m1.df.loc[:, ['shift_x', 'shift_y', 'shift_z']], m2.df.loc[:, ['shift_x', 'shift_y', 'shift_z']])

            ad_m = np.mean(ad)
            ad_std = np.std(ad)

            dr_m = np.mean(dist_rmse)
            dr_std = np.std(dist_rmse)

            # TODO: REWORK
            #ad_pidx=m2(4,ad>ad_m+fstd*ad_std);
            ad_pidx_arr = m2.df['subtomo_id'].values
            ad_pidx_bool = ad > ad_m + f_std * ad_std
            ad_pidx = np.column_stack((ad_pidx_arr, ad_pidx_bool))
            #assuming that each part is an array
            #dr_pidx=m2(4,dist_rmse>dr_m+fstd*dr_std);
            dr_pidx_arr = m2.df['subtomo_id'].values
            dr_pidx_bool = dist_rmse > dr_m + fstd * dr_std
            dr_pidx = np.column_stack((dr_pidx_arr, dr_pidx_bool))

            #f_pidx=ad_pidx(ismember(ad_pidx,dr_pidx));
            f_pidx = ad_pidx[np.isin(ad_pidx, dr_pidx)]

            #particle_conv(i-1,ismember(m2(4,:),f_pidx))=1;
            particle_conv[np.isin(m2.df[:, 'subtomo_id'], f_pidx), i-1] = 1

            ad_all.append(ad)
            dist_rmse_all.append(dist_rmse)
        
        sp = np.sum(particle_conv, 1)
        p = (sp > np.mean(sp) + np.std(sp)).nonzero()

        good_motl = m2
        good_motl.df[p, :] = []
        bad_motl = m2.df[p, :]

        # TODO: check it's correct this way
        emfile.write(good_motl, [f'{motl_base_name}_good_{iteration_range}.em'])
        emfile.write(bad_motl, [f'{motl_base_name}_bad_{iteration_range}.em'])

        # TODO: REWORK the plots
        plt.hist(sp)
        plt.show()    
        plt.plot(np.mean(ad_all, axis=1))
        plt.show()

        return ad_all, dist_rmse_all, particle_conv

    @classmethod
    def class_convergence(cls, motl_base_name, iteration_range):
        """outcome: [ overall_changes, particle_stability, m_idx, particle_changes]"""

        c = 1
        i = 1

        m = cls.load(f'{motl_base_name}_{i}.em')
        pc = m.shape[0]

        particle_stability = np.zeros((pc, 2))
        particle_stability[:, 0] = m.loc[:, 'subtomo_id']

        particle_changes = []
        overall_changes = []

        for i in range(2, iteration_range+1):
            m1 = cls.load(f'{motl_base_name}_{iteration_range[i-1]}.em')
            m2 = cls.load(f'{motl_base_name}_{iteration_range[i]}.em')

            ch_idx = m1.score.ne(m2.score)
            overall_changes.append(sum(ch_idx))

            for i in range(particle_stability.shape[0]):
                particle_stability[i, 1] += ch_idx[i]
            
            if c != 1:
                ch_particles = sum(np.isin(m2[ch_idx, 'subtomo_id'], m_idx))
                particle_changes.append(ch_particles/(pc * 100))
            
            # TODO: edit this part and check the fuctionality
            m_idx = m2.loc[ch_idx, 'subtomo_id']
            c += 1

        # TODO: REWORK the plots
        plt.plot(overall_changes)
        plt.show()
        plt.plot(particle_changes)
        plt.show()
        plt.hist(particle_stability[:, 1])
        plt.show()

        return overall_changes, particle_stability, m_idx, particle_changes

    def split_particles_in_assymetric_units(self, motl, symmetry, particle_x_shift, start_offset=None, output_name=None):
        """outcome: new_motl"""
        if start_offset:
            start_offset = start_offset
        else:
            start_offset = 0
        
        phi_angles = np.array([])
        inplane_step = 360 / symmetry

        for a in range(0, 360, int(inplane_step)):
            phi_angles = np.append(phi_angles, a+start_offset)

        # make up vectors
        starting_vector = np.array([particle_x_shift, 0, 0])
        rho = np.sqrt(starting_vector[0]**2 + starting_vector[1]**2)
        the = np.arctan2(starting_vector[1], starting_vector[0])
        z = starting_vector[2]

        rot_rho = repmat(rho, symmetry, 1)
        rot_rho = np.array([x + y for x,y in zip(rot_rho, phi_angles * np.pi / 180)])
        rep_the = repmat(the, symmetry, 1)
        rep_z = repmat(z, symmetry, 1)

        # https://stackoverflow.com/questions/20924085/python-conversion-between-coordinates
        # [center_shift(:, 1), center_shift(:, 2), center_shift(:, 3)] = pol2cart([0;0.785398163397448;1.570796326794897;2.356194490192345;3.141592653589793;3.926990816987241;4.712388980384690;5.497787143782138], repmat(10,8,1), repmat(0,8,1));
        center_shift = np.zeros([rot_rho.shape[0], 3])
        for i in range(center_shift.shape[0]):
            center_shift[i][0] = rot_rho[i] * np.cos(rep_the[i])
            center_shift[i][1] = rot_rho[i] * np.sin(rep_the[i])
            center_shift[i][2] = rep_z[i]

        #TODO: check the outcomes, vars and functionality from here 
        motl = self.__class__.load(motl)
        new_motl = self.create_empty_motl()

        for i in range(len(motl)):

            nm = repmat(motl.df[i, :], 1, symmetry)
            nm[:, 15] = repmat(motl.df(i, 3), 1, symmetry)
            nm.df.loc[:, 2] = [i for i in range(1, symmetry+1)] 

            phi = motl.df[i, 16]
            psi = motl.df[i, 17]
            the = motl.df[i, 18]

            xshift = motl.df[i, 10]
            yshift = motl.df[i, 11]
            zshift = motl.df[i, 12]

            shift = repmat(np.array([xshift, yshift, zshift]), symmetry, 1)

            #create centers
            #new_centers = tom_pointrotate(center_shift, phi, psi, the) + shift
            euler_angles = np.array([[phi, the, psi]])
            orientations = rot.from_euler(seq='zxz', angles=euler_angles, degrees=True)
            new_centers = orientations.apply(shift)

            new_shifts = np.array([x + y for x,y in zip(new_centers, new_centers.round)]) # np.ndarray.round method
            nm.df[:, 7:10] = np.array([x + y for x,y in zip(new_centers.round, nm.df[:, 7:10])])
            nm.df[:, 10:13] = new_shifts

            # phi angle correction 
            nm.df[:, 16] = np.array([x + y for x, y in zip(repmat(phi, 1, symmetry), phi_angles)])

            new_motl.df.concat(pd.DataFrame(nm.df.reshape(1,-1), columns=list(new_motl)), ignore_index=True)


        new_motl.df.loc[:, 3] = [i for i in range(len(new_motl.df))]

        if output_name:
            self.__class__.write(new_motl, output_name) #replace function

        self.df = new_motl
        return self    

    def sg_create_multiref(input_motl, number_of_classes, class_occupancy, number_of_runs, output_motl_name=None):
        if output_motl_name:
            output_motl = output_motl_name
        else:
            output_motl = f'{input_motl[:-2]}star'
        
        sg_motl_av3_to_stopgap(input_motl, output_motl)

        iteration_number = string_get_numbers(output_motl, 'last')

        # create random subsets for initial averages
        for i in range(number_of_runs):
            sg_motl_initialize_random_subset_classes(output_motl,
                                                    number_of_classes,
                                                    class_occupancy,
                                                    f'{output_motl[:-6]}ref_mr{i}_{iteration_number}.star')

        # random split into classes
        sg_motl_apply_random_classes(output_motl,
                                    number_of_classes,
                                    output_motl) # ??? TODO: duplicate in the original script! repair
