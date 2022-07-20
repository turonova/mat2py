import emfile
import numpy as np
import os
import pandas as pd
import subprocess

from exceptions import UserInputError


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
                                              'geom5', 'geom6', 'phi', 'psi', 'theta', 'class'])
        return empty_motl_df

    @staticmethod  # TODO move to different module
    def pad_with_zeros(number, total_digits):
        # Creates a string of the length specified by total_digits, containing a given number and fills the rest
        # (at the beginning) with zeros, i.e. from the input parameters 3 and 6 it creates 000003)
        #
        # Input:  number - number to be padded with zero
        #         total_digits - final length of the output string
        # Output: string of length specified by total_digits and containing the input number at the very end

        padded_str = ''  # TODO do more efficiently ?
        zeros = total_digits - len(str(number))
        for _ in zeros:
            padded_str += '0'
        padded_str += str(number)

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

    @classmethod
    def read_from_emfile(cls, emfile_path):
        header, parsed_emfile = emfile.read(emfile_path)
        if not len(parsed_emfile[0][0]) == 20:
            raise UserInputError(
                f'Provided file contains {len(parsed_emfile[0][0])} columns, while 20 columns are expected.')

        # TODO do we really want everything as float? (taken from the original parsed file)
        motl = pd.DataFrame(data=parsed_emfile[0], dtype=float,
                            columns=['score', 'geom1', 'geom2', 'subtomo_id', 'tomo_id', 'object_id',
                                     'subtomo_mean', 'x', 'y', 'z', 'shift_x', 'shift_y', 'shift_z', 'geom4',
                                     'geom5', 'geom6', 'phi', 'psi', 'theta', 'class'])

        # TODO do we want to keep these? (from the original emmotl.py)
        # motl['subtomo_id'] = np.arange(1, number_of_particles + 1, 1)
        # motl['class'] = 1
        # round coord
        # motl[['x', 'y', 'z']] = np.round(coordinates.values)
        # get shifts
        # motl[['shift_x', 'shift_y', 'shift_z']] = coordinates.values - np.round(coordinates.values)

        return cls(motl, header)

    @classmethod
    def load(cls, input_motl):
        # TODO allow to load correct motl/s if there are one or more corrupted?
        # Input: Load one or more emfiles (as a list), or already initialized instances of the Motl class
        #        E.g. `Motl.load([cryo1.em, cryo2.em, motl_instance1])`
        # Output: Returns one instance, or a list of instances if multiple inputs are provided

        loaded = list()
        motls = [input_motl] if not isinstance(input_motl, list) else input_motl
        if len(motls) == 0:
            raise UserInputError('At least one em file or a Motl instance must be provided.')
        else:
            for motl in motls:
                # TODO can emfile have any other suffix?
                if isinstance(motl, str) and os.path.isfile(motl) and (os.path.splitext(motl)[-1] == '.em'):
                    new_motl = cls.read_from_emfile(motl)

                elif isinstance(motl, cls):
                    new_motl = motl
                else:
                    # TODO or will it still be possible to receive the motl in form of a pure matrix?
                    raise UserInputError(f'Unknown input type: {motl}. '
                                         f'Input needs to be either an em file (.em), or an instance of the Motl class.')

                if not np.array_equal(new_motl.df.columns, cls.create_empty_motl().columns):
                    raise UserInputError(f'Provided Motl object {motl} seems to be corrupted and can not be loaded.')
                else:
                    loaded.append(new_motl)

            if len(loaded) == 1:
                loaded = loaded[0]

        return loaded

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

    @classmethod
    def get_particle_intersection(cls, motl1, motl2):
        # TODO currently keeps rows from motl1, is that ok? (are the remaining values of the row identical to motl2?)
        m1, m2 = cls.load([motl1, motl2])
        m2_values = m2.df.loc[:, 'subtomo_id'].values
        intersected = cls.create_empty_motl()

        for value in m2_values:
            submotl = m1.df.loc[m1.df['subtomo_id'] == value]
            intersected = pd.concat([intersected, submotl])

        return cls(intersected)

    def write_to_emfile(self, outfile_path):
        motl_array = self.df.to_numpy()
        motl_array = motl_array.reshape((1, motl_array.shape[0], motl_array.shape[1]))
        # FIXME fails on writing back the header
        emfile.write(outfile_path, motl_array, self.header, overwrite=True)

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

    def update_coordinates(self):  # TODO add tests
        shifted_x = self.df.loc[:, 'x'] + self.df.loc[:, 'shift_x']
        shifted_y = self.df.loc[:, 'y'] + self.df.loc[:, 'shift_y']
        shifted_z = self.df.loc[:, 'z'] + self.df.loc[:, 'shift_z']

        self.df.loc[:, 'x'] = round(shifted_x)
        self.df.loc[:, 'y'] = round(shifted_y)
        self.df.loc[:, 'z'] = round(shifted_z)
        self.df.loc[:, 'shift_x'] = shifted_x - self.df.loc[:, 'x']
        self.df.loc[:, 'shift_y'] = shifted_y - self.df.loc[:, 'y']
        self.df.loc[:, 'shift_z'] = shifted_z - self.df.loc[:, 'z']

        return self

    def tomo_subset(self, tomo_numbers, renumber_particles=False):  # TODO add tests
        # Updates motl to contain only particles from tomograms specified by tomo numbers
        # Input: tomo_numbers - list of selected tomogram numbers to be included
        #        renumber_particles - renumber from 1 to the size of the new motl if True

        new_motl = self.__class__.create_empty_motl()
        for i in tomo_numbers:
            df_i = self.df.loc[self.df['tomo_id'] == i]
            new_motl = pd.concat([new_motl, df_i])
        self.df = new_motl

        if renumber_particles: self.renumber_particles()
        return self

    def renumber_particles(self):  # TODO add tests
        # new_motl(4,:)=1: size(new_motl, 2);
        self.df.loc[:, 'subtomo_id'] = list(range(1, len(self.df)+1))
        return self

    ############################
    # PARTIALLY FINISHED METHODS

    def write_to_model_file(self, feature_id, output_base, point_size, binning=None):
        feature = self.get_feature(self.df.columns, feature_id)
        uniq_values = self.df.loc[:, feature].unique()
        output_base = f'{output_base}_{feature}_'

        if binning:
            bin = binning
        else:
            bin = 1

        for value in uniq_values:
            fm = self.df.loc[self.df[feature] == value]
            tomo_str = self.pad_with_zeros(value, 3)
            output_txt = f'{output_base}{tomo_str}_model.txt'
            output_mod = f'{output_base}{tomo_str}.mod'

            pos_x = (fm.loc[:, 'x'] + fm.loc[:, 'shift_x']) * bin
            pos_y = (fm.loc[:, 'y'] + fm.loc[:, 'shift_y']) * bin
            pos_z = (fm.loc[:, 'z'] + fm.loc[:, 'shift_z']) * bin

            # pos = [ fm(20,:)' repmat(1,size(fm,2),1) pos]; TODO

            pos_df = pd.concat([pos_x, pos_y, pos_z], axis=1)
            pos_df.to_csv(output_txt, sep='\t')

            # Create model files from the coordinates
            # system(['point2model -sc -sphere ' num2str(point_size) ' ' output_txt ' ' output_mod]);
            subprocess.run(['point2model', '-sc', '-sphere', str(point_size), output_txt, output_mod])

    @staticmethod
    def batch_stopgap2em(motl_base_name, iter_no):
        for i in range(iter_no):
            motl_name = f'{motl_base_name}_{str(i)}'
            sg_motl_stopgap_to_av3(f'{motl_name}.star', f'{motl_name}.em')

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

        for t in tomos:
            tm = self.df.loc[self.df['tomo_id'] == t]
            features = self.df.loc[:, feature].unique()

            for f in features:
                fm = tm.loc[tm[feature] == f]
                h = histogram(fm.loc[:, 'score'], hbin)
                bn = math_otsu_threshold(h.BinCounts)
                cc_t = h.BinEdges(bn)
                fm = fm.loc[fm['score'] >= cc_t]

                cleaned_motl = pd.concat([cleaned_motl, fm])

        self.df = cleaned_motl
        return self

    def shift_positions(self, shift, recenter_particles=False):
        # Shifts positions of all subtomgoram in the motl in the direction given by subtomos' rotations

        def shift_coords(row):
            rshifts = tom_pointrotate(shift, row['phi'], row['psi'], row['theta'])
            # rshifts = rshifts';  TODO what ' does do?
            row['shift_x'] = row['shift_x'] + rshifts
            row['shift_y'] = row['shift_y'] + rshifts
            row['shift_z'] = row['shift_z'] + rshifts
            return row

        self.df = self.df.apply(shift_coords, axis=1)
        if recenter_particles: self.update_coordinates()
        return self

    def clean_particles_on_carbon(self, model_path, model_suffix, distance_threshold, dimensions, renumber_particles=False):
        if os.path.isfile(dimensions):
            # TODO what is the commonly used delimeter? The matlab dlmread detected the delimeter automatically
            # Does it have a header, index, or any other specificities?
            tomos_dim = pd.read_csv(dimensions)
        else:  # TODO where does the raw matrix come from?
            tomos_dim = pd.DataFrame(dimensions)

        model_path = string_path_complete(model_path)
        tomos = self.df.loc[:, 'tomo_id'].unique()
        cleaned_motl = self.__class__.create_empty_motl()

        for t in tomos:
            tomo_str = self.pad_with_zeros(t, 3)
            tm = self.df.loc[self.df['tomo_id'] == t]

            tdim = tomos_dim.loc[tomos_dim[0] == t, ['geom1', 'geom2', 'subtomo_id']]
            pos = tm.loc[:, ['x', 'y', 'z']] + tm.loc[:, ['shift_x', 'shift_y', 'shift_z']]

            mod_file_name = os.path.join(model_path, tomo_str, model_suffix, '.mod')

            # if(exist(mod_file_name, 'file') ~= 2)
            if os.path.isfile(mod_file_name):
                cleaned_motl = pd.concat([cleaned_motl, tm])

            subprocess.run(['model2point', '-object', mod_file_name, f'{mod_file_name}.txt'])
                           # stdout=out_file, stderr=out_err, check=True)
            coord = pd.read_csv(f'{mod_file_name}.txt')  # TODO dlmread
            # carbon_edge=geometry_spline_sampling(coord(:,3:5)',2);
            carbon_edge = geometry_spline_sampling(coord.iloc[2:4, :], 2)

            all_points = []
            # for z=1:2:tdim(3) TODO
                # z_points=carbon_edge';
                # z_points(:,3)=z;
                # all_points=[all_points; z_points];

            # for p=1:size(pos,2) TODO remove from df
                # [np npd]=dsearchn(all_points,pos(:,p)');
                # if npd < distance_threshold:
                #     rm_idx=[rm_idx p];
            # tm(:,rm_idx)=[];

            cleaned_motl = pd.concat([cleaned_motl, tm])

        if renumber_particles: self.renumber_particles()

        return self

    def split_by_feature(self, feature_id, write_out=False, output_prefix=None, feature_desc_id=None):
        # Split motl by uniq values of a selected feature
        # Inputs:   feature_id - column name or index of the feature based on witch the motl will be split
        #           write: save all the resulting Motl instances into separate files if True
        #           output_prefix:
        #           feature_desc_id:
        # Output: list of Motl instances, each containing only rows with one unique value of the given feature

        feature = self.get_feature(self.df.columns, feature_id)
        uniq_values = self.df.loc[:, feature].unique()
        motls = list()

        for value in uniq_values:
            submotl = self.__class__(self.df.loc[self.df[feature] == value])
            motls.append(submotl)

            if write_out:  # TODO really keep it here, or make a class method to support batch export?
                if feature_desc_id:  # TODO what's that supposed to do?
                    out_name = output_prefix
                    # out_name=output_prefix;
                    # for d=feature_desc_id
                    #     out_name=[out_name '_' num2str(nm(d,1))];
                    # out_name=[out_name  '.em'];
                else:
                    out_name = f'{output_prefix}_{str(value)}.em'
                submotl.write_to_emfile(out_name)

        return motls

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
