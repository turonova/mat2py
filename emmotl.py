import emfile
import numpy as np
import os
import pandas as pd
import subprocess


class Motl:
    # Motl module example usage
    #
    # Initialize a Motl instance from an emfile
    #   `motl = Motl.load(’path_to_em_file’)`
    # Run clean_by_otsu and write the result to a new file
    #   `motl.clean_by_otsu(4, histogram_bin=20).write_to_emfile('path_to_output_em_file')`
    # Run class_consistency on multiple Motl instances
    #   `motl_intersect, motl_bad, cl_overlap = Motl.class_consistency(Motl.load('emfile1', 'emfile2', 'emfile3'))`

    def __init__(self, motl_df):
        self.df = motl_df

    @staticmethod
    def create_empty_motl():
        empty_motl_df = pd.DataFrame(columns = ['score', 'geom1', 'geom2', 'subtomo_id', 'tomo_id', 'object_id',
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

    @classmethod
    def merge_and_renumber(cls, motl_list):
        merged_df = cls.create_empty_motl()
        feature_add = 0

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

    @staticmethod
    def batch_stopgap2em(motl_base_name, iter_no):
        for i in range(iter_no):
            motl_name = f'{motl_base_name}_{str(i)}'
            sg_motl_stopgap_to_av3(f'{motl_name}.star', f'{motl_name}.em')

    @classmethod
    def motl_class_consistency(cls, motl1, motl2):
        # to be defined
        pass
    def clean_by_otsu(self, feature, histogram_bin=None):
        # Cleans motl by Otsu threshold (based on CC values)
        # feature: a feature by which the subtomograms will be grouped together for cleaning;
    	# corresponds to the rows in motl; 4 to group by tomogram, 5 to clean by a particle (e.g. VLP, virion)
		# histogram_bin: how fine to split the histogram. Default is 30 for feature 5 and 40 for feature 4;
		# for smaller number of subtomograms per feature the number should be lower

        tomos = self.df.loc[:, 'tomo_id'].unique()
        cleaned_motl = self.__class__.create_empty_motl()

        if histogram_bin:
            hbin = histogram_bin
        else:
            if feature == 5:
                hbin = 40
            elif feature == 6:
                hbin = 30

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

    def shift_positions(self, shift, recenter_particles=False):
        # Shifts positions of all subtomgoram in the motl in the direction given by subtomos' rotations

        def shift_coords(row):
            rshifts = tom_pointrotate(shift, row[16], row[17], row[18])
            # rshifts = rshifts';  TODO what ' does do?
            row[10] = row[10] + rshifts
            row[11] = row[11] + rshifts
            row[12] = row[12] + rshifts
            return row

        self.df = self.df.apply(shift_coords, axis=1)
        if recenter_particles: self.update_coordinates()

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

            # tdim=tomos_dim(tomos_dim(:,1)==t,2:4);
            tdim = tomos_dim.iloc[1:3, tomos_dim[0] == t]
            pos = tm.iloc[:, 7:9] + tm.iloc[:, 10:12]

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

    def remove_feature(self, feature, feature_values):
       # Removes particles based on their feature (i.e. tomo number)
       # Inputs: feature - column name or index based on which the particles will be removed (i.e. 4 for tomogram number)
       #         feature_values - list of values to be removed
       #         output_motl_name - name of the new motl; if empty the motl will not be written out
       # Usage: motl.remove_feature(4, [3, 7, 8]) - removes all particles from tomograms number 3, 7, and 8

        if isinstance(feature, int):
            feature = self.df.columns[feature]
        for value in feature_values:
            self.df = self.df.loc[self.df[feature] != value]

    def update_coordinates(self):
        shifted_x = self.df.iloc[:, 7] + self.df.iloc[:, 10]
        shifted_y = self.df.iloc[:, 8] + self.df.iloc[:, 11]
        shifted_z = self.df.iloc[:, 9] + self.df.iloc[:, 12]

        self.df.iloc[:, 7] = round(shifted_x)
        self.df.iloc[:, 8] = round(shifted_y)
        self.df.iloc[:, 9] = round(shifted_z)
        self.df.iloc[:, 10] = shifted_x - self.df.iloc[:, 7]
        self.df.iloc[:, 11] = shifted_y - self.df.iloc[:, 8]
        self.df.iloc[:, 12] = shifted_z - self.df.iloc[:, 9]

    def tomo_subset(self, tomo_numbers, renumber_particles=False):
        # Updates motl to contain only particles from tomograms specified by tomo numbers
        # Input: tomo_numbers - list of selected tomogram numbers to be included
        #        renumber_particles - renumber from 1 to the size of the new motl if True

        new_motl = self.__class__.create_empty_motl()
        for i in tomo_numbers:
            df_i = self.df.loc[self.df['tomo_id'] == i]
            new_motl = pd.concat([new_motl, df_i])
        self.df = new_motl

        if renumber_particles: self.renumber_particles()

    def renumber_particles(self):
        # new_motl(4,:)=1: size(new_motl, 2);
        self.df.loc[:, 'subtomo_id'] = list(range(1, len(self.df)+1))

    def split_by_feature(self, feature, write_out=False, output_prefix=None, feature_desc_id=None):
        # Split motl by uniq values of a selected feature
        # Inputs:   feature - column name or index of the feature based on witch the motl will be split
        #           write: save all the resulting Motl instances into separate files if True
        #           output_prefix:
        #           feature_desc_id:
        # Output: list of Motl instances, each containing only rows with one unique value of the given feature

        if isinstance(feature, int): feature = self.df.columns[feature]
        uniq_values = self.df.loc[:, feature].unique()
        motls = list()

        for value in uniq_values:
            submotl = self.__class__(self.df.loc[self.df[feature] == value])
            motls.append(submotl)

            if write_out:  # TODO really keep it here, or make a class method to support batch export?
                if feature_desc_id: # TODO what's that supposed to do?
                    out_name = output_prefix
                    # out_name=output_prefix;
                    # for d=feature_desc_id
                    #     out_name=[out_name '_' num2str(nm(d,1))];
                    # out_name=[out_name  '.em'];
                else:
                    out_name = f'{output_prefix}_{str(value)}.em'
                submotl.write_to_emfile(out_name)

        return motls

    def keep_multiple_positions(self, feature, min_no_positions, distance_threshold):
        if isinstance(feature, int): feature = self.df.columns[feature]
        uniq_values = self.df.loc[:, feature].unique()
        new_motl = self.create_empty_motl()

        for value in uniq_values:
            fm = self.df.loc[self.df[feature] == value]

            pos_x = fm.iloc[:, 7] + fm.iloc[:, 10]
            pos_y = fm.iloc[:, 8] + fm.iloc[:, 11]
            pos_z = fm.iloc[:, 9] + fm.iloc[:, 12]

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


