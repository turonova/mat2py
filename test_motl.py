import emfile
import numpy as np
import os
import pandas as pd
import pytest
import starfile

from emmotl import Motl
from exceptions import UserInputError


@pytest.fixture
def motl():
    motl = Motl.read_from_emfile('./example_files/au_1.em')
    return motl


@pytest.fixture
def sg():
    sg = starfile.read('./example_files/au_1.star')
    return sg


@pytest.mark.parametrize('feature_id', ['score', 5])
def test_get_feature_existing(motl, feature_id):
    feature = motl.get_feature(feature_id)
    assert isinstance(feature, str) and feature in motl.df.columns


@pytest.mark.parametrize('feature_id', ['missing', 99])
def test_get_feature_not_existing(motl, feature_id):
    with pytest.raises(UserInputError):
        motl.get_feature(feature_id)


@pytest.mark.parametrize('feature', ['score', 0])
def test_remove_feature_existing(motl, feature):
    assert float('0.063319') not in motl.remove_feature(feature, 0.063319).df.loc[:, 'score'].values


def check_emmotl(motl):
    # TODO check other critical aspects of the motl ?
    assert isinstance(motl.header, dict)
    assert np.array_equal(motl.df.columns, ['score', 'geom1', 'geom2', 'subtomo_id', 'tomo_id', 'object_id',
                                            'subtomo_mean', 'x', 'y', 'z', 'shift_x', 'shift_y', 'shift_z',
                                            'geom4', 'geom5', 'geom6', 'phi', 'psi', 'theta', 'class'])
    assert all(dt == 'float64' for dt in motl.df.dtypes.values)


@pytest.mark.parametrize('m', ['./example_files/test/au_1.em', './example_files/test/au_2.em'])
def test_read_from_emfile(m):
    motl = Motl.read_from_emfile(m)
    check_emmotl(motl)


@pytest.mark.parametrize('m', ['./example_files/test/col_missing.em', './example_files/test/extra_col.em'])
# TODO did not manage to write out corrupted em file '/test/na_values.em', '/test/bad_values.em'
def test_read_from_emfile_wrong(m):
    with pytest.raises(UserInputError):
        Motl.read_from_emfile(m)


@pytest.mark.parametrize('m', ['./example_files/test/au_1.em',
                               ['./example_files/test/au_1.em', './example_files/test/au_2.em']])
def test_load(m):
    loaded = Motl.load(m)
    if isinstance(m, list):
        assert len(m) == len(loaded)
        assert [isinstance(l, Motl) for l in loaded]
        for l in loaded:
            assert np.array_equal(l.df.columns, ['score', 'geom1', 'geom2', 'subtomo_id', 'tomo_id', 'object_id',
                                                 'subtomo_mean', 'x', 'y', 'z', 'shift_x', 'shift_y', 'shift_z',
                                                 'geom4', 'geom5', 'geom6', 'phi', 'psi', 'theta', 'class'])
    else:
        assert isinstance(loaded, Motl)
        assert np.array_equal(loaded.df.columns, ['score', 'geom1', 'geom2', 'subtomo_id', 'tomo_id', 'object_id',
                                                  'subtomo_mean', 'x', 'y', 'z', 'shift_x', 'shift_y', 'shift_z',
                                                  'geom4', 'geom5', 'geom6', 'phi', 'psi', 'theta', 'class'])


@pytest.mark.parametrize('m', ['./example_files/test/au_1.txt', './example_files/test/au_1', '', (), [],
                               'not_a_file_path', ['./example_files/test/au_1.em', './example_files/au_1.txt']])
def test_load_wrong(m):
    with pytest.raises(UserInputError):
        Motl.load(m)


@pytest.mark.parametrize('motl_list', [['./example_files/test/au_1.em', './example_files/test/au_2.em'],
                                       ['./example_files/test/au_1.em', './example_files/test/au_1.em']])
def test_merge_and_renumber(motl_list):
    # TODO how should we check the 'object_id' is numbered correctly ?
    combined_len = 0
    for m in motl_list:
        combined_len += len(Motl.load(m).df)
    merged_motl = Motl.merge_and_renumber(motl_list)
    assert len(merged_motl.df) == combined_len


@pytest.mark.parametrize('motl_list', ['./example_files/test/au_1.em', [], (), 'not_a_list', 42,
                                       ['./example_files/test/au_1.em', None]])
def test_merge_and_renumber_wrong(motl_list):
    with pytest.raises(UserInputError):
        Motl.merge_and_renumber(motl_list)


@pytest.mark.parametrize('m1, m2', [('./example_files/test/au_1.em', './example_files/test/au_2.em'),
                                    ('./example_files/test/au_1.em', './example_files/test/au_1.em')])
def test_get_particle_intersection(m1, m2):
    intersected = Motl.get_particle_intersection(m1, m2)
    m1_values = Motl.load(m1).df.loc[:, 'subtomo_id'].values
    m2_values = Motl.load(m2).df.loc[:, 'subtomo_id'].values
    assert all((value in m1_values) and (value in m2_values) for value in intersected.df.loc[:, 'subtomo_id'].values)


@pytest.mark.parametrize('m1, m2', [('./example_files/test/au_1.em', None), ('./example_files/test/au_1.em', 'a'),
                                    (None, None), ('./example_files/test/au_1.txt', './example_files/test/au_2.tf')])
def test_get_particle_intersection_wrong(m1, m2):
    with pytest.raises(UserInputError):
        Motl.get_particle_intersection(m1, m2)


def test_stopgap_to_av3(sg):
    motl = Motl.stopgap_to_av3(sg)
    check_emmotl(motl)


@pytest.mark.parametrize('basename, iterations', [('./example_files/test/star/allmotl_sp_cl1', 5)])
def test_batch_stopgap2em(basename, iterations):
    converted = Motl.batch_stopgap2em(basename, iterations)
    assert len(converted) == iterations
    assert all(os.path.isfile(file) for file in converted)


@pytest.mark.parametrize('f', [0, 5, 'subtomo_id', 'geom2'])
def test_split_by_feature(motl, f):
    motls = motl.split_by_feature(f)
    for motl in motls:
        check_emmotl(motl)


@pytest.mark.parametrize('m1, m2', [('./example_files/test/au_1.em', './example_files/test/au_2.em')])
def test_class_consistency(m1, m2):  # TODO
    intersect, bad, clo = Motl.class_consistency(m1, m2)


@pytest.mark.parametrize('m1, m2, ref',
                         [('./example_files/test/intersection/allmotl_sp_cl1_1.em', './example_files/test/intersection/allmotl_sp_cl1_1.em', './example_files/test/intersection/intersected_equal.em'),
                          ('./example_files/test/intersection/allmotl_sp_cl1_1.em', './example_files/test/intersection/allmotl_sp_cl1_2.em', './example_files/test/intersection/intersected_same.em'),
                          ('./example_files/test/intersection/allmotl_sp_cl1_1.em', './example_files/test/intersection/allmotl_sp_cl1_1_edited.em', './example_files/test/intersection/intersected_dif.em'),
                          ('./example_files/test/intersection/allmotl_sp_cl1_1.em', './example_files/test/intersection/au_1.em', 'empty')])
def test_get_particle_intersection(m1, m2, ref):
    intersected = Motl.get_particle_intersection(m1, m2)
    if os.path.isfile(ref):
        ref_df = Motl.load(ref).df
        assert intersected.df.equals(ref_df)
    elif ref == 'empty':
        assert len(intersected.df) == 0


@pytest.mark.parametrize('m, feature, hist, ref',
                         [('./example_files/test/otsu/allmotl_sp_cl1_1.em', 'tomo_id', None, './example_files/test/otsu/cleaned_1.em')])
                         # [('./example_files/test/otsu/allmotl_sp_cl1_1.em', 'tomo_id', None, './example_files/test/otsu/cleaned_1.em'),
                         #  ('./example_files/test/otsu/allmotl_sp_cl1_1.em', 'object_id', None, './example_files/test/otsu/cleaned_2.em'),
                         #  ('./example_files/test/otsu/allmotl_sp_cl1_1.em', 'tomo_id', 30, './example_files/test/otsu/cleaned_3.em'),
                         #  ('./example_files/test/otsu/allmotl_sp_cl1_1.em', 'object_id', 20, './example_files/test/otsu/cleaned_4.em'),
                         #  ('./example_files/test/otsu/au_1.em', 'tomo_id', None, './example_files/test/otsu/cleaned_5.em')
                         #  ])
def test_clean_by_otsu(m, feature, hist, ref):
    motl, ref_motl = Motl.load([m, ref])
    motl.clean_by_otsu(feature, hist)

    different_rows = motl.df.merge(ref_motl.df, how='outer', indicator=True).loc[lambda x: x['_merge'] != 'both']
    print(different_rows)
    # assert motl.df.equals(ref_motl.df)


@pytest.mark.parametrize('m, feature_id, output_base, point_size, binning',  # TODO
                         [('./example_files/mod/allmotl_sp_cl1_1.em', 4, './example_files/test/mod/testmod', None, None)])
def test_write_to_model_file(m, feature_id, output_base, point_size, binning):
    motl = Motl.load(m)
    motl.write_to_model_file(feature_id, output_base, point_size, binning)


@pytest.mark.parametrize('m, ref', [
    ('./example_files/test/recenter/allmotl_sp_cl1_1.em', './example_files/test/recenter/ref1.em'),
    ('./example_files/test/recenter/allmotl_sp_cl1_2.em', './example_files/test/recenter/ref2.em')])
def test_recenter_particles(m, ref):
    motl, ref_motl = Motl.load([m, ref])
    motl.df = Motl.recenter_particles(motl.df)
    assert motl.df.equals(ref_motl.df)


@pytest.mark.parametrize('m, dimensions, boundary_type, box_size, recenter, ref', [
    ('./example_files/test/outofbounds/allmotl_sp_cl1_1.em', './example_files/test/outofbounds/dimensions.txt', 'whole', -1000, True, './example_files/test/outofbounds/ref1.em'),
    ('./example_files/test/outofbounds/allmotl_sp_cl1_1.em', './example_files/test/outofbounds/dimensions.txt', 'whole', -1000, False, './example_files/test/outofbounds/ref2.em'),
    ('./example_files/test/outofbounds/allmotl_sp_cl1_1.em', './example_files/test/outofbounds/dimensions.txt', 'center', None, True, './example_files/test/outofbounds/ref3.em'),
    ('./example_files/test/outofbounds/allmotl_sp_cl1_1_edit.em', './example_files/test/outofbounds/dimensions.txt', 'whole', -1000, True, './example_files/test/outofbounds/ref4.em'),
    ('./example_files/test/outofbounds/allmotl_sp_cl1_1_edit.em', './example_files/test/outofbounds/dimensions.txt', 'center', None, True, './example_files/test/outofbounds/ref5.em'),
    ('./example_files/test/outofbounds/allmotl_sp_cl1_1_edit.em', './example_files/test/outofbounds/dimensions.txt', 'center', None, False, './example_files/test/outofbounds/ref6.em')])
def test_remove_out_of_bounds_particles(m, dimensions, boundary_type, box_size, recenter, ref):
    motl, ref_motl = Motl.load([m, ref])
    motl.remove_out_of_bounds_particles(dimensions, boundary_type, box_size, recenter)
    assert motl.df.equals(ref_motl.df)


@pytest.mark.parametrize('m, feature, min_no_positions, distance_threshold, ref', [
    ('./example_files/test/keep_multiple/allmotl_sp_cl1_1.em', 'object_id', 2, 1, './example_files/test/keep_multiple/ref1.em'),
    ('./example_files/test/keep_multiple/allmotl_sp_cl1_1.em', 'tomo_id', 2, 1, './example_files/test/keep_multiple/ref2.em'),
    ('./example_files/test/keep_multiple/allmotl_sp_cl1_5.em', 'tomo_id', 2, 2, './example_files/test/keep_multiple/ref3.em'),
    ('./example_files/test/keep_multiple/allmotl_sp_cl1_5.em', 'object_id', 2, 1, './example_files/test/keep_multiple/ref4.em')])
def test_keep_multiple_positions(m, feature, min_no_positions, distance_threshold, ref):
    motl, ref_motl = Motl.load([m, ref])
    motl.keep_multiple_positions(feature, min_no_positions, distance_threshold)
    assert motl.df.equals(ref_motl.df)


@pytest.mark.parametrize('m, shift, ref', [
    ('./example_files/test/shift_positions/allmotl_sp_cl1_1.em', [1, 2, 3], './example_files/test/shift_positions/ref1.em'),
    ('./example_files/test/shift_positions/allmotl_sp_cl1_1.em', [-10, 200, 3.5], './example_files/test/shift_positions/ref2.em'),
    ('./example_files/test/shift_positions/allmotl_sp_cl1_1.em', [0, 0, 0], './example_files/test/shift_positions/ref3.em'),
    ('./example_files/test/shift_positions/allmotl_sp_cl1_1.em', [1, 1, 1], './example_files/test/shift_positions/ref4.em'),
    ('./example_files/test/shift_positions/allmotl_sp_cl1_5.em', [-10, 10, 100], './example_files/test/shift_positions/ref5.em')])
def test_shift_positions(m, shift, ref):
    motl, ref_motl = Motl.load([m, ref])
    motl.shift_positions(shift)
    assert motl.df.equals(ref_motl.df)
    # if not motl.df.equals(ref_motl.df):
    #     merged = motl.df.merge(ref_motl.df, how='outer', indicator=True)
    #     print('Rows only in the TEST dataframe:\n', merged.loc[merged['_merge'] == 'left_only', 'shift_x':'shift_z'])
    #     print('Rows only in the REF dataframe\n: ', merged.loc[merged['_merge'] == 'right_only', 'shift_x':'shift_z'], '\n')


def test_spline_sampling():
    coords = pd.read_csv('./example_files/test/spline/coords.txt', sep='\t', header=None)
    ref = pd.read_csv('./example_files/test/spline/carbon_edge.txt', sep='\t', header=None)
    carbon_edge = Motl.spline_sampling(coords, 2)
    assert carbon_edge.equals(ref)


@pytest.mark.parametrize('m, model_path, model_suffix, distance_threshold, dimensions', [
    ('./example_files/test/clean_on_carbon/allmotl_sp_cl1_1.em', './example_files/test/clean_on_carbon/', '_lt_motl_model', 1, './example_files/test/clean_on_carbon/dimensions.txt')])
def test_clean_particles_on_carbon(m, model_path, model_suffix, distance_threshold, dimensions):
    motl = Motl.load(m)
    motl.clean_particles_on_carbon(model_path, model_suffix, distance_threshold, dimensions)


@pytest.mark.parametrize('motl_list, mask_list, size_list, rotations, ref, ref_masks', [
    (['./example_files/test/recenter_subparticle/sp_motl_cl1_14.em', './example_files/test/recenter_subparticle/sp_motl_cl2_14.em'],
     ['./example_files/test/recenter_subparticle/mask56_recenter1.em', './example_files/test/recenter_subparticle/mask56_recenter2.em'],
     [48, 32], None, './example_files/test/recenter_subparticle/ref1.em',
     ['./example_files/test/recenter_subparticle/mask56_recenter1_centered_ref1.em', './example_files/test/recenter_subparticle/mask56_recenter2_centered_ref1.em']),
    (['./example_files/test/recenter_subparticle/sp_motl_cl1_14.em', './example_files/test/recenter_subparticle/sp_motl_cl2_14.em'],
     ['./example_files/test/recenter_subparticle/mask56_recenter1.em', './example_files/test/recenter_subparticle/mask56_recenter2.em'],
     [48, 32], [[0, 90, 45], [-90, 0, 0]], './example_files/test/recenter_subparticle/ref2.em',
     ['./example_files/test/recenter_subparticle/mask56_recenter1_centered_ref2.em', './example_files/test/recenter_subparticle/mask56_recenter2_centered_ref2.em'])
])
def test_recenter_subparticle(motl_list, mask_list, size_list, rotations, ref, ref_masks):
    ref_df = Motl.load(ref).df
    centered_motl = Motl.recenter_subparticle(motl_list, mask_list, size_list, rotations)

    for i, mask in enumerate(mask_list):
        centered_mask = mask.replace('.em', '_centered.em')
        assert os.path.isfile(centered_mask)
        assert np.array_equal(emfile.read(centered_mask)[1], emfile.read(ref_masks[i])[1])
    assert centered_motl.df.equals(ref_df)
#