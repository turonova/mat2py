import numpy as np
import os
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


def test_after_recenter(motl, ref_motl):
    # TO BE USED IN ALL TEST FOR METHODS APPLYING RECENTER_PARTICLES
    # to take into account Python 0.5 rounding: round(1.5) = 2, BUT round(2.5) = 2, while in Matlab round(2.5) = 3
    if not motl.df.equals(ref_motl.df):
        merged = motl.df.merge(ref_motl.df, how='outer', indicator=True)
        print('Rows only in the TEST dataframe:\n', merged.loc[merged['_merge'] == 'left_only', 'x':'z'])
        print('Rows only in the REF dataframe\n: ', merged.loc[merged['_merge'] == 'right_only', 'x':'z'], '\n')
        assert (motl.df.loc[:, 'x'] + motl.df.loc[:, 'shift_x']).equals(
            ref_motl.df.loc[:, 'x'] + ref_motl.df.loc[:, 'shift_x'])
        assert (motl.df.loc[:, 'y'] + motl.df.loc[:, 'shift_y']).equals(
            ref_motl.df.loc[:, 'y'] + ref_motl.df.loc[:, 'shift_y'])
        assert (motl.df.loc[:, 'z'] + motl.df.loc[:, 'shift_z']).equals(
            ref_motl.df.loc[:, 'z'] + ref_motl.df.loc[:, 'shift_z'])


@pytest.mark.parametrize('m, ref', [
    ('./example_files/test/recenter/allmotl_sp_cl1_1.em', './example_files/test/recenter/ref1.em'),
    ('./example_files/test/recenter/allmotl_sp_cl1_2.em', './example_files/test/recenter/ref2.em')])
def test_recenter_particles(m, ref):
    motl, ref_motl = Motl.load([m, ref])
    motl.df = Motl.recenter_particles(motl.df)

    test_after_recenter(motl, ref_motl)


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
    test_after_recenter(motl, ref_motl)
