import numpy as np
import pandas as pd
import pytest

from emmotl import Motl
from exceptions import UserInputError


@pytest.fixture
def motl():
    df = pd.read_csv('./example_files/au_1.csv',
                     dtype=float, header=None,
                     names=['score', 'geom1', 'geom2', 'subtomo_id', 'tomo_id', 'object_id',
                            'subtomo_mean', 'x', 'y', 'z', 'shift_x', 'shift_y', 'shift_z', 'geom4',
                            'geom5', 'geom6', 'phi', 'psi', 'theta', 'class'])
    motl = Motl(df)
    return motl


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
    assert float('0.0633') not in motl.remove_feature(feature, 0.0633).df.loc[:, 'score'].values
@pytest.mark.parametrize('m', ['./example_files/test/au_1.em', './example_files/test/au_2.em'])
def test_read_from_emfile(m):
    # TODO check other critical aspects of the motl ?
    motl = Motl.read_from_emfile(m)
    assert isinstance(motl.header, dict)
    assert np.array_equal(motl.df.columns, ['score', 'geom1', 'geom2', 'subtomo_id', 'tomo_id', 'object_id',
                                            'subtomo_mean', 'x', 'y', 'z', 'shift_x', 'shift_y', 'shift_z',
                                            'geom4', 'geom5', 'geom6', 'phi', 'psi', 'theta', 'class'])
    assert all(dt == 'float64' for dt in motl.df.dtypes.values)


@pytest.mark.parametrize('m', ['./example_files/test/col_missing.em', './example_files/test/extra_col.em'])
# TODO did not manage to write out corrupted em file '/test/na_values.em', '/test/bad_values.em'
def test_read_from_emfile_wrong(m):
    with pytest.raises(UserInputError):
        Motl.read_from_emfile(m)


@pytest.mark.parametrize('m', ['./example_files/test/au_1.em',
                               ['./example_files/test/au_1.em', './example_files/test/au_2.em']])
def test_load(m):
    # TODO test Motl instance given (does not work within parametrize)
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
