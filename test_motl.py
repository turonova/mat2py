import numpy as np
import pandas as pd
import pytest
from emmotl import Motl


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
    with pytest.raises(Exception):
        motl.get_feature(feature_id)


@pytest.mark.parametrize('feature', ['score', 0])
def test_remove_feature_existing(motl, feature):
    assert float('0.0633') not in motl.remove_feature(feature, 0.0633).df.loc[:, 'score'].values


@pytest.mark.parametrize('m', ['./example_files/test/au_1.em',
                               ['./example_files/test/au_1.em', './example_files/test/au_2.em']])
def test_load(m):
    # TODO check other critical aspects of the motl ?
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
                               './example_files/test/col_missing.em', './example_files/test/na_values.em',
                               './example_files/test/extra_col.em', './example_files/test/bad_values.em',
                               'not_a_file_path', ['./example_files/test/au_1.em', './example_files/au_1.txt']])
def test_load_wrong(m):
    with pytest.raises(Exception):
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
    with pytest.raises(Exception):
        Motl.merge_and_renumber(motl_list)