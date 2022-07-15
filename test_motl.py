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
