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
