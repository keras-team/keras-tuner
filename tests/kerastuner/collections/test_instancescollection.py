import os
from pathlib import Path

from kerastuner.collections.instancescollection import InstancesCollection


def test_loading():
    data_path = Path(__file__).parents[2]
    data_path = Path.joinpath(data_path, 'data', 'results')

    col = InstancesCollection()
    count = col.load_from_dir(data_path)
    assert count == 2