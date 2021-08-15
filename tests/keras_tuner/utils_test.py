import pytest

from keras_tuner import utils


def test_check_tf_version_error():
    utils.tf.__version__ = "1.15.0"

    with pytest.warns(ImportWarning) as record:
        utils.check_tf_version()
    assert len(record) == 1
    assert (
        "Tensorflow package version needs to be at least 2.0.0"
        in record[0].message.args[0]
    )
