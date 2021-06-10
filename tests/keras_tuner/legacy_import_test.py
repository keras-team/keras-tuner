import pytest


def test_kerastuner_same_as_keras_tuner():
    with pytest.deprecated_call():
        import kerastuner

    import keras_tuner

    attr1 = [attr for attr in dir(kerastuner) if not attr.startswith("__")]
    attr2 = [attr for attr in dir(keras_tuner) if not attr.startswith("__")]

    assert len(attr1) > 20
    assert set(attr1) >= set(attr2)
