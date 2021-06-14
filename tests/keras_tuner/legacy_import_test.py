import pytest


def test_kerastuner_same_as_keras_tuner():
    with pytest.deprecated_call():
        import kerastuner
        from kerastuner.tuners import RandomSearch
        from kerastuner.tuners import BayesianOptimization
        from kerastuner.tuners import Hyperband
        from kerastuner.tuners import Sklearn  # noqa: F401
        from kerastuner.oracles import RandomSearch  # noqa: F401,F811
        from kerastuner.oracles import BayesianOptimization  # noqa: F401,F811
        from kerastuner.oracles import Hyperband  # noqa: F401,F811
        from kerastuner.engine.base_tuner import BaseTuner  # noqa: F401
        from kerastuner.engine.conditions import Condition  # noqa: F401
        from kerastuner.engine.hypermodel import HyperModel  # noqa: F401
        from kerastuner.engine.hyperparameters import HyperParameter  # noqa: F401
        from kerastuner.engine.hyperparameters import HyperParameters  # noqa: F401
        from kerastuner.engine.logger import CloudLogger  # noqa: F401
        from kerastuner.engine.logger import Logger  # noqa: F401
        from kerastuner.engine.metrics_tracking import (  # noqa: F401
            MetricObservation,
        )
        from kerastuner.engine.oracle import Objective  # noqa: F401
        from kerastuner.engine.oracle import Oracle  # noqa: F401
        from kerastuner.engine.tuner import Tuner  # noqa: F401
        from kerastuner.engine.stateful import Stateful  # noqa: F401
        from kerastuner.engine.trial import Trial  # noqa: F401
        from kerastuner.engine.multi_execution_tuner import (  # noqa: F401
            MultiExecutionTuner,
        )
        from kerastuner.applications import HyperResNet  # noqa: F401
        from kerastuner.applications import HyperXception  # noqa: F401

    import keras_tuner

    attr1 = [attr for attr in dir(kerastuner) if not attr.startswith("__")]
    attr2 = [attr for attr in dir(keras_tuner) if not attr.startswith("__")]

    assert len(attr1) > 20
    assert set(attr1) >= set(attr2)
