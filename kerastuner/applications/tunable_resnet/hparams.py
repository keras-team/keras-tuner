""" Hyperparameter definitions for Tunable Resnet.
"""

from kerastuner.distributions import Choice, Linear, Range


def default_fixed_hparams(input_shape, num_classes):
    return {
        "version": "v2",
        "conv3_depth": 4,
        "conv4_depth": 6,
        "optimizer": "adam",
        "learning_rate": .01
    }


def default_hparams(input_shape, num_classes):
    hp = {
        "version": Choice("version", ["v1", "v2", "next"]),
        "learning_rate": Choice('lr', [0.1, 0.01, 0.001]),
        "optimizer": Choice("optimizer", ["adam", "rmsprop", "sgd"])
    }
    if hp["version"] == "v1":
        hp['conv3_depth'] = Choice('conv3_depth', [4, 8], group= 'stack')
        hp['conv4_depth'] = Choice('conv4_depth', [6, 23, 36], group= 'stack')
        hp['preact'] = False
        hp['use_bias'] = True
    elif hp["version"] == "v2":
        hp['conv3_depth'] = Choice('conv3_depth', [4, 8], group= 'stack')
        hp['conv4_depth'] = Choice('conv4_depth', [6, 23, 36], group= 'stack')
        hp['preact'] = True
        hp['use_bias'] = True
    elif hp["version"] == "next":
        hp['conv3_depth'] = Choice('conv3_depth', [4], group= 'stack')
        hp['conv4_depth'] = Choice('conv4_depth', [6, 23], group= 'stack')
        hp['preact'] = False
        hp['use_bias'] = False
    return hp
