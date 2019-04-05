from kerastuner.distributions import Choice, Linear, Range


def default_fixed_hparams(input_shape, num_classes):
    ### Parameters ###
    hp = {
        "kernel_size": (3, 3),
        "initial_strides": (2, 2),
        "activation": "relu",
        "optimizer": "rmsprop",
        "learning_rate": .001,
        "learning_rate_decay": .9,
        "momentum": .9,
        "conv2d_num_filters": 64,
        "sep_num_filters": 256,
        "num_residual_blocks": 4,
        "dense_use_bn": True,
        "dropout_rate": 0.0,
        "dense_merge_type": "avg",
        "num_dense_layers": 1
    }
    return hp


def default_hparams(input_shape, num_classes):
    ### Parameters ###
    hp = {}
    # [general]

    kernel_size = Range("kernel_size", 3, 5, 2, group="general")

    hp["kernel_size"] = (kernel_size, kernel_size)
    hp["initial_strides"] = (2, 2)
    hp["activation"] = "relu"
    #Choice("activation", ["relu", "selu"], group="general")

    hp["optimizer"] = Choice(
        "optimizer", ["adam", "rmsprop", "sgd"], "general")

    if hp["optimizer"] == "sgd":
        hp["learning_rate"] = Linear(
            "learning_rate", start=.03, stop=.05, num_buckets=10, group="general")
        hp["learning_rate_decay"] = Linear(
            "learning_rate_decay", start=.9, stop=.95, num_buckets=5, group="general")
        hp["momentum"] = Linear("momentum", start=.8, stop=.95,
                                num_buckets=6, group="general")
    elif hp["optimizer"] == "rmsprop":
        hp["learning_rate"] = Choice(
            "learning_rate", [.001, .0001, .00001], group="general")
        hp["momentum"] = Linear("momentum", start=.8, stop=.95,
                                num_buckets=6, group="general")
        hp["learning_rate_decay"] = Linear(
            "learning_rate_decay", start=.87, stop=.92, num_buckets=5, group="general")
    else:
        hp["learning_rate"] = Choice(
            "learning_rate", [.001, .0001, .00001], group="general")

    # [entry flow]

    # -conv2d
    hp["conv2d_num_filters"] = Choice(
        "num_filters", [32, 64, 128], group="conv2d")

    # seprarable block > not an exact match to the paper
    hp["sep_num_filters"] = Range(
        "num_filters", 128, 768, 64, group="entry_flow")

    # [Middle Flow]
    hp["num_residual_blocks"] = Range(
        "num_residual_blocks", 1, 8, group="middle_flow")

    # [Exit Flow]
    hp["dense_merge_type"] = Choice(
        "merge_type", ["avg", "flatten", "max"], group="exit_flow")
    hp["num_dense_layers"] = Range(
        "dense_layers", 1, 3, group="exit_flow")

    hp["dropout_rate"] = Linear(
        "dropout", start=0.0, stop=0.3, num_buckets=4, group="exit_flow")
    hp["dense_use_bn"] = Choice(
        "batch_normalization", [True, False], "exit_flow")

    return hp
