# Copyright 2019 The Keras Tuner Authors
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     https://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

import gc
import numpy as np
from tensorflow.python import Session, ConfigProto
from tensorflow.python.keras import backend as K


def compute_model_size(model):
    "comput the size of a given model"
    params = [K.count_params(p) for p in model.trainable_weights]
    return int(np.sum(params))


def clear_tf_session():
    "Clear tensorflow graph to avoid OOM issues"
    K.clear_session()
    # K.get_session().close() # unsure if it is needed
    gc.collect()

    if hasattr(K, 'set_session'):
        cfg = ConfigProto()
        cfg.gpu_options.allow_growth = True  # pylint: disable=no-member
        K.set_session(Session(config=cfg))
