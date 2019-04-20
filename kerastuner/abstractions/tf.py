import gc
import tensorflow as tf
import tensorflow.keras.backend as K  # pylint: disable=import-error


def clear_tf_session():
    "Clear tensorflow graph to avoid OOM issues"
    K.clear_session()
    # K.get_session().close() # unsure if it is needed
    gc.collect()

    cfg = tf.ConfigProto()
    cfg.gpu_options.allow_growth = True  # pylint: disable=no-member
    K.set_session(tf.Session(config=cfg))
