# Distributed Tuning

Keras Tuner makes it easy to perform distributed hyperparameter search. No changes to your code are needed to scale up from running single-threaded locally to running on dozens or hundreds of workers in parallel. Distributed Keras Tuner uses a chief-worker model. The chief runs a service to which the workers report results and query for the hyperparameters to try next. The chief should be run on a single-threaded CPU instance (or alternatively as a separate process on one of the workers).

### Configuring distributed mode

Configuring distributed mode for Keras Tuner only requires setting three environment variables:

**KERASTUNER_TUNER_ID**: This should be set to "chief" for the chief process. Other workers should be passed a unique ID (by convention, "tuner0", "tuner1", etc).

**KERASTUNER_ORACLE_IP**: The IP address or hostname that the chief service should run on. All workers should be able to resolve and access this address.

**KERASTUNER_ORACLE_PORT**: The port that the chief service should run on. This can be freely chosen, but must be a port that is accessible to the other workers. Instances communicate via the [gRPC](https://www.grpc.io) protocol.

The same code can be run on all workers. Additional considerations for distributed mode are:

- All workers should have access to a centralized file system to which they can write their results.
- All workers should be able to access the necessary training and validation data needed for tuning.
- To support fault-tolerance, `overwrite` should be kept as `False` in `Tuner.__init__` (`False` is the default).

Example bash script for chief service (sample code for `run_tuning.py` at bottom of page):

```bash
export KERASTUNER_TUNER_ID="chief"
export KERASTUNER_ORACLE_IP="127.0.0.1"
export KERASTUNER_ORACLE_PORT="8000"
python run_tuning.py
```

Example bash script for worker:

```bash
export KERASTUNER_TUNER_ID="tuner0"
export KERASTUNER_ORACLE_IP="127.0.0.1"
export KERASTUNER_ORACLE_PORT="8000"
python run_tuning.py
```

### Data parallelism with tf.distribute

Keras Tuner also supports data parallelism via [tf.distribute](https://www.tensorflow.org/tutorials/distribute/keras). Data parallelism and distributed tuning can be combined. For example, if you have 10 workers with 4 GPUs on each worker, you can run 10 parallel trials with each trial training on 4 GPUs by using [tf.distribute.MirroredStrategy](https://www.tensorflow.org/api_docs/python/tf/distribute/MirroredStrategy). You can also run each trial on TPUs via [tf.distribute.experimental.TPUStrategy](https://www.tensorflow.org/api_docs/python/tf/distribute/experimental/TPUStrategy). Currently [tf.distribute.MultiWorkerMirroredStrategy](https://www.tensorflow.org/api_docs/python/tf/distribute/experimental/MultiWorkerMirroredStrategy) is not supported, but support for this is on the roadmap.


### Example code

When the enviroment variables described above are set, the example below will run distributed tuning and will also use data parallelism within each trial via `tf.distribute`. The example loads MNIST from `tensorflow_datasets` and uses hyperband for the hyperparameter search.


```python
import kerastuner as kt
import tensorflow as tf
import tensorflow_datasets as tfds


def build_model(hp):
    """Builds a convolutional model."""
    inputs = tf.keras.Input(shape=(28, 28, 1))
    x = inputs
    for i in range(hp.Int('conv_layers', 1, 3, default=3)):
        x = tf.keras.layers.Conv2D(
            filters=hp.Int('filters_' + str(i), 4, 32, step=4, default=8),
            kernel_size=hp.Int('kernel_size_' + str(i), 3, 5),
            activation='relu',
            padding='same')(x)

        if hp.Choice('pooling' + str(i), ['max', 'avg']) == 'max':
            x = tf.keras.layers.MaxPooling2D()(x)
        else:
            x = tf.keras.layers.AveragePooling2D()(x)

        x = tf.keras.layers.BatchNormalization()(x)
        x = tf.keras.layers.ReLU()(x)

    if hp.Choice('global_pooling', ['max', 'avg']) == 'max':
        x = tf.keras.layers.GlobalMaxPooling2D()(x)
    else:
        x = tf.keras.layers.GlobalAveragePooling2D()(x)
    outputs = tf.keras.layers.Dense(10, activation='softmax')(x)

    model = tf.keras.Model(inputs, outputs)

    optimizer = hp.Choice('optimizer', ['adam', 'sgd'])
    model.compile(optimizer, loss='sparse_categorical_crossentropy', metrics=['accuracy'])
    return model


def convert_dataset(item):
    """Puts the mnist dataset in the format Keras expects, (features, labels)."""
    image = item['image']
    label = item['label']
    image = tf.dtypes.cast(image, 'float32') / 255.
    return image, label


def main():
    """Runs the hyperparameter search."""
    tuner = kt.Hyperband(
        hypermodel=build_model,
        objective='val_accuracy',
        max_epochs=8,
        factor=2,
        hyperband_iterations=3,
        distribution_strategy=tf.distribute.MirroredStrategy(),
        directory='results_dir',
        project_name='mnist')

    mnist_data = tfds.load('mnist')
    mnist_train, mnist_test = mnist_data['train'], mnist_data['test']
    mnist_train = mnist_train.map(convert_dataset).shuffle(1000).batch(100).repeat()
    mnist_test = mnist_test.map(convert_dataset).batch(100)

    tuner.search(mnist_train,
                 steps_per_epoch=600,
                 validation_data=mnist_test,
                 validation_steps=100,
                 callbacks=[tf.keras.callbacks.EarlyStopping('val_accuracy')])


if __name__ == '__main__':
    main()
```
