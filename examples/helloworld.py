""" A simple helloworld example

Different workflows are shown here.
"""
from tensorflow import keras
from tensorflow.keras import layers


from kerastuner.tuners import RandomSearch
from kerastuner.engine.hypermodel import HyperModel
from kerastuner.engine.hyperparameters import HyperParameters


(x, y), (val_x, val_y) = keras.datasets.mnist.load_data()
x = x.astype('float32') / 255.
val_x = val_x.astype('float32') / 255.

x = x[:10000]
y = y[:10000]


"""Basic case:
- We define a `build_model` function
- It returns a compiled model
- It uses hyperparameters defined on the fly
"""


def build_model(hp):
    model = keras.Sequential()
    model.add(layers.Flatten(input_shape=(28, 28)))
    for i in range(hp.Int('num_layers', 2, 20)):
        model.add(layers.Dense(units=hp.Int('units_' + str(i), 32, 512, 32),
                               activation='relu'))
    model.add(layers.Dense(10, activation='softmax'))
    model.compile(
        optimizer=keras.optimizers.Adam(
            hp.Choice('learning_rate', [1e-2, 1e-3, 1e-4])),
        loss='sparse_categorical_crossentropy',
        metrics=['accuracy'])
    return model


tuner = RandomSearch(
    build_model,
    objective='val_accuracy',
    max_trials=5,
    executions_per_trial=3,
    directory='test_dir')

tuner.search_space_summary()

tuner.search(x=x,
             y=y,
             epochs=3,
             validation_data=(val_x, val_y))

tuner.results_summary()


# """Case #2:
# - We override the loss and metrics
# """

tuner = RandomSearch(
    build_model,
    objective='val_accuracy',
    loss=keras.losses.SparseCategoricalCrossentropy(name='my_loss'),
    metrics=['accuracy', 'mse'],
    max_trials=5,
    directory='test_dir')

tuner.search(x, y,
             epochs=5,
             validation_data=(val_x, val_y))


# """Case #3:
# - We define a HyperModel subclass
# """


class MyHyperModel(HyperModel):

    def __init__(self, img_size, classes):
        self.img_size = img_size
        self.classes = classes

    def build(self, hp):
        model = keras.Sequential()
        model.add(layers.Flatten(input_shape=self.img_size))
        for i in range(hp.Int('num_layers', 2, 20)):
            model.add(layers.Dense(units=hp.Int('units_' + str(i), 32, 512, 32),
                                   activation='relu'))
        model.add(layers.Dense(self.classes, activation='softmax'))
        model.compile(
            optimizer=keras.optimizers.Adam(
                hp.Choice('learning_rate', [1e-2, 1e-3, 1e-4])),
            loss='sparse_categorical_crossentropy',
            metrics=['accuracy'])
        return model


tuner = RandomSearch(
    MyHyperModel(img_size=(28, 28), classes=10),
    objective='val_accuracy',
    max_trials=5,
    directory='test_dir')

tuner.search(x,
             y=y,
             epochs=5,
             validation_data=(val_x, val_y))

# """Case #4:
# - We restrict the search space
# - This means that default values are being used for params that are left out
# """

hp = HyperParameters()
hp.Choice('learning_rate', [1e-1, 1e-3])

tuner = RandomSearch(
    build_model,
    max_trials=5,
    hyperparameters=hp,
    tune_new_entries=False,
    objective='val_accuracy')

tuner.search(x=x,
             y=y,
             epochs=5,
             validation_data=(val_x, val_y))

# """Case #5:
# - We override specific parameters with fixed values that aren't the default
# """

hp = HyperParameters()
hp.Fixed('learning_rate', 0.1)

tuner = RandomSearch(
    build_model,
    max_trials=5,
    hyperparameters=hp,
    tune_new_entries=True,
    objective='val_accuracy')

tuner.search(x=x,
             y=y,
             epochs=5,
             validation_data=(val_x, val_y))


# """Case #6:
# - We reparameterize the search space
# - This means that we override the distribution of specific hyperparameters
# """

hp = HyperParameters()
hp.Choice('learning_rate', [1e-1, 1e-3])

tuner = RandomSearch(
    build_model,
    max_trials=5,
    hyperparameters=hp,
    tune_new_entries=True,
    objective='val_accuracy')

tuner.search(x=x,
             y=y,
             epochs=5,
             validation_data=(val_x, val_y))


# """Case #7:
# - We predefine the search space
# - No unregistered parameters are allowed in `build`
# """

hp = HyperParameters()
hp.Choice('learning_rate', [1e-1, 1e-3])
hp.Int('num_layers', 2, 20)


def build_model(hp):
    model = keras.Sequential()
    model.add(layers.Flatten(input_shape=(28, 28)))
    for i in range(hp.get('num_layers')):
        model.add(layers.Dense(32,
                               activation='relu'))
    model.add(layers.Dense(10, activation='softmax'))
    model.compile(
        optimizer=keras.optimizers.Adam(hp.get('learning_rate')),
        loss='sparse_categorical_crossentropy',
        metrics=['accuracy'])
    return model


tuner = RandomSearch(
    build_model,
    max_trials=5,
    hyperparameters=hp,
    allow_new_entries=False,
    objective='val_accuracy')

tuner.search(x=x,
             y=y,
             epochs=5,
             validation_data=(val_x, val_y))


# """Case #8:
# - Similar to Base Case.
# - However, specify conditions on units so that the summary show
# - only relevant hyperparameters.
# """


def build_model(hp):
    model = keras.Sequential()
    model.add(layers.Flatten(input_shape=(28, 28)))
    min_layers = 2
    max_layers = 5
    for i in range(hp.Int('num_layers', min_layers, max_layers)):
        with hp.conditional_scope('num_layers', list(range(i + 1, max_layers + 1))):
            model.add(layers.Dense(units=hp.Int('units_' + str(i), 32, 256, 32),
                                   activation='relu'))
    model.add(layers.Dense(10, activation='softmax'))
    model.compile(
        optimizer=keras.optimizers.Adam(1e-4),
        loss='sparse_categorical_crossentropy',
        metrics=['accuracy'])
    return model


tuner = RandomSearch(
    build_model,
    objective='val_accuracy',
    max_trials=10,
    executions_per_trial=3,
    directory='test_dir')

tuner.search_space_summary()

tuner.search(x=x,
             y=y,
             epochs=3,
             validation_data=(val_x, val_y))

tuner.results_summary()


# """Case #9:
# - Similar to Case #8, but use parent_name, parent_value keywords pair for
# - conditional scope Using keywords for conditional scope does not
# - support nested conditions.
# """


def build_model(hp):
    model = keras.Sequential()
    model.add(layers.Flatten(input_shape=(28, 28)))
    min_layers = 2
    max_layers = 5
    for i in range(hp.Int('num_layers', min_layers, max_layers)):
        model.add(layers.Dense(units=hp.Int(
            'units_' + str(i),
            32,
            256,
            32,
            parent_name='num_layers',
            parent_values=list(range(i + 1, max_layers + 1))),
            activation='relu'))
        model.add(layers.Dense(10, activation='softmax'))
        model.compile(
            optimizer=keras.optimizers.Adam(1e-4),
            loss='sparse_categorical_crossentropy',
            metrics=['accuracy'])
        return model


tuner = RandomSearch(
    build_model,
    objective='val_accuracy',
    max_trials=10,
    executions_per_trial=3,
    directory='test_dir')

tuner.search_space_summary()

tuner.search(x=x,
             y=y,
             epochs=3,
             validation_data=(val_x, val_y))

tuner.results_summary()
