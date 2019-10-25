# Subclassing Tuner for Custom Training Loops

The `Tuner` class at `kerastuner.engine.tuner.Tuner` can be subclassed to support advanced uses such as:

- Custom training loops (GANs, reinforement learning, etc.)
- Adding hyperparameters for preprocessing data.

This tutorial will not cover subclassing to support non-Keras models. To accomplish this, you can subclass the `kerastuner.engine.base_tuner.BaseTuner` class (See `kerastuner.tuners.sklearn.Sklearn` for an example).

### Understanding the search process.

`Tuner.search` can be passed any arguments. These arguments will be passed directly to `Tuner.run_trial`, along with a `Trial` object that contains information about the current trial, including hyperparameters and the status of the trial. Typically, `Tuner.run_trial` is the only method that users need to override when subclassing `Tuner`.

### Example

```python
import kerastuner as kt
import tensorflow as tf
import tensorflow_datasets as tfds

class MyTuner(kt.Tuner):

    def run_trial(self, trial, train_ds):
        hp = trial.hyperparameters

        # Hyperparameters can be added anywhere inside `run_trial`.
        # When the first trial is run, they will take on their default values.
        # Afterwards, they will be tuned by the `Oracle`.
        train_ds = train_ds.batch(
            hp.Int('batch_size', 32, 128, step=32, default=64))

        model = self.hypermodel.build(trial.hyperparameters)
        lr = hp.Float('learning_rate', 1e-4, 1e-2, sampling='log', default=1e-3)
        optimizer = tf.keras.optimizers.Adam(lr)
        epoch_loss_metric = tf.keras.metrics.Mean()

        @tf.function
        def run_train_step(data):
            images = tf.dtypes.cast(data['image'], 'float32') / 255.
            labels = data['label']
            with tf.GradientTape() as tape:
                logits = model(images)
                loss = tf.keras.losses.sparse_categorical_crossentropy(
                    labels, logits)
                # Add any regularization losses.
                loss += tf.math.add_n(model.losses)
                gradients = tape.gradient(loss, model.trainable_variables)
            optimizer.apply_gradients(zip(gradients, model.trainable_variables))
            epoch_loss_metric.update_state(loss)
            return loss

        for epoch in range(10):
            self.on_epoch_begin(trial, model, epoch, logs={})
            for batch, (x, y) in enumerate(train_ds):
                self.on_batch_begin(trial, model, batch, logs={})
                batch_loss = run_train_step(x, y).numpy()
                self.on_batch_end(trial, model, batch, logs={'loss': batch_loss})
            epoch_loss = epoch_loss_metric.result().numpy()
            self.on_epoch_end(trial, model, epochs, logs={'loss': epoch_loss})
            epoch_loss_metric.reset_states()


tuner = MyTuner(
    oracle=kt.oracles.BayesianOptimization(
        objective='loss',
        max_trials=40),
    hypermodel=build_model,
    directory='results',
    project_name='mnist_custom_training')


