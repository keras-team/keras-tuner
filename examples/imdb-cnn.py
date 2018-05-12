'''This example demonstrates the use of Convolution1D for text classification.
Hyper-version of https://github.com/keras-team/keras/blob/master/examples/imdb_cnn.py

Original example test accuracy: 0.89
'''
from __future__ import print_function

from keras.preprocessing import sequence
from keras.models import Sequential
from keras.layers import Dense, Dropout, Activation
from keras.layers import Embedding
from keras.layers import Conv1D, GlobalMaxPooling1D
from keras.datasets import imdb
from keras.regularizers import l1, l2, l1_l2
from keras.optimizers import Adam
# hypertuning
from kerastuner.distributions import Range, Choice, Linear
from kerastuner.tuners import RandomSearch

# model constant as defined in original example
NUM_EPOCHS = 2 
MAX_LENGTH = 400
MAX_FEATURES = 5000
BATCH_SIZE = 32

KERASTUNER_NUM_ITERATIONS = 100 # how many models to test 
KERASTUNER_NUM_EXECUTIONS = 3 # how many time to execute each model
NUM_GPU = 1 # How many GPU



def model_fn():
  # set of hyper parameters based of the choice made in the initial example
  embedding_dims = Range(40, 60, 5)
  filters = Range(100, 500, 10)
  kernel_size = Range(1, 5)
  hidden_dims = Range(100, 400, 20)
  dropout_rate = Linear(0.1, 0.5, 6)
  learning_rate = Linear(0.001, 0.01, 11)
  bias_regularizer = Linear(0.01, 0.03, 10)
  model = Sequential()
  model.add(Embedding(MAX_FEATURES, embedding_dims, input_length=MAX_LENGTH))
  model.add(Dropout(dropout_rate))
  model.add(Conv1D(filters, kernel_size, padding='valid', activation='relu', strides=1, bias_regularizer=l1_l2(bias_regularizer, bias_regularizer)))
  model.add(GlobalMaxPooling1D())
  model.add(Dense(hidden_dims))
  model.add(Dropout(dropout_rate))
  model.add(Activation('relu'))
  model.add(Dense(1))
  model.add(Activation('sigmoid'))

  model.compile(loss='binary_crossentropy', optimizer=Adam(lr=learning_rate), metrics=['accuracy'])
  return model



print('Loading data...')
(x_train, y_train), (x_test, y_test) = imdb.load_data(num_words=MAX_FEATURES)
print(len(x_train), 'train sequences')
print(len(x_test), 'test sequences')

print('Pad sequences (samples x time)')
x_train = sequence.pad_sequences(x_train, maxlen=MAX_LENGTH)
x_test = sequence.pad_sequences(x_test, maxlen=MAX_LENGTH)
print('x_train shape:', x_train.shape)
print('x_test shape:', x_test.shape)

# which metrics to track across the runs and display
METRICS_TO_REPORT = [('acc', 'max'), ('val_acc', 'max')] 
hypermodel = RandomSearch(model_fn, num_iterations=KERASTUNER_NUM_ITERATIONS, num_executions=KERASTUNER_NUM_EXECUTIONS, 
                      metrics=METRICS_TO_REPORT, num_gpu=NUM_GPU, batch_size=BATCH_SIZE)
hypermodel.search(x_train, y_train, epochs=NUM_EPOCHS, validation_data=(x_test, y_test))