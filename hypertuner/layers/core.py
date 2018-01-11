import keras
import random

from ..engine import MLayer

class MDense(MLayer):
    def __init__(self, units,
                 activation=[None],
                 use_bias=True,
                 kernel_initializer='glorot_uniform',
                 bias_initializer='zeros',
                 kernel_regularizer=None,
                 bias_regularizer=None,
                 activity_regularizer=None,
                 kernel_constraint=None,
                 bias_constraint=None,
                 input_shape=None,
                 input_dim=None,
                 **kwargs):
        
        super(MDense, self).__init__(**kwargs)
        self.type = "Dense"
        
        for units_val in units:
            for activation_val in activation:
                if input_shape:
                    l = keras.layers.Dense(units_val, input_shape=input_shape)
                else:
                    l = keras.layers.Dense(units_val)    
            self.instances.append(l)