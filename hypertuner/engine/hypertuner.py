import keras
import random
import pprint
import copy
            
from termcolor import cprint
from xxhash import xxh32
from tabulate import tabulate

class HyperTuner(object):
    """Abstract hypertuner class."""

    def __init__(self, **kwargs):
        self.layers = []
        self.compile_options = []
        self.iterations = kwargs.get('iterations', 10)
        self.dryrun = kwargs.get('dryrun', False)
        self.max_fail_streak = kwargs.get('max_fail_streak', 20)
        self.invalid_architectures = 0
        self.failed_compilations = 0
        self.processed_architetures = []
 

    def add(self, mlayer):
        "Add meta-layer to the meta-model"
        self.layers.append(mlayer)
    
    def compile(self, optimizer=['adam'], loss=['binary_crossentropy'], metrics=[['accuracy']]):
        "Add set of compiles options"
        for optimizer_val in optimizer:
            for loss_val in loss:
                for metrics_val in metrics:
                    self.compile_options.append([optimizer_val, loss_val, metrics_val])

    
    def get_random_model_instance(self, verbose=0):
        "Return a random construction of the model"
        fail_streak = 0
        max_fail_streak = self.max_fail_streak
        while 1:
            # building architecture
            instance = []
            instance_hash = xxh32()
            for layer in self.layers:
                # skipping optional layers at rando
                if layer.optional and random.randint(0, 1) == 0:
                    continue

                layer_instance = layer.get_random_instance()
                instance.append(layer_instance)
                instance_hash.update(str(layer_instance.get_config()))

            digest = instance_hash.hexdigest()
            if digest in self.processed_architetures:
                fail_streak += 1
                if verbose:
                    print "[Info] %s Already processed -- skipping" % digest
                if fail_streak == max_fail_streak:
                    raise ValueError("[ERROR] Generated %s invalid archictures in a raw - something is wrong. Quitting" % fail_streak)
                continue

            self.processed_architetures.append(digest)
            
            #let's keras to all the heavy lifting
            model = None
            try:
                model = keras.models.Sequential(instance)
            except:
                self.invalid_architectures += 1
                if verbose:
                    print "[Warning] invalid model, skipping"
                    continue

            # adding compiler options
            i = random.randint(0, len(self.compile_options) - 1)
            co = self.compile_options[i]
            try:
                model.compile(optimizer=co[0], loss=co[1], metrics=co[2])
            except:
                self.failed_compilations += 1
                if verbose:
                    print "[Warning] invalid model!, skipping"
                continue
            
            if verbose:
                model.summary()
            
            return model

    def summary(self):
        num_models = 1
        table = [['Layer type', 'Optional?', 'Instances #']]
        for layer in self.layers:
            num_instances = len(layer.instances)
            num_models *= num_instances
            table.append([layer.type, layer.optional, num_instances])
        num_models *= len(self.compile_options)
        print tabulate(table, headers='firstrow', tablefmt="grid")
        print 'Layers:%s' % len(self.layers)
        print 'Potential models: %s' % num_models
    
    def statistics(self):
        print 'Invalid architectures:%s' % self.invalid_architectures
        print 'Failed compilation:%s' % self.failed_compilations

    def debug(self):
        'cause life sometime sucks'

        print "num layers:%s" % (len(self.layers))
        for i, lr in enumerate(self.layers):
            inputs_shapes = []
            units = []
            
            print "[%s]" % i
            print "|-num configurations:%s"  % ( len(lr.instances))
            
            for ist in lr.instances:
                cfg = ist.get_config()
                if 'batch_input_shape'in cfg:
                    inputs_shapes.append(str(cfg['batch_input_shape']))
                units.append(cfg['units'])

    def debug_instance(self, instance):
        "Display in-depth stats about a given instance architecture"
        print "|-input: %s" % (inputs_shapes)
        print "|-units: %s" % (units)

        cprint('=-=-=-=[ Model ]=-=-=-=', 'magenta')
        for i, layer in enumerate(instance):
            cprint("layer %s" % i, 'yellow')
            cfg = layer.get_config()
            if 'batch_input_shape'in cfg:
                print "|-input:%s" % str(cfg['batch_input_shape'])
            print "|-units: %s" % cfg['units']