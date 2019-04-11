from .metric import Metric


class Loss(Metric):
    def __init__(self):
        super(Loss, self).__init__('loss', 'min')


class ValLoss(Metric):
    def __init__(self):
        super(ValLoss, self).__init__('val_loss', 'min')


class Accuracy(Metric):
    def __init__(self):
        super(Accuracy, self).__init__('acc', 'max')


class ValAccuracy(Metric):
    def __init__(self):
        super(ValAccuracy, self).__init__('val_acc', 'max')


class BinaryAccuracy(Metric):
    def __init__(self):
        super(BinaryAccuracy, self).__init__('binary_accuracy', 'max')


class ValBinaryAccuracy(Metric):
    def __init__(self):
        super(ValBinaryAccuracy, self).__init__('val_binary_accuracy', 'max')


class CategoricalAccuracy(Metric):
    def __init__(self):
        super(CategoricalAccuracy, self).__init__('categorical_accuracy',
                                                  'max')


class ValCategoricalAccuracy(Metric):
    def __init__(self):
        super(ValCategoricalAccuracy, self).__init__(
             'val_categorical_accuracy', 'max')


class SparseCategoricalAccuracy(Metric):
    def __init__(self):
        super(SparseCategoricalAccuracy, self).__init__(
            'sparse_categorical_accuracy', 'max')


class ValSparseCategoricalAccuracy(Metric):
    def __init__(self):
        super(ValSparseCategoricalAccuracy, self).__init__(
            'val_sparse_categorical_accuracy', 'max')


class TopKCategoricalAccuracy(Metric):
    def __init__(self):
        super(TopKCategoricalAccuracy, self).__init__(
            'top_k_categorical_accuracy', 'max')


class ValTopKCategoricalAccuracy(Metric):
    def __init__(self):
        super(ValTopKCategoricalAccuracy, self).__init__(
            'val_top_k_categorical_accuracy', 'max')


class SparseTopKCategoricalAccuracy(Metric):
    def __init__(self):
        super(SparseTopKCategoricalAccuracy, self).__init__(
            'sparse_top_k_categorical_accuracy', 'max')


class ValSparseTopKCategoricalAccuracy(Metric):
    def __init__(self):
        super(ValSparseTopKCategoricalAccuracy, self).__init__(
            'val_sparse_top_k_categorical_accuracy', 'max')
