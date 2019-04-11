from kerastuner.metrics import Loss, ValLoss
from kerastuner.metrics import Accuracy, ValAccuracy
from kerastuner.metrics import CategoricalAccuracy, ValCategoricalAccuracy
from kerastuner.metrics import SparseCategoricalAccuracy
from kerastuner.metrics import ValSparseCategoricalAccuracy
from kerastuner.metrics import BinaryAccuracy, ValBinaryAccuracy
from kerastuner.metrics import TopKCategoricalAccuracy
from kerastuner.metrics import ValTopKCategoricalAccuracy
from kerastuner.metrics import SparseTopKCategoricalAccuracy
from kerastuner.metrics import ValSparseTopKCategoricalAccuracy


def _test_metric(metric, name, direction):
    assert name == metric.name
    assert direction == metric.direction


def test_loss():
    mm = Loss()
    _test_metric(mm, 'loss', 'min')


def test_val_loss():
    mm = ValLoss()
    _test_metric(mm, 'val_loss', 'min')


def test_accuracy():
    mm = Accuracy()
    _test_metric(mm, 'acc', 'max')


def test_val_accuracy():
    mm = ValAccuracy()
    _test_metric(mm, 'val_acc', 'max')


def test_binary_accuracy():
    mm = BinaryAccuracy()
    _test_metric(mm, 'binary_accuracy', 'max')


def test_val_binary_accuracy():
    mm = ValBinaryAccuracy()
    _test_metric(mm, 'val_binary_accuracy', 'max')


def test_categorical_accuracy():
    mm = CategoricalAccuracy()
    _test_metric(mm, 'categorical_accuracy', 'max')


def test_val_categorical_accuracy():
    mm = ValCategoricalAccuracy()
    _test_metric(mm, 'val_categorical_accuracy', 'max')


def test_sparse_categorical_accuracy():
    mm = SparseCategoricalAccuracy()
    _test_metric(mm, 'sparse_categorical_accuracy', 'max')


def test_val_sparse_categorical_accuracy():
    mm = ValSparseCategoricalAccuracy()
    _test_metric(mm, 'val_sparse_categorical_accuracy', 'max')


def test_topk_categorical_accuracy():
    mm = TopKCategoricalAccuracy()
    _test_metric(mm, 'top_k_categorical_accuracy', 'max')


def test_val_topk_categorical_accuracy():
    mm = ValTopKCategoricalAccuracy()
    _test_metric(mm, 'val_top_k_categorical_accuracy', 'max')


def test_sparse_topk_categorical_accuracy():
    mm = SparseTopKCategoricalAccuracy()
    _test_metric(mm, 'sparse_top_k_categorical_accuracy', 'max')


def test_val_sparse_topk_categorical_accuracy():
    mm = ValSparseTopKCategoricalAccuracy()
    _test_metric(mm, 'val_sparse_top_k_categorical_accuracy', 'max')
