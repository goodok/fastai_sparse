# -*- coding: utf-8 -*-

import sys
import numpy as np

from fastai.basic_train import LearnerCallback
from fastai.torch_core import add_metrics


class IouMean(LearnerCallback):
    """
    Calc IoU by classes, then mean.

    Without splitting by examples.
    """
    _order = -19  # Needs to run before the recorder

    def __init__(self, learn, n_classes, names=['train_iou', 'valid_iou'], epsilon=sys.float_info.epsilon, **kwargs):
        self.n_classes = n_classes
        self.names = names
        self.epsilon = epsilon

        super().__init__(learn, **kwargs)

    def append_metrics_names(self, names):
        recorder = self.learn.recorder
        if not hasattr(recorder, '_added_met_names'):
            recorder._added_met_names = []
        recorder._added_met_names += names

    def on_train_begin(self, **kwargs):
        self.append_metrics_names(self.names)

    def on_epoch_begin(self, **kwargs):

        self._d = {'train': {}, 'valid': {}}

        for datatype in ['train', 'valid']:
            d = self._d[datatype]
            d['runned'] = False
            d['cm'] = np.zeros(shape=(self.n_classes, self.n_classes), dtype=np.uint64)
            d['iou_per_class'] = None
            d['iou'] = 0

    def on_batch_end(self, last_output, last_target, last_input, train, **kwargs):

        if train:
            d = self._d['train']
        else:
            d = self._d['valid']

        y_true = last_target.detach().cpu().numpy()

        predictions = last_output.detach().cpu().numpy()
        y_pred = predictions.argmax(1)

        indexer = y_true >= 0

        cm = confusion_matrix(y_pred[indexer], y_true[indexer], self.n_classes)
        d['cm'] += cm
        d['runned'] = True

    def on_epoch_end(self, last_metrics, **kwargs):

        for datatype in ['train', 'valid']:
            d = self._d[datatype]

            if d['runned']:
                cm = d['cm']
                d['iou_per_class'] = iou_per_class_from_cm(cm)
                d['iou'] = np.mean(d['iou_per_class'])

        iou_train = self._d['train']['iou']
        iou_valid = self._d['valid']['iou']

        return add_metrics(last_metrics, [iou_train, iou_valid])


def confusion_matrix(y_pred, y_true, n_classes=None, validate_inputs=True, **kwargs):
    if validate_inputs:
        assert y_pred.min() >= 0
        assert y_true.min() >= 0

        if n_classes is not None:
            assert y_pred.max() < n_classes
            assert y_true.max() < n_classes
        assert y_pred.max() < 256
        assert y_true.max() < 256

    merged_maps = np.bitwise_or(np.left_shift(y_true.astype('uint16'), 8),
                                y_pred.astype('uint16'))
    hist = np.bincount(np.ravel(merged_maps))
    nonzero = np.nonzero(hist)[0]
    pred, label = np.bitwise_and(255, nonzero), np.right_shift(nonzero, 8)

    if n_classes is None:
        n_classes = np.array([pred, label]).max() + 1
    cm = np.zeros([n_classes, n_classes], dtype='uint64')

    cm.put(pred * n_classes + label, hist[nonzero])
    return cm

# # variant from sparseconvnet/scannet (the same result)
# def confusion_matrix_2(pred_ids, gt_ids):
#     assert pred_ids.shape == gt_ids.shape, (pred_ids.shape, gt_ids.shape)
#     # filter -100
#     idxs= gt_ids>=0
#     return np.bincount(pred_ids[idxs]*20+gt_ids[idxs],minlength=400).reshape((20,20)).astype(np.ulonglong)


def iou_per_class_from_cm(cm, epsilon=1e-12):
    "Compute iou per classes from confusion matrix"
    # true positives (list by classes)
    intersection = np.diag(cm)
    ground_truth_set = cm.sum(axis=1)
    predicted_set = cm.sum(axis=0)
    union = ground_truth_set + predicted_set - intersection
    IoU = (intersection + epsilon) / (union.astype(np.float32) + epsilon)
    return IoU


# # from sparseconvnet/scannet (the same result)
# def get_iou(label_id, confusion):
#     VALID_CLASS_IDS = np.array([0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15, 16, 17, 18, 19])

#     # true positives
#     tp = np.longlong(confusion[label_id, label_id])
#     # false negatives
#     fn = np.longlong(confusion[label_id, :].sum()) - tp
#     # false positives
#     not_ignored = [l for l in VALID_CLASS_IDS if not l == label_id]
#     fp = np.longlong(confusion[not_ignored, label_id].sum())

#     denom = (tp + fp + fn)
#     if denom == 0:
#         return float('nan')

def calc_iou_per_class(y_pred, y_true, n_classes=None):
    """
    Calculate IoU per class.

    Return: array of float
        shape is (n_classes, )
     """

    assert y_pred.shape == y_true.shape, "{} != {}".format(y_pred.shape, y_true.shape)
    assert y_pred.dtype == y_true.dtype, "{} != {}".format(y_pred.dtype, y_true.dtype)
    assert y_true.ndim == 1, "{} != 1".format(y_true.ndim)
    assert y_pred.max() < n_classes, "{} >= {}".format(y_pred.max(), n_classes)
    assert y_pred.min() >= 0, "{} < 0".format(y_pred.min())
    assert y_true.max() < n_classes, "{} >= {}".format(y_true.max(), n_classes)
    assert y_true.min() >= 0, "{} < 0".format(y_true.min())

    eps = sys.float_info.epsilon

    iou = np.zeros(n_classes)

    for j in range(n_classes):
        iou[j] = iou_per_parts(y_pred, y_true, j, eps)
    return iou


def inter(pred, gt, label):
    assert pred.size == gt.size, ('Predictions incomplete!', pred.size, gt.size)
    return np.sum(np.logical_and(pred.astype('int') == label, gt.astype('int') == label))


def union(pred, gt, label):
    assert pred.size == gt.size, 'Predictions incomplete!'
    return np.sum(np.logical_or(pred.astype('int') == label, gt.astype('int') == label))


def iou_per_parts(pred, gt, label, epsilon=sys.float_info.epsilon):
    return (inter(pred, gt, label) + epsilon) / (union(pred, gt, label) + epsilon)
