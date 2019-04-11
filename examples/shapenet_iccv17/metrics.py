# -*- coding: utf-8 -*-

import sys
import numpy as np

from fastai.basic_train import LearnerCallback
from fastai.torch_core import add_metrics

# from fastai_sparse.utils import log
from fastai_sparse.metrics import calc_iou_per_class


class IouByCategories(LearnerCallback):
    """
    From original paper [1]:

    To evaluate the accuracy of our models, we adopt the intersection-over-union (IoU) metric of [23].
    The IoU is computed for each part per object category and averagedover parts and examples for the
    category to produce a “per-category IoU”. This way of averaging the IoU scores rewards models that
    make accurate predictions even for object-parts that are very small: small parts have the same weight
    in the final accuracy measure as larger parts.

    The final accuracy measure is obtained by taking a weighted average of the per-category IoUs,
    using  the  fraction  of training examples per category as weights

    Remarks
    -------
    The weighted average for valid is using the fraction of valid examples.

    Notes
    -----
    [1] 3D Semantic Segmentation with Submanifold Sparse Convolutional Networks
    Benjamin Graham, Martin Engelcke†, Laurens van der Maaten
    https://arxiv.org/pdf/1711.10275.pdf
    """

    _order = -19  # Needs to run before the recorder

    def __init__(self, learn, n_categories, epsilon=sys.float_info.epsilon, weighted=True, **kwargs):
        self.n_categories = n_categories
        self.epsilon = epsilon
        self.weighted = weighted

        super().__init__(learn, **kwargs)

    def append_metrics_names(self, names):
        recorder = self.learn.recorder
        if not hasattr(recorder, '_added_met_names'):
            recorder._added_met_names = []
        recorder._added_met_names += names

    def on_train_begin(self, **kwargs):
        if self.weighted:
            self.append_metrics_names(['train_waiou', 'valid_waiou'])
        else:
            self.append_metrics_names(['train_ioucateg', 'valid_ioucateg'])

    def on_epoch_begin(self, **kwargs):

        self._d = {'train': {}, 'valid': {}}

        for datatype in ['train', 'valid']:
            d = self._d[datatype]
            d['examples_by_category'] = np.zeros(self.n_categories, dtype='int')
            d['sum_iou_by_category'] = np.zeros(self.n_categories, dtype=np.float32)

    def on_batch_end(self, last_output, last_target, last_input, train, **kwargs):

        n, iou = calc_iou_by_category(last_input, last_target, last_output, self.n_categories, are_tensors=True)

        if train:
            d = self._d['train']
        else:
            d = self._d['valid']

        d['examples_by_category'] += n
        d['sum_iou_by_category'] += iou

    def on_epoch_end(self, last_metrics, **kwargs):

        for datatype in ['train', 'valid']:
            d = self._d[datatype]
            iou_by_categ, non_zeros = self.mean_iou_by_category(d['sum_iou_by_category'], d['examples_by_category'])
            if iou_by_categ is not None:
                d['mean_iou_by_category'] = iou_by_categ
                d['non_zeros'] = non_zeros
                d['average'] = self.average_iou(iou_by_categ, d['examples_by_category'][non_zeros])
            else:
                d['mean_iou_by_category'] = 0
                d['non_zeros'] = non_zeros
                d['average'] = 0

        iou_train = self._d['train']['average']
        iou_valid = self._d['valid']['average']

        return add_metrics(last_metrics, [iou_train, iou_valid])

    def mean_iou_by_category(self, iou_by_category, n_examples_by_category):
        # filter categories where examples exist
        non_zeros = n_examples_by_category != 0
        # if no examples
        if non_zeros.sum() == 0:
            return None, 0

        iou = iou_by_category[non_zeros]
        n_examples = n_examples_by_category[non_zeros]

        # devide by examples, because there is sumation by examples in the `calc_iou_per_class`
        iou = iou / n_examples

        return iou, non_zeros

    def average_iou(self, mean_iou_by_category, n_examples_by_category):
        """
        Average by categories.
        """
        if self.weighted:
            return (mean_iou_by_category * n_examples_by_category).sum() / n_examples_by_category.sum()
        else:
            return np.mean(mean_iou_by_category)


def calc_iou_by_category(xb, yb, output, n_categories, are_tensors=False):
    """
    Calculate mean (per parts) IoU of examples through batch and accumulate it by the categories.
    """
    if are_tensors:
        output = output.detach().cpu().numpy()
        # categ = categ.detach().cpu().numpy()
        yb = yb.detach().cpu().numpy()

    segments = np.cumsum([0] + xb['nPoints'])

    n_examples_by_category = np.zeros(n_categories, dtype='int')
    res = np.zeros(n_categories, dtype=np.float32)

    for i_example in range(len(xb['nPoints'])):
        start = segments[i_example]
        end = segments[i_example + 1]

        # class offset of the example
        class_offset = xb['classOffset'][i_example]
        # number of classes in the example
        n_classes = xb['nClasses'][i_example]

        preds = output[start:end]
        preds = preds[:, class_offset: class_offset + n_classes]
        y_pred = preds.argmax(1)

        gt = yb[start:end]
        gt = gt - class_offset

        # calculate iou per part.  shape = (n_classes, )
        # TODO: option n_classes = gt.max() + 1
        # n_classes = gt.max() + 1
        iou_per_parts = calc_iou_per_class(y_pred, gt, n_classes)
        # mean iou of example
        mean_iou = np.mean(iou_per_parts)

        # obtain category index
        categ_idx = xb['categ'][i_example]

        # accumulate number of examples and mean (per parts) IoU of example
        n_examples_by_category[categ_idx] = n_examples_by_category[categ_idx] + 1
        res[categ_idx] = res[categ_idx] + mean_iou

        # print(iou)

    return n_examples_by_category, res
