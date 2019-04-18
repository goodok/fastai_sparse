# -*- coding: utf-8 -*-

import sys
import numpy as np
import tracemalloc

from torch import Tensor
from fastai.torch_core import Rank0Tensor
from fastai.callback import Callback
from fastai.basic_train import LearnerCallback
from fastai.torch_core import add_metrics

from fastai_sparse.utils import log
from fastai_sparse.metrics import confusion_matrix, iou_per_class_from_cm

import warnings

class IouMeanFiltred(LearnerCallback):
    """
    Calc IoU by classes, filter incorrect classes (-100), then mean.
    """

    _order=-19 # Needs to run before the recorder
    def __init__(self, learn, n_classes, names=['train_iouf', 'valid_iouf'], epsilon=sys.float_info.epsilon, **kwargs):
        self.n_classes = n_classes
        self.epsilon = epsilon
        self.names = names

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
            #d['predictions_all'] = []
            #d['gt_all'] = []
            d['runned'] = False
            d['cm'] = np.zeros(shape=(self.n_classes, self.n_classes), dtype=np.uint64)


    def on_batch_end(self, last_output, last_target, last_input, train, **kwargs):

        if train:
            d = self._d['train']
        else:
            d = self._d['valid']


        predictions = last_output.detach().cpu().numpy()
        xb = last_input

        num_points_actual_cumsum = np.cumsum([0] + xb['num_points'])

        # for each example in the batch extract prediction, argmax, fill omitted by 0-label class (bug), and store
        for k in range(len(xb['id'])):
            # actual number of points
            #num_points = xb['num_points'][k]     # equal len(y)
            
            labels_raw = xb['labels_raw'][k]
            filtred_mask = xb['filtred_mask'][k]
            num_points_raw = len(labels_raw)
        
        
            # extract prediction of example
            start = num_points_actual_cumsum[k]
            end = num_points_actual_cumsum[k + 1]
            example_preds_actual = predictions[start:end]
        
        
            # variant I,
            # Don't compute argmax by axis 1 now, for the accumulations of probabilities when TTA ? But `preds_all` will be is 2 Gb of memory
            # form target prediction
            # shape is (source points in the example, eg 1000 , num_classes eg 20)
            #example_preds = np.zeros(shape=(num_points_raw,  example_preds_actual.shape[1]), dtype=np.float32)
            
            # fill preds for points thet net outputs, eg 800, than 200 will be remains with zeros
            # for 200 zeroes rows (unfilled, pointas which are not predicted by the net)  argmax(1) will return label 0. (this is bug or вуаусе)
            #example_preds[filtred_mask] = example_preds_actual
            
            # store
            #preds_all.append(example_preds)

            # variant II, Use argmax now
            # form target prediction
            example_y_pred = np.ones(shape=(num_points_raw), dtype=np.int32) * (self.n_classes - 1)
            
            # fill preds for the points that net outputs, eg 800, than 200 will be remains with zeros
            example_y_pred[filtred_mask] = example_preds_actual.argmax(1)

            # store (old)
            #d['predictions_all'].append(example_y_pred)
            #d['gt_all'].append(labels_raw)

            # filter
            indexer = labels_raw >= 0
            
            # accumulate cm of example
            y_pred = example_y_pred[indexer]
            y_true = labels_raw[indexer]
            if len(y_pred) == 0:
                warnings.warn(f"Wrong example is found: all `labels_raw` < 0. Id={xb['id'][k]}")
            else:
                cm = confusion_matrix(y_pred, y_true, self.n_classes)
                d['cm'] += cm

                d['runned'] = True


    def on_epoch_end(self, last_metrics, **kwargs):

        for datatype in ['train', 'valid']:
            d = self._d[datatype]

            if d['runned'] :
                # y_pred = np.hstack(d.pop('predictions_all'))
                # y_true = np.hstack(d.pop('gt_all'))
                # # filter -100 ground true
                # indexer = y_true >= 0
                # cm = confusion_matrix(y_pred[indexer], y_true[indexer], self.n_classes)
                # d['cm'] = cm
    
                cm = d['cm']
                d['iou_per_class'] = iou_per_class_from_cm(cm)
                d['iou'] = np.mean(d['iou_per_class'])
            else:
                d['cm'] = None
                d['iou_per_class'] = None
                d['iou'] = 0

        iou_train = self._d['train']['iou']
        iou_valid = self._d['valid']['iou']

        return add_metrics(last_metrics, [iou_train, iou_valid])

