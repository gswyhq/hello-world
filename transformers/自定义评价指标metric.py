#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import torch
import os
import math
import pandas as pd
import numpy as np
from glob import glob
from tqdm import tqdm
import re
import json
import functools
import itertools
import datasets
from datasets import ClassLabel, load_dataset
from typing import Callable, List, Optional, Tuple, Type, Union

USERNAME = os.getenv("USERNAME")


class Tokens:

    def __init__(self, tokens: List[str], delimiter: str = '-', outside_token = 'O', scheme='BIO'):
        self.outside_token = outside_token
        self.tokens = tokens
        self.delimiter = delimiter
        self.scheme = scheme
        self.extended_tokens = self.tokens + [self.outside_token]

    @property
    def entities(self):
        """Extract entities from tokens.

        Returns:
            list: list of Entity.

        Example:
            >>> tokens = Tokens(['B-PER', 'I-PER', 'O', 'B-LOC'])
            >>> tokens.entities
            [('PER', 0, 2), ('LOC', 3, 4)]

            tokens = Tokens(['LABEL_1', 'LABEL_2', 'LABEL_5', 'LABEL_6', 'LABEL_6', 'LABEL_5', 'LABEL_6', 'LABEL_11', 'LABEL_12', 'LABEL_12', 'LABEL_12', 'LABEL_0', 'LABEL_0', 'LABEL_0'], delimiter= '_', outside_token = 'LABEL_0', scheme='LABEL')
            tokens.entities

        """
        entities = []
        last_token = self.outside_token
        start = -1
        end = -1
        for i, token in enumerate(self.tokens):
            if token == self.outside_token:
                entities.append((last_token, start, end+1))
                start = i
                end = i
                last_token = self.outside_token
                continue
            else:
                tag, token = token.split(self.delimiter)
                if self.scheme=='LABEL':
                    tag = 'B' if int(token)%2 == 1 else 'I'
                    token = (int(token)+1)//2

            if start < 0:
                start = i
                end = i
                last_token = token

            elif last_token == token and tag != 'B':
                end = i
            else:
                entities.append((last_token, start, end+1))
                start = i
                end = i
                last_token = token

            if i + 1 == len(self.tokens):
                entities.append((last_token, start, end+1))
        entities = [t for t in entities if t[0] != self.outside_token]
        return entities

class Entities:

    def __init__(self, sequences: List[List[str]], delimiter: str = '-', outside_token: str = 'O', scheme: str='BIO'):
        self.entities = [
            Tokens(seq, delimiter=delimiter, outside_token = outside_token, scheme=scheme ).entities
            for sent_id, seq in enumerate(sequences)
        ]

    def filter(self, tag_name: str):
        entities = {entity for entity in itertools.chain(*self.entities) if entity[0] == tag_name}
        return entities

    @property
    def unique_tags(self):
        tags = {
            entity[0] for entity in itertools.chain(*self.entities)
        }
        return tags



def _prf_divide(numerator, denominator, metric,
                modifier, average, warn_for, zero_division='warn'):
    """Performs division and handles divide-by-zero.

    On zero-division, sets the corresponding result elements equal to
    0 or 1 (according to ``zero_division``). Plus, if
    ``zero_division != "warn"`` raises a warning.

    The metric, modifier and average arguments are used only for determining
    an appropriate warning.
    """
    mask = denominator == 0.0
    denominator = denominator.copy()
    denominator[mask] = 1  # avoid infs/nans
    result = numerator / denominator

    if not np.any(mask):
        return result

    # if ``zero_division=1``, set those with denominator == 0 equal to 1
    result[mask] = 0.0 if zero_division in ['warn', 0] else 1.0

    # the user will be removing warnings if zero_division is set to something
    # different than its default value. If we are computing only f-score
    # the warning will be raised only if precision and recall are ill-defined
    if zero_division != 'warn' or metric not in warn_for:
        return result

    # build appropriate warning
    # E.g. "Precision and F-score are ill-defined and being set to 0.0 in
    # labels with no predicted samples. Use ``zero_division`` parameter to
    # control this behavior."

    if metric in warn_for and 'f-score' in warn_for:
        msg_start = '{0} and F-score are'.format(metric.title())
    elif metric in warn_for:
        msg_start = '{0} is'.format(metric.title())
    elif 'f-score' in warn_for:
        msg_start = 'F-score is'
    else:
        return result

    _warn_prf(average, modifier, msg_start, len(result))

    return result

def _precision_recall_fscore_support(y_true: List[List[str]],
                                     y_pred: List[List[str]],
                                     *,
                                     average: Optional[str] = None,
                                     warn_for=('precision', 'recall', 'f-score'),
                                     beta: float = 1.0,
                                     sample_weight: Optional[List[int]] = None,
                                     zero_division: str = 'warn',
                                     delimiter: str = '-', outside_token: str = 'O', scheme: str='BIO',
                                     suffix: bool = False,
                                     extract_tp_actual_correct: Callable = None) :
    if beta < 0:
        raise ValueError('beta should be >=0 in the F-beta score')

    average_options = (None, 'micro', 'macro', 'weighted')
    if average not in average_options:
        raise ValueError('average has to be one of {}'.format(average_options))

    # check_consistent_length(y_true, y_pred)

    pred_sum, tp_sum, true_sum = extract_tp_actual_correct(y_true, y_pred, delimiter=delimiter, outside_token=outside_token, scheme=scheme)

    if average == 'micro':
        tp_sum = np.array([tp_sum.sum()])
        pred_sum = np.array([pred_sum.sum()])
        true_sum = np.array([true_sum.sum()])

    # Finally, we have all our sufficient statistics. Divide! #
    beta2 = beta ** 2

    # Divide, and on zero-division, set scores and/or warn according to
    # zero_division:
    precision = _prf_divide(
        numerator=tp_sum,
        denominator=pred_sum,
        metric='precision',
        modifier='predicted',
        average=average,
        warn_for=warn_for,
        zero_division=zero_division
    )
    recall = _prf_divide(
        numerator=tp_sum,
        denominator=true_sum,
        metric='recall',
        modifier='true',
        average=average,
        warn_for=warn_for,
        zero_division=zero_division
    )

    # warn for f-score only if zero_division is warn, it is in warn_for
    # and BOTH prec and rec are ill-defined
    if zero_division == 'warn' and ('f-score',) == warn_for:
        if (pred_sum[true_sum == 0] == 0).any():
            _warn_prf(
                average, 'true nor predicted', 'F-score is', len(true_sum)
            )

    # if tp == 0 F will be 1 only if all predictions are zero, all labels are
    # zero, and zero_division=1. In all other case, 0
    if np.isposinf(beta):
        f_score = recall
    else:
        denom = beta2 * precision + recall

        denom[denom == 0.] = 1  # avoid division by 0
        f_score = (1 + beta2) * precision * recall / denom

    # Average the results
    if average == 'weighted':
        weights = true_sum
        if weights.sum() == 0:
            zero_division_value = 0.0 if zero_division in ['warn', 0] else 1.0
            # precision is zero_division if there are no positive predictions
            # recall is zero_division if there are no positive labels
            # fscore is zero_division if all labels AND predictions are
            # negative
            return (zero_division_value if pred_sum.sum() == 0 else 0.0,
                    zero_division_value,
                    zero_division_value if pred_sum.sum() == 0 else 0.0,
                    sum(true_sum))

    elif average == 'samples':
        weights = sample_weight
    else:
        weights = None

    if average is not None:
        precision = np.average(precision, weights=weights)
        recall = np.average(recall, weights=weights)
        f_score = np.average(f_score, weights=weights)
        true_sum = sum(true_sum)

    return precision, recall, f_score, true_sum


def precision_recall_fscore_support(y_true: List[List[str]],
                                    y_pred: List[List[str]],
                                    *,
                                    average: Optional[str] = None,
                                    warn_for=('precision', 'recall', 'f-score'),
                                    beta: float = 1.0,
                                    sample_weight: Optional[List[int]] = None,
                                    zero_division: str = 'warn',
                                    delimiter: str = '-', outside_token: str = 'O', scheme: str='BIO',
                                    suffix: bool = False,
                                    **kwargs) :
    """Compute precision, recall, F-measure and support for each class.
    Args:
        y_true : 2d array. Ground truth (correct) target values.
        y_pred : 2d array. Estimated targets as returned by a tagger.
        beta : float, 1.0 by default
            The strength of recall versus precision in the F-score.
        average : string, [None (default), 'micro', 'macro', 'weighted']
            If ``None``, the scores for each class are returned. Otherwise, this
            determines the type of averaging performed on the data:
            ``'micro'``:
                Calculate metrics globally by counting the total true positives,
                false negatives and false positives.
            ``'macro'``:
                Calculate metrics for each label, and find their unweighted
                mean.  This does not take label imbalance into account.
            ``'weighted'``:
                Calculate metrics for each label, and find their average weighted
                by support (the number of true instances for each label). This
                alters 'macro' to account for label imbalance; it can result in an
                F-score that is not between precision and recall.
        warn_for : tuple or set, for internal use
            This determines which warnings will be made in the case that this
            function is being used to return only one of its metrics.
        sample_weight : array-like of shape (n_samples,), default=None
            Sample weights.
        zero_division : "warn", 0 or 1, default="warn"
            Sets the value to return when there is a zero division:
               - recall: when there are no positive labels
               - precision: when there are no positive predictions
               - f-score: both
            If set to "warn", this acts as 0, but warnings are also raised.
        scheme : Token, [IOB2, IOE2, IOBES]
        suffix : bool, False by default.
    Returns:
        precision : float (if average is not None) or array of float, shape = [n_unique_labels]
        recall : float (if average is not None) or array of float, , shape = [n_unique_labels]
        fbeta_score : float (if average is not None) or array of float, shape = [n_unique_labels]
        support : int (if average is not None) or array of int, shape = [n_unique_labels]
            The number of occurrences of each label in ``y_true``.
    Examples:
        >>> from seqeval.metrics.v1 import precision_recall_fscore_support
        >>> from seqeval.scheme import IOB2
        >>> y_true = [['O', 'O', 'O', 'B-MISC', 'I-MISC', 'I-MISC', 'O'], ['B-PER', 'I-PER', 'O']]
        >>> y_pred = [['O', 'O', 'B-MISC', 'I-MISC', 'I-MISC', 'I-MISC', 'O'], ['B-PER', 'I-PER', 'O']]
        >>> precision_recall_fscore_support(y_true, y_pred, average='macro', scheme=IOB2)
        (0.5, 0.5, 0.5, 2)
        >>> precision_recall_fscore_support(y_true, y_pred, average='micro', scheme=IOB2)
        (0.5, 0.5, 0.5, 2)
        >>> precision_recall_fscore_support(y_true, y_pred, average='weighted', scheme=IOB2)
        (0.5, 0.5, 0.5, 2)
        It is possible to compute per-label precisions, recalls, F1-scores and
        supports instead of averaging:
        >>> precision_recall_fscore_support(y_true, y_pred, average=None, scheme=IOB2)
        (array([0., 1.]), array([0., 1.]), array([0., 1.]), array([1, 1]))
    Notes:
        When ``true positive + false positive == 0``, precision is undefined;
        When ``true positive + false negative == 0``, recall is undefined.
        In such cases, by default the metric will be set to 0, as will f-score,
        and ``UndefinedMetricWarning`` will be raised. This behavior can be
        modified with ``zero_division``.
    """
    def extract_tp_actual_correct(y_true, y_pred, delimiter: str = '-', outside_token: str = 'O', scheme: str='BIO'):
        # If this function is called from classification_report,
        # try to reuse entities to optimize the function.
        entities_true = kwargs.get('entities_true') or Entities(y_true, delimiter= delimiter, outside_token = outside_token, scheme=scheme)
        entities_pred = kwargs.get('entities_pred') or Entities(y_pred, delimiter= delimiter, outside_token = outside_token, scheme=scheme)
        target_names = sorted(entities_true.unique_tags | entities_pred.unique_tags)

        tp_sum = np.array([], dtype=np.int32)
        pred_sum = np.array([], dtype=np.int32)
        true_sum = np.array([], dtype=np.int32)
        for type_name in target_names:
            entities_true_type = entities_true.filter(type_name)
            entities_pred_type = entities_pred.filter(type_name)
            tp_sum = np.append(tp_sum, len(entities_true_type & entities_pred_type))
            pred_sum = np.append(pred_sum, len(entities_pred_type))
            true_sum = np.append(true_sum, len(entities_true_type))

        return pred_sum, tp_sum, true_sum

    precision, recall, f_score, true_sum = _precision_recall_fscore_support(
        y_true, y_pred,
        average=average,
        warn_for=warn_for,
        beta=beta,
        sample_weight=sample_weight,
        zero_division=zero_division,
        delimiter=delimiter, outside_token=outside_token, scheme=scheme,
        suffix=suffix,
        extract_tp_actual_correct=extract_tp_actual_correct
    )

    return precision, recall, f_score, true_sum


def accuracy_score(y_true, y_pred):
    """Accuracy classification score.

    In multilabel classification, this function computes subset accuracy:
    the set of labels predicted for a sample must *exactly* match the
    corresponding set of labels in y_true.

    Args:
        y_true : 2d array. Ground truth (correct) target values.
        y_pred : 2d array. Estimated targets as returned by a tagger.

    Returns:
        score : float.

    Example:
        >>> from seqeval.metrics import accuracy_score
        >>> y_true = [['O', 'O', 'O', 'B-MISC', 'I-MISC', 'I-MISC', 'O'], ['B-PER', 'I-PER', 'O']]
        >>> y_pred = [['O', 'O', 'B-MISC', 'I-MISC', 'I-MISC', 'I-MISC', 'O'], ['B-PER', 'I-PER', 'O']]
        >>> accuracy_score(y_true, y_pred)
        0.80
    """
    if any(isinstance(s, list) for s in y_true):
        y_true = [item for sublist in y_true for item in sublist]
        y_pred = [item for sublist in y_pred for item in sublist]

    nb_correct = sum(y_t == y_p for y_t, y_p in zip(y_true, y_pred))
    nb_true = len(y_true)

    score = nb_correct / nb_true

    return score


class NerMetric():
    def compute(self, predictions=None, references=None):
        y_true, y_pred = references, predictions
        precision, recall, f_score, true_sum = precision_recall_fscore_support(y_true, y_pred, average='macro', scheme=None)
        return {
                "overall_precision": precision,
                "overall_recall": recall,
                "overall_f1": f_score,
                "overall_accuracy": accuracy_score(y_true, y_pred),
            }

label2desc = {'LABEL_0': '其他',
 'LABEL_2': '省份',
 'LABEL_4': '城市',
 'LABEL_6': '区县',
 'LABEL_8': '乡镇街道',
 'LABEL_10': '乡村社区',
 'LABEL_12': '核心信息点', # 公司企业、事业单位、学校、地铁站、楼宇大厦、购物广场等
 'LABEL_14': '道路道路号'}

label_dict = {'LABEL_0': '其他',
'LABEL_1': '省份-头',
'LABEL_2': '省份-其余部分',
'LABEL_3': '城市-头',
'LABEL_4': '城市-其余部分',
'LABEL_5': '区县-头',
'LABEL_6': '区县-其余部分',
'LABEL_7': '乡镇街道-头',
'LABEL_8': '乡镇街道-其余部分',
'LABEL_9': '乡村社区-头',
'LABEL_10': '乡村社区-其余部分',
'LABEL_11': '核心信息点-头',
'LABEL_12': '核心信息点-其余部分',
'LABEL_13': '道路道路号-头',
'LABEL_14': '道路道路号-其余部分', }

def main():
    y_true = [['O', 'O', 'O', 'B-MISC', 'I-MISC', 'I-MISC', 'O'], ['B-PER', 'I-PER', 'O']]
    y_pred = [['O', 'O', 'B-MISC', 'I-MISC', 'I-MISC', 'I-MISC', 'O'], ['B-PER', 'I-PER', 'O']]
    precision, recall, f_score, true_sum = precision_recall_fscore_support(y_true, y_pred, average='macro', scheme='')
    print(precision, recall, f_score, true_sum)

    # import evaluate
    # metric = evaluate.load("seqeval")
    metric = NerMetric()
    results = metric.compute(predictions=y_pred, references=y_true)
    print(results)

    #
    y_pred = [['LABEL_3', 'LABEL_4', 'LABEL_11', 'LABEL_12', 'LABEL_12', 'LABEL_12'], ['LABEL_1', 'LABEL_2', 'LABEL_2', 'LABEL_3', 'LABEL_4', 'LABEL_4', 'LABEL_5', 'LABEL_8', 'LABEL_7', 'LABEL_8', 'LABEL_8', 'LABEL_8', 'LABEL_13', 'LABEL_14', 'LABEL_14', 'LABEL_14', 'LABEL_14', 'LABEL_11', 'LABEL_12', 'LABEL_12', 'LABEL_12'], ['LABEL_1', 'LABEL_2', 'LABEL_5', 'LABEL_6', 'LABEL_6', 'LABEL_5', 'LABEL_6', 'LABEL_11', 'LABEL_12', 'LABEL_12', 'LABEL_12', 'LABEL_0', 'LABEL_0', 'LABEL_0']]
    y_true = [['LABEL_3', 'LABEL_4', 'LABEL_11', 'LABEL_12', 'LABEL_12', 'LABEL_12'], ['LABEL_1', 'LABEL_2', 'LABEL_2', 'LABEL_3', 'LABEL_4', 'LABEL_4', 'LABEL_5', 'LABEL_6', 'LABEL_7', 'LABEL_8', 'LABEL_8', 'LABEL_8', 'LABEL_13', 'LABEL_14', 'LABEL_14', 'LABEL_14', 'LABEL_14', 'LABEL_11', 'LABEL_12', 'LABEL_12', 'LABEL_12'], ['LABEL_1', 'LABEL_2', 'LABEL_5', 'LABEL_6', 'LABEL_6', 'LABEL_7', 'LABEL_8', 'LABEL_11', 'LABEL_12', 'LABEL_12', 'LABEL_12', 'LABEL_0', 'LABEL_0', 'LABEL_0']]

    precision, recall, f_score, true_sum = precision_recall_fscore_support(y_true, y_pred, average='macro', delimiter= '_', outside_token = 'LABEL_0', scheme='LABEL')
    print(precision, recall, f_score, true_sum)

if __name__ == '__main__':
    main()
