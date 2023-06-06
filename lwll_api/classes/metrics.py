# Copyright (c) 2023 California Institute of Technology ("Caltech"). U.S.
# Government sponsorship acknowledged.
# All rights reserved.
# Redistribution and use in source and binary forms, with or without
# modification, are permitted provided that the following conditions are
# met:
# * Redistributions of source code must retain the above copyright notice, this
#   list of conditions and the following disclaimer.
# * Redistributions in binary form must reproduce the above copyright notice,
#   this list of conditions and the following disclaimer in the documentation
#   and/or other materials provided with the distribution.
# * Neither the name of Caltech nor its operating division, the Jet Propulsion
#   Laboratory, nor the names of its contributors may be used to endorse or
#   promote products derived from this software without specific prior written
#   permission.
# THIS SOFTWARE IS PROVIDED BY THE COPYRIGHT HOLDERS AND CONTRIBUTORS "AS IS"
# AND ANY EXPRESS OR IMPLIED WARRANTIES, INCLUDING, BUT NOT LIMITED TO, THE
# IMPLIED WARRANTIES OF MERCHANTABILITY AND FITNESS FOR A PARTICULAR PURPOSE
# ARE DISCLAIMED. IN NO EVENT SHALL THE COPYRIGHT OWNER OR CONTRIBUTORS BE
# LIABLE FOR ANY DIRECT, INDIRECT, INCIDENTAL, SPECIAL, EXEMPLARY, OR
# CONSEQUENTIAL DAMAGES (INCLUDING, BUT NOT LIMITED TO, PROCUREMENT OF
# SUBSTITUTE GOODS OR SERVICES; LOSS OF USE, DATA, OR PROFITS; OR BUSINESS
# INTERRUPTION) HOWEVER CAUSED AND ON ANY THEORY OF LIABILITY, WHETHER IN
# CONTRACT, STRICT LIABILITY, OR TORT (INCLUDING NEGLIGENCE OR OTHERWISE)
# ARISING IN ANY WAY OUT OF THE USE OF THIS SOFTWARE, EVEN IF ADVISED OF THE
# POSSIBILITY OF SUCH DAMAGE.

from typing import Dict, Iterable, List, Optional, Tuple, Any

import numpy as np
import pandas as pd
import sacrebleu
from nltk.translate.bleu_score import corpus_bleu
from sklearn.metrics import (
    balanced_accuracy_score,
    log_loss,
    precision_recall_fscore_support,
    roc_auc_score,
    top_k_accuracy_score,
)

from lwll_api.classes.obj_detection_metrics import (
    format_obj_detection_data,
    mAP_ranges,
    mean_average_precision,
)
from lwll_api.utils.logger import get_module_logger

"""
File to house metric calculation code

"""

log = get_module_logger(__name__)

def accuracy(df: pd.DataFrame, actuals: pd.DataFrame) -> float:
    """
    Args:
        df: pd.DataFrame, The performer's predictions
        actuals: pd.DataFrame, actual labels
    Return:
        float
    """
    _validate_input_ids_one_to_one(df, actuals)
    df['id'] = df['id'].astype(str)
    _df = df.set_index('id')
    _actuals = actuals.set_index('id')
    joined = pd.merge(_df, _actuals, left_index=True, right_index=True)
    log.debug(f"joined looks like: {joined.head()}")
    joined.iloc[:, 0] = joined.iloc[:, 0].astype(str)
    joined.iloc[:, 1] = joined.iloc[:, 1].astype(str)
    acc = float(sum(joined.iloc[:, 0] == joined.iloc[:, 1])) / len(joined)
    return acc

def top_5_accuracy(y_true: np.array, preds: np.array) -> float:
    """
    Args:
        y_true: {np.array}, must be 1-D (Actually this is an array-like type
        preds: must be 1-D
    Return:
        float
    """
    top_5 = top_k_accuracy_score(y_true, preds.drop('id', axis=1).values, k=5)
    return float(top_5)

def roc_auc(y_true: np.array, preds: np.array) -> float:
    """
    One-vs-rest computes the AUC of each class against the rest. Sensitive to class imbalance

    Args:
        y_true: {np.array}, must be 1-D (Actually this is an array-like type
        preds: must be 1-D
    Return:
        float
    """
    # roc = roc_auc_score(y_true, preds, multi_class='ovr', average="macro")
    roc = roc_auc_score(y_true, preds.drop('id', axis=1).values, multi_class='ovr', average="macro")
    return float(roc)


def recall(
    y_true: np.array,
    y_pred: np.array,
    labels: Optional[List[str]] = None,
    average: Optional[str] = None,
) -> float:
    """Compute recall for each class.
    The recall is the ratio ``tp / (tp + fn)`` where ``tp`` is the number of
    true positives and ``fn`` the number of false negatives. The recall is
    intuitively the ability of the classifier to find all the positive samples.

    Parameters
    ----------
    y_true : 1d array-like, or label indicator array / sparse matrix
        Ground truth (correct) target values.
    y_pred : 1d array-like, or label indicator array / sparse matrix
        Estimated targets as returned by a classifier.
    labels : array-like, default=None
        The set of labels to include when ``average != 'binary'``, and their
        order if ``average is None``. Labels present in the data can be
        excluded, for example to calculate a multiclass average ignoring a
        majority negative class, while labels not present in the data will
        result in 0 components in a macro average. For multilabel targets,
        labels are column indices. By default, all labels in ``y_true`` and
        ``y_pred`` are used in sorted order.
    average : {'binary', 'micro', 'macro', 'samples','weighted'}, \
            default=None
        If ``None``, the scores for each class are returned. Otherwise, this
        determines the type of averaging performed on the data:
        ``'binary'``:
            Only report results for the class specified by ``pos_label``.
            This is applicable only if targets (``y_{true,pred}``) are binary.
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
        ``'samples'``:
            Calculate metrics for each instance, and find their average (only
            meaningful for multilabel classification where this differs from
            :func:`accuracy_score`).
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

    Returns
    -------
    recall : float (if average is not None) or array of float, shape =\
        [n_unique_labels]
    Notes
    -----
    When ``true positive + false negative == 0``, recall is undefined.
    In such cases, by default the metric will be set to 0, and
    ``UndefinedMetricWarning`` will be raised. This behavior can be
    modified with ``zero_division``.
    """
    _, recall, _, _ = precision_recall_fscore_support(
        y_true, y_pred, labels=labels, average=average
    )
    # note that recall may be an array of floats if average=None
    return float(recall)

def mAP(df: pd.DataFrame, actuals: pd.DataFrame) -> float:
    """
    Args:
        df: pd.DataFrame, The performer's predictions
        actuals: pd.DataFrame, actual labels
    Return:
        float
    """
    _validate_input_ids_many_to_many(df, actuals)
    _df = format_obj_detection_data(df)
    _actuals = format_obj_detection_data(actuals)
    mAP = mean_average_precision(_actuals, _df)
    return mAP


def mAP_range(df: pd.DataFrame, actuals: pd.DataFrame) -> Dict:
    """
    Args:
        df: pd.DataFrame, The performer's predictions
        actuals: pd.DataFrame, actual labels
    Return:
        dict: nested dictionary with precision level as the top key, class name as a subkey, and AP as value
    """
    _validate_input_ids_many_to_many(df, actuals)
    _df = format_obj_detection_data(df)
    _actuals = format_obj_detection_data(actuals)
    ranges = mAP_ranges(_actuals, _df, 0.5, 1, 0.05)
    return ranges


def cross_entropy_logloss(y_true: np.array, preds: np.array) -> float:
    """
    Args:
        y_true: {np.array}, must be 1-D (Actually this is an array-like type
        preds: must be 1-D
    Return:
        float
    """
    return float(log_loss(y_true, preds))

def weighted_accuracy(y_true: np.array, y_pred: np.array) -> float:
    """
    Computes class weighted accuracy.
    TODO update type hinting to ArrayLike
    Args:
        y_true: {np.array}, must be 1-D (Actually this is an array-like type
        y_pred: must be 1-D
    Return:
        float
    """
    return float(balanced_accuracy_score(y_true, y_pred))

def brier_score(y_true: np.array, preds: np.array) -> float:
    """
    Args:
        y_true: {np.array}, must be 1-D (Actually this is an array-like type
        preds: must be 1-D
    Return:
        float
    """
    return float(np.mean(np.sum((preds - y_true)**2, axis=1)))

def bleu(df: pd.DataFrame, actuals: pd.DataFrame) -> List[float]:
    """
    Old implementation of using nltk bleu score. This has been deprecated since it has been
    advised sacrebleu is much better: https://gitlab.lollllz.com/lwll/lwll_api/-/issues/80#3-bleu
    Args:
        df: pd.DataFrame, The performer's predictions
        actuals: pd.DataFrame, actual labels
    Return:
        List[float]
    """
    _validate_input_ids_one_to_one(df, actuals)
    actual_series = actuals['text'].apply(lambda x: [x.split()])
    predicted_series = df['text'].apply(lambda x: x.split())
    bleu_1 = corpus_bleu(actual_series, predicted_series, weights=(1.0, 0, 0, 0))
    bleu_2 = corpus_bleu(actual_series, predicted_series, weights=(0.5, 0.5, 0, 0))
    bleu_3 = corpus_bleu(actual_series, predicted_series, weights=(0.3, 0.3, 0.3, 0))
    bleu_4 = corpus_bleu(actual_series, predicted_series, weights=(0.25, 0.25, 0.25, 0.25))
    log.info('BLEU-1: %f' % bleu_1)
    log.info('BLEU-2: %f' % bleu_2)
    log.info('BLEU-3: %f' % bleu_3)
    log.info('BLEU-4: %f' % bleu_4)
    return [bleu_1, bleu_2, bleu_3, bleu_4]

def bleu_sacre(df: pd.DataFrame, actuals: pd.DataFrame) -> Tuple[float, str]:
    """
    Implemention of bleu using sacrebleu as it is more accepted in MT than the nltk version
    https://gitlab.lollllz.com/lwll/lwll_api/-/issues/80#3-bleu

    Example 'refs' and 'sys' for sacrebleu calculation
    refs = [['The dog bit the man.', 'It was not unexpected.', 'The man bit him first.']]
    sys = ['The dog bit the man.', "It wasn't surprising.", 'The man had just bitten him.']

    Args:
        df: pd.DataFrame, The performer's predictions
        actuals: pd.DataFrame, actual labels
    Return:
        Tuple[float, str]
    """
    _validate_input_ids_one_to_one(df, actuals)
    df['_id'] = df['id'].apply(lambda x: int(x))
    actuals['_id'] = actuals['id'].apply(lambda x: int(x))
    _combined = df.merge(actuals, left_on='_id', right_on='_id')

    sys = _combined['text_x'].tolist()
    refs = [_combined['text_y'].tolist()]
    # log.info(f"sys:: {sys[:5]}")
    # log.info(f"refs::{refs[0][:5]}")
    bleu = sacrebleu.corpus_bleu(sys, refs)
    score: float = bleu.score
    bleu_str: str = str(bleu)
    return score, bleu_str


def precision_at_k(actuals: np.array, df: pd.DataFrame, k: int = 50, output: str = 'str') -> Any:
    """
    Args:
        df: pd.DataFrame, The performer's predictions
        actuals: pd.DataFrame, actual labels
        k: int, top k predictions to use, default 50.
    Return:
        pd.DataFrame
    """
    result = []

    df['id'] = df['id'].astype(str)
    df = df.sort_values(by=['id']).drop('id', axis=1)

    for i in range(0, len(df.columns)):
        label = df.columns[i]
        p = df[label].to_numpy()
        p_k = (actuals[np.argsort(-p)[:k]] == label).mean()
        result.append({"class": label, f"p_{k}": p_k})
    result = pd.DataFrame(result)

    if output == 'str':
        return result.to_json(orient="table")  # type: ignore
    else:
        return result


def average_precision_at_50(actuals: np.array, df: pd.DataFrame, k: int = 50) -> float:
    # """
    # Args:
    #     df: pd.DataFrame, The performer's predictions
    #     actuals: pd.DataFrame, actual labels
    #     k: int, top k predictions to use, default 50.
    # Return:
    #     float
    # """

    score = precision_at_k(actuals, df, k, output='num')
    score = score[score['class'] != '__none__'][f'p_{k}'].mean()
    return float(score)


def _align_ids(df: pd.DataFrame, actuals: pd.DataFrame) -> Tuple[np.array, np.array]:
    """
    Args:
        df: pd.DataFrame, The performer's predictions
        actuals: pd.DataFrame, actual labels
    Return:
        Tuple[np.array, np.array]
    """
    # TODO: change to use df merge
    # merged_df = df.merge(actuals, on="id", how="outer", suffixes=("_pred", "_actual"))
    # y_true = merged_df["class_actual"].values
    # preds = merged_df["class_pred"].values
    df['id'] = df['id'].astype(str)
    df_sort = df.sort_values(by=['id'])
    actual_sort = actuals.sort_values(by=['id'])
    _df = df_sort.set_index('id')
    _actuals = actual_sort.set_index('id')

    preds = _df.to_numpy()
    y_true = _actuals['class'].to_numpy()
    return y_true, preds

def _preprocess_df_to_array(df: pd.DataFrame, actuals: pd.DataFrame, classes: Optional[List[str]]) -> Tuple[np.array, np.array]:
    """
    Args:
        df: pd.DataFrame, The performer's predictions
        actuals: pd.DataFrame, actual labels
        classes: List, list of classes
    Return:
        Tuple[np.array, np.array]
    """
    df['id'] = df['id'].astype(str)
    df_sort = df.sort_values(by=['id'])
    actual_sort = actuals.sort_values(by=['id'])

    # One-hot encode the classes
    if "class" in df.columns:
        df_sort = pd.get_dummies(df_sort, columns=['class'], prefix='', prefix_sep='')
    actual_sort = pd.get_dummies(actual_sort, columns=['class'], prefix='', prefix_sep='')

    _df = df_sort.set_index('id')
    _actuals = actual_sort.set_index('id')

    # Add missing columns to predictions and actuals for test set
    if classes:
        preds_missing_cols = list(set(classes).difference(set(_df.columns)))
        if preds_missing_cols:
            for col in preds_missing_cols:
                _df[col] = 0.0
    if classes:
        actuals_missing_cols = list(set(classes).difference(set(_actuals.columns)))
        if actuals_missing_cols:
            for col in actuals_missing_cols:
                _actuals[col] = 0.0

    # Sort the columns and make sure that they are all the same
    _df = _df.reindex(sorted(_df.columns), axis=1)
    _actuals = _actuals.reindex(sorted(_actuals.columns), axis=1)
    _df = _df.loc[_df.index.isin(_actuals.index)][_actuals.columns]
    assert (_df.columns == _actuals.columns).all()

    # Convert to numpy and check that all rows sum to 1
    preds = _df.to_numpy()
    y_true_oh = _actuals.to_numpy()

    # This is not always true with ZSL since the actuals might be a subset of all classes
    # assert np.allclose(preds.sum(axis=1), np.ones(preds.shape[0]))
    # assert np.array_equal(y_true_oh.sum(axis=1), np.ones(y_true_oh.shape[0]))

    return y_true_oh, preds

def _probabilistic_to_hard_assignment(df: pd.DataFrame) -> pd.DataFrame:
    """
    Args:
        df: pd.DataFrame, The performer's predictions
    Return:
        pd.DataFrame
    """
    # change df and actuals to be like old format
    class_cols = df.drop(columns='id').columns
    if "video_id" in df.columns:
        class_cols = df.drop(columns=["id", "video_id", "end_frame", "start_frame"]).columns
    class_col = pd.Series(df.loc[:, class_cols].idxmax(axis=1), name="class")
    reformat_df = pd.merge(class_col, df['id'], left_index=True, right_index=True)
    # print(f"reformat_df looks like: {reformat_df.head()}")
    return reformat_df

def _validate_input_ids_one_to_one(df: pd.DataFrame, actuals: pd.DataFrame) -> None:
    """
    Args:
        df: pd.DataFrame, The performer's predictions
        actuals: pd.DataFrame, actual labels
    """
    # Check lengths of dataframes are the same
    log.debug('Validating input ids one to one relationship...')
    pred_ids = df['id'].tolist()
    actual_ids = actuals['id'].tolist()
    set_difference = set(pred_ids).difference(set(actual_ids))
    log.debug(f"set difference is: {set_difference}")
    log.debug(f"{len(set_difference)} ids are different, {len(pred_ids)} preds, {len(actual_ids)} actuals")
    if len(df['id']) != len(actuals['id']):
        raise Exception(f"Hitting condition: `len(df[{'id'}]) != len(actuals[{'id'}])`\nThis probably means that you are missing some test ids")
    # Check all test labels are accounted for in one to one relationship
    if len(set_difference) != 0:
        raise Exception(
            f"Hitting condition: `len(set(df[{'id'}].tolist()).difference(set(actuals[{'id'}].tolist()))) != 0`\nThis probably means that you are \
            missing some test ids")
    return

def _validate_input_ids_many_to_many(df: pd.DataFrame, actuals: pd.DataFrame) -> None:
    """
    Args:
        df: pd.DataFrame, The performer's predictions
        actuals: pd.DataFrame, actual labels
    """
    # Check all test labels are accounted for in a possible many to many relationship
    if len(set(df['id'].unique().tolist()).difference(set(actuals['id'].unique().tolist()))) != 0:
        raise Exception(
            "Hitting condition: `len(set(df['id'].unique().tolist()).difference(set(actuals['id'].unique().tolist()))) != 0`\nThis \
            probably means that you are missing some test ids")
    return


# define a one-hot encode function that takes in a variable number of arrays (e.g., y_true and y_pred)
# and one-hot encodes each array based on the prescribed classes
def one_hot_encode(*arrays: np.array, classes: List[str]) -> Iterable[np.array]:
    one_hot_encoded_arrays = []
    for array in arrays:
        y = np.array([classes.index(y_) for y_ in array])
        one_hot_array = np.zeros((len(y), len(classes)))
        one_hot_array[np.arange(len(y)), y] = 1
        one_hot_encoded_arrays.append(one_hot_array)
    return tuple(one_hot_encoded_arrays)
