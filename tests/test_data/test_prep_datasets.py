# Copyright (c) 2024 Houssem Ben Braiek, Emilio Rivera-Landos, IVADO, SEMLA
#
# Permission is hereby granted, free of charge, to any person obtaining a copy
# of this software and associated documentation files (the "Software"), to deal
# in the Software without restriction, including without limitation the rights
# to use, copy, modify, merge, publish, distribute, sublicense, and/or sell
# copies of the Software, and to permit persons to whom the Software is
# furnished to do so, subject to the following conditions:
#
# The above copyright notice and this permission notice shall be included in all
# copies or substantial portions of the Software.
#
# THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
# IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
# FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE
# AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
# LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM,
# OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN THE
# SOFTWARE.

import logging

import numpy as np
import pytest

from bank_marketing.data.prep_datasets import prepare_binary_classfication_tabular_data

logger = logging.getLogger('tests.test_data.test_prep_datasets')


def test_prepare_binary_classfication_tabular_data_X_y_equal_splits(
    dataframe, predictors, predicted
):
    """Test the equality of X and y for the three generated splits.

    This example is provided to you in order to assist with the following test functions.
    """
    dataset = prepare_binary_classfication_tabular_data(
        dataframe, predictors, predicted, pos_neg_pair=('yes', 'no')
    )
    logger.info(
        'prepare_binary_classification_tabular_data was called successfully without errors.'
    )
    for_train = len(dataset.train_x) == len(dataset.train_y)
    if not for_train:
        logger.error(
            'The sizes of feature (X) and label (y) data for the training split are not equal.'
        )
    for_val = len(dataset.val_x) == len(dataset.val_y)
    if not for_val:
        logger.error(
            'The sizes of feature (X) and label (y) data for the validation split are not equal.'
        )
    for_test = len(dataset.test_x) == len(dataset.test_y)
    if not for_test:
        logger.error(
            'The sizes of feature (X) and label (y) data for the testing split are not equal.'
        )
    assert for_train
    assert for_val
    assert for_test


def test_prepare_binary_classfication_tabular_data_correctness(dataframe, predictors, predicted):
    """Test the splits correctness in regards to X and y alignment.

    Checks if each x is associated with its correct y.
    """
    dataframe['gen_id'] = dataframe.index
    response_dict = dict(zip(dataframe['gen_id'], dataframe['curr_outcome']))
    dataset = prepare_binary_classfication_tabular_data(
        dataframe, predictors + ['gen_id'], predicted, None
    )
    logger.info(
        'prepare_binary_classification_tabular_data was called successfully without errors.'
    )
    train_y_true = dataset.train_x.apply(lambda row: response_dict[row['gen_id']], axis=1)
    val_y_true = dataset.val_x.apply(lambda row: response_dict[row['gen_id']], axis=1)
    test_y_true = dataset.test_x.apply(lambda row: response_dict[row['gen_id']], axis=1)
    train_is_aligned = (train_y_true == dataset.train_y).all()
    if not train_is_aligned:
        logger.error('The features (X) and labels (y) for the training split are aligned.')
    val_is_aligned = (val_y_true == dataset.val_y).all()
    if not val_is_aligned:
        logger.error('The features (X) and labels (y) for the validation split are aligned.')
    test_is_aligned = (test_y_true == dataset.test_y).all()
    if not test_is_aligned:
        logger.error('The features (X) and labels (y) for the testing split are aligned.')
    assert train_is_aligned
    assert val_is_aligned
    assert test_is_aligned


@pytest.mark.parametrize('splits_sizes', [[0.7, 0.2, 0.1], [0.7, 0.1, 0.2], [0.2, 0.4, 0.4]])
def test_prepare_binary_classfication_tabular_data_splits_sizes(
    splits_sizes, dataframe, predictors, predicted
):
    """Test the actual splits sizes are conform to the provided ratios."""
    total_rows = len(dataframe)
    expected_absolute_sizes = np.round(total_rows * np.array(splits_sizes))
    dataset = prepare_binary_classfication_tabular_data(
        dataframe, predictors, predicted, splits_sizes=splits_sizes, seed=42
    )
    logger.info(
        'prepare_binary_classification_tabular_data was called successfully without errors.'
    )
    train_diff = abs(expected_absolute_sizes[0] - len(dataset.train_x))
    if train_diff > 1:
        logger.error('The actual size of the training split is not as expected.')
    val_diff = abs(expected_absolute_sizes[1] - len(dataset.val_x))
    if val_diff > 1:
        logger.error('The actual size of the validation split is not as expected.')
    test_diff = abs(expected_absolute_sizes[2] - len(dataset.test_x))
    if test_diff > 1:
        logger.error('The actual size of the testing split is not as expected.')
    assert train_diff <= 1
    assert val_diff <= 1
    assert test_diff <= 1


@pytest.mark.parametrize(('seed', 'row'), [(43, 0), (77, 10), (89, 50)])
def test_prepare_binary_classfication_tabular_data_reproducibility(
    seed, row, dataframe, predictors, predicted
):
    """Test the generated splits with fixed seed are the same.

    Based on an arbitrary row index (argument).
    """
    dataset_1 = prepare_binary_classfication_tabular_data(
        dataframe, predictors, predicted, seed=seed
    )
    dataset_2 = prepare_binary_classfication_tabular_data(
        dataframe, predictors, predicted, seed=seed
    )
    logger.info(
        'prepare_binary_classification_tabular_data was called successfully without errors.'
    )
    train_x_eq = (dataset_1.train_x.iloc[row] == dataset_2.train_x.iloc[row]).all()
    if not train_x_eq:
        logger.error('The training split is not the same with fixed seed.')
    val_x_eq = (dataset_1.val_x.iloc[row] == dataset_2.val_x.iloc[row]).all()
    if not val_x_eq:
        logger.error('The validation split is not the same with fixed seed.')
    test_x_eq = (dataset_1.test_x.iloc[row] == dataset_2.test_x.iloc[row]).all()
    if not test_x_eq:
        logger.error('The testing split is not the same with fixed seed.')
    assert train_x_eq
    assert val_x_eq
    assert test_x_eq


@pytest.mark.parametrize(('seed_1', 'seed_2', 'row'), [(42, 43, 0), (51, 63, 10)])
def test_prepare_binary_classfication_tabular_data_variability(
    seed_1, seed_2, row, dataframe, predictors, predicted
):
    """Test the generated splits with different seeds are actually different.

    Based on an arbitrary row index (argument).
    """
    dataset_1 = prepare_binary_classfication_tabular_data(
        dataframe, predictors, predicted, seed=seed_1
    )
    dataset_2 = prepare_binary_classfication_tabular_data(
        dataframe, predictors, predicted, seed=seed_2
    )
    logger.info(
        'prepare_binary_classification_tabular_data was called successfully without errors.'
    )
    train_x_neq = (dataset_1.train_x.iloc[row] != dataset_2.train_x.iloc[row]).any()
    if not train_x_neq:
        logger.error('The training split is not different when seed is changed.')
    val_x_neq = (dataset_1.val_x.iloc[row] != dataset_2.val_x.iloc[row]).any()
    if not val_x_neq:
        logger.error('The validation split is not different when seed is changed.')
    test_x_neq = (dataset_1.test_x.iloc[row] != dataset_2.test_x.iloc[row]).any()
    if not test_x_neq:
        logger.error('The testing split is not different when seed is changed.')
    assert train_x_neq
    assert val_x_neq
    assert test_x_neq
