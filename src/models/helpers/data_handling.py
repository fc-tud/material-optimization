#!/usr/bin/env python
# coding: utf-8

import os
import numpy as np
import math

import pandas as pd
from sklearn.model_selection import ShuffleSplit, GroupShuffleSplit
from sklearn.preprocessing import LabelEncoder
from sklearn.decomposition import PCA
import sklearn.metrics as metrics
from scipy.spatial.distance import cdist, pdist

import config
from src.models.helpers.quantile_reg import pinball_loss


def regression_results(y_true, y_pred, quantile=0.5):
    explained_variance = metrics.explained_variance_score(y_true, y_pred)
    mean_absolute_error = metrics.mean_absolute_error(y_true, y_pred)
    mse = metrics.mean_squared_error(y_true, y_pred)
    mape = metrics.mean_absolute_percentage_error(y_true, y_pred)
    r2 = metrics.r2_score(y_true, y_pred)
    quantile_score = pinball_loss(y_true, y_pred, quantile=quantile)
    print(f'quantile({quantile}) score:', round(quantile_score, 4))
    print('explained_variance: ', round(explained_variance, 4))
    print('r2: ', round(r2, 4))
    print('MAE: ', round(mean_absolute_error, 4))
    print('MAPE: ', round(mape, 4))
    print('MSE: ', round(mse, 4))
    print('RMSE: ', round(np.sqrt(mse), 4))

    return [quantile_score, r2, np.sqrt(mse), mean_absolute_error, mse, mape]


def check_dataset(path):
    try:
        files = [f for f in os.listdir(path) if os.path.isfile(os.path.join(path, f))]
    except NotADirectoryError:
        print(path, 'is not a dataset directory!')
        files = []

    return files


def is_pareto_efficient(costs, direction):
    """
    Find the Pareto efficient solutions from a 2D array of costs.

    Parameters:
        costs (numpy.ndarray): 2D array where each row represents the costs of a solution.
        direction (list): List of 1s and -1s indicating whether each objective should be minimized (1) or maximized (-1).

    Returns:
        numpy.ndarray: Boolean array indicating whether each solution is Pareto efficient.
    """
    is_efficient = np.ones(costs.shape[0], dtype=bool)
    for i, c in enumerate(costs):
        if is_efficient[i]:
            is_efficient[is_efficient] = np.any((costs[is_efficient] * direction) < (c * direction), axis=1)
            is_efficient[i] = True  # Keep self
    return is_efficient


def create_rand_splits(model, num_split):
    splits = []
    if model.group_split_col:
        rs = GroupShuffleSplit(n_splits=num_split, train_size=model.train_size,
                               random_state=config.SEED)
    else:
        rs = ShuffleSplit(n_splits=num_split, train_size=model.train_size, random_state=config.SEED)
    rs.get_n_splits(model.X_data)
    for train_index, test_index in rs.split(model.X_data, groups=model.groups_outer):
        splits.append([train_index, test_index])
    return splits


def create_pca_splits(model, num_split):
    splits = []
    for split in range(num_split):
        model.read_data()
        # model.X_data = model.X_data.sample(math.ceil(model.train_size / 0.8), random_state=split)
        model.train_index = model.X_data.index
        model.scale()
        pca = PCA(n_components=0.9, svd_solver='full')
        pca_result = pca.fit_transform(model.X)
        pca_dist = cdist(pca_result, pca_result, 'euclidean')
        pca_dist = np.partition(pca_dist, 10)
        pca_dist = np.sum(pca_dist[:, :10], axis=1)
        model.X['PCA'] = pca_dist
        # Calculate the threshold value for the top 20%
        threshold = model.X['PCA'].quantile(0.8)
        train_index = model.X[model.X['PCA'] <= threshold].index
        print('LEN TRAIN-INDEX:', len(train_index))
        test_index = model.X_data.index.difference(train_index)
        print('LEN TEST-INDEX:', len(test_index))
        splits.append([train_index, test_index])
    model.read_data()
    return splits


def create_extra_splits(model, num_split):
    # Optional kann noch ein split option dict in die constants, wo man das y_label für den
    options = config.SPLIT_OPTIONS['extra_split']
    # extra split definieren kann, damit auch die anderen Labels bei SIM-PAN verwendet werden können
    splits = []
    for split in range(num_split):
        # y = model.y_data[model.y_label[-1]].sample(math.ceil(model.train_size / 0.8), random_state=split)
        y = model.y_data[model.y_label[options['label']]]
        lower_threshold, upper_threshold = y.quantile(0.1), y.quantile(0.9)
        train_index = y[y.between(lower_threshold, upper_threshold, inclusive='both')].index
        test_index = y.index.difference(train_index)
        splits.append([train_index, test_index])
    return splits


def create_pareto_splits(model, num_split):
    splits = []
    for split in range(num_split):
        # y = model.y_data[model.y_label].sample(math.ceil(model.train_size / 0.8), random_state=split)
        y = model.y_data[model.y_label]
        if 'TRC' in model.name:
            y['area'] = model.data['area']
            direction = [-1, 1]
        if 'PAN' in model.name:
            direction = [1, 1, 1]
        test_index = pd.Index([], dtype='int64')
        train_index = y.index
        while len(train_index) > model.train_size:
            costs = np.array(y)
            pareto_index = is_pareto_efficient(costs, direction=direction)
            pareto_index = y[pareto_index].index
            if len(train_index) - len(pareto_index) < model.train_size:
                rest_length = len(train_index) - model.train_size
                pareto_index = pd.Index(pd.Series(pareto_index.to_list()).sample(n=rest_length))
            test_index = test_index.append(pareto_index)
            y = y.drop(pareto_index, axis=0)
            train_index = y.index.difference(test_index)
        test_index = sorted(test_index)
        splits.append([train_index, test_index])
    return splits


# dictionary mapping strategy names to their respective functions
splitting_functions = {
    'random': create_rand_splits,
    'PCA': create_pca_splits,
    'extra': create_extra_splits,
    'pareto': create_pareto_splits
}
