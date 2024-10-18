#!/usr/bin/env python
# coding: utf-8

import config
import os
import pandas as pd
import numpy as np
from contextlib import redirect_stdout
import pickle
import collections
from src.models.ModelBase import ModelBase


class BaseModelAutoML(ModelBase):
    def __init__(self, path, name):
        super().__init__(path, name)
        self.model_class = 'ML-MODEL'
        self.dirs = None
        self.files = None
        self.data = None
        self.X_train = pd.DataFrame()
        self.X_test = pd.DataFrame()
        self.y_train = pd.DataFrame()
        self.y_test = pd.DataFrame()
        self.task_path = collections.defaultdict(dict)
        self.metric_path = collections.defaultdict(dict)
        self.metric = config.METRICS
        self.scorer = collections.defaultdict(dict)
        self.model = collections.defaultdict(dict)

    def create_task_dirs(self, modus, task_key):
        key = (modus, task_key)
        self.task_path[key] = os.path.join(self.split_path, modus, task_key)
        if not os.path.exists(self.task_path[key]):
            os.makedirs(self.task_path[key])

    def create_metric_dirs(self, modus, task_key, metric_key, *args):
        key = (modus, task_key, metric_key)
        self.metric_path[key] = os.path.join(self.task_path[(modus, task_key)], metric_key)
        if not os.path.exists(self.metric_path[key]):
            os.makedirs(self.metric_path[key])

    def train_stl(self, y_label, metric_key):
        # Train STL
        print('Dataset-Size')
        self.train('STL', y_label, metric_key,
                   self.X_train[self.y_train[y_label].notna()], self.y_train[y_label].dropna())
        # self.save_model('STL', y_label)

    def predict_eval(self, metric_key):
        # Function for the evaluation (if nessecary)
        self.X_test, self.y_test = self.X_data.loc[self.test_index], self.y_data.loc[self.test_index]
        preds = pd.DataFrame(index=self.y_test.index)
        # Pred STL
        for y_label in self.y_label:
            preds[f'{y_label}_STL'] = self.model[('STL', y_label, metric_key)].predict(self.X_test)

        preds.replace([np.inf, -np.inf], np.nan, inplace=True)
        preds.fillna(preds.mean(), inplace=True)
        return preds

    def evaluate(self, metric_key):
        # Function which trains and test the model on a single split to evaluate the peformance
        self.X_train, self.y_train = self.X_data.loc[self.train_index], self.y_data.loc[self.train_index]
        for y_label in self.y_label:
            self.train_stl(y_label, metric_key)

        y_pred = self.predict_eval(metric_key)
        y_pred.to_csv(os.path.join(self.split_path, f'preds_{metric_key}.csv'), index=None)
        return y_pred

    def predict(self, x):
        # Function which output a prediction of all tasks [1, 2, ..., n]
        # start_time = time.time()
        y_pred = []
        x = pd.DataFrame([x], columns=self.X_cols, dtype=float)
        # elapsed_time = time.time() - start_time
        # print(f"Execution time DF: {elapsed_time} seconds")
        # start_time = time.time()
        for y in self.y_label:
            y_pred_task = self.model[('STL', y, 'mean')].predict(x)
            y_pred.append(y_pred_task.item())

        # Calculate and print the elapsed time
        # elapsed_time = time.time() - start_time
        # print(f"Execution time Pred: {elapsed_time} seconds")
        return y_pred

    def save_exp_def(self):
        with open(os.path.join(self.output_dir, 'experiment_def.txt'), 'w') as f:
            with redirect_stdout(f):
                print('INNER_SPLITS = ', config.INNER_SPLITS)
                print('OUTER_SPLITS = ', config.OUTER_SPLITS)
                print('OUTER_SPLITS_MODE = ', config.SPLIT_MODE)
                print('SPLIT_OPTIONS = ', config.SPLIT_OPTIONS)
                print('NUM_CORES = ', config.NUM_CORES)
                print('MAX_TIME = ', config.MAX_TIME_MINUTES)
                print('MTL_LIST = ', config.MTL_LIST)
                print('SEED = ', config.SEED)

    def save_model(self):
        pickle.dump(self, open(os.path.join(self.output_dir, 'model.pkl'), 'wb'))
        return

    def save_results(self):
        return
