#!/usr/bin/env python
# coding: utf-8

import numpy as np
import matplotlib.pyplot as plt
import os
import pandas as pd
from contextlib import redirect_stdout
from datetime import datetime
import config
from src.models.auto_ml.base_auto_ml import BaseModelAutoML
from src.models.helpers.quantile_reg import pinball_loss
from smac.utils.constants import MAXINT
from sklearn.model_selection import GroupShuffleSplit
from sklearn.preprocessing import LabelEncoder

import autosklearn.regression
from autosklearn.metrics import r2
from autosklearn.metrics import mean_squared_error as mse
from autosklearn.metrics import mean_absolute_error as mae
from autosklearn.metrics import median_absolute_error as mabse
from autosklearn.regression import AutoSklearnRegressor as Regressor


class AutoSklearn(BaseModelAutoML):
    def __init__(self, path, name):
        super().__init__(path, name)
        self.model_name = 'autosklearn'

    def define_scorer(self, metric_key, metric):
        if metric[0] == 'quantile':
            self.scorer[metric_key] = autosklearn.metrics.make_scorer(name=f'quantile_reg_{metric[1]}',
                                                                      worst_possible_result=MAXINT,
                                                                      score_func=pinball_loss,
                                                                      quantile=metric[1],
                                                                      greater_is_better=False)
        elif metric == 'mse':
            self.scorer[metric_key] = autosklearn.metrics.mean_squared_error
        elif metric == 'r2':
            self.scorer[metric_key] = autosklearn.metrics.r2
        else:
            self.scorer[metric_key] = None

    def train(self, modus, task_key, metric_key, X_data, y_data):
        self.create_task_dirs(modus, task_key)
        self.create_metric_dirs(modus, task_key, metric_key)
        # --- FRAMEWORK SPECIFIC START --- #1 -------------------------------------

        if self.group_split_col:
            val_strategy = GroupShuffleSplit(n_splits=config.INNER_SPLITS, train_size=.8, random_state=config.SEED)
            le = LabelEncoder()
            self.groups_inner = le.fit_transform(self.data[self.group_split_col].loc[self.train_index])
            val_strategy_arguments = {'groups': self.groups_inner}

        else:
            val_strategy = 'cv'
            val_strategy_arguments = {'folds': config.INNER_SPLITS}

        auto_tmp_folder = os.path.join(self.metric_path[(modus, task_key, metric_key)], 'autosklearn_tmp')

        self.model[(modus, task_key, metric_key)] = Regressor(time_left_for_this_task=config.MAX_TIME_MINUTES * 60,
                                                              per_run_time_limit=config.MAX_TIME_MINUTES * 5,
                                                              resampling_strategy=val_strategy,
                                                              resampling_strategy_arguments=val_strategy_arguments,
                                                              # ensemble_size=2,
                                                              n_jobs=config.NUM_CORES,
                                                              memory_limit=None,
                                                              seed=config.SEED,
                                                              metric=self.scorer[metric_key],
                                                              scoring_functions=[r2, mse, mae, mabse],
                                                              tmp_folder=auto_tmp_folder,
                                                              delete_tmp_folder_after_terminate=True)
        # --- FRAMEWORK SPECIFIC END ----- #1 -------------------------------------

        print(datetime.now())
        print('Training over {} min started'.format(config.MAX_TIME_MINUTES))
        self.model[(modus, task_key, metric_key)].fit(X_data, y_data)

        # Perform refit?
        # self.model[(modus, task_key, metric_key)].refit(X_data, y_data)
        print(datetime.now())

        # --- FRAMEWORK SPECIFIC START --- #2 -------------------------------------
        print(self.model[(modus, task_key, metric_key)].sprint_statistics())

        # plot train hist
        ra = np.arange(len(self.model[(modus, task_key, metric_key)].cv_results_['status']))
        test_score = self.model[(modus, task_key, metric_key)].cv_results_['mean_test_score']
        test_score[test_score < 0] = 0

        best = []
        for i in test_score:
            try:
                best.append(max(0, max(best), i))
            except ValueError:  # best is empty
                best.append(0)
        best = np.array(best)

        labels = []
        for i in self.model[(modus, task_key, metric_key)].cv_results_['params']:
            labels.append(i['regressor:__choice__'])
        labels = np.array(labels)

        df = pd.DataFrame(dict(x=ra, y=test_score, label=labels))
        groups = df.groupby('label')

        fig, ax = plt.subplots(figsize=(15, 6))
        for name, group in groups:
            ax.plot(group.x, group.y, marker='o', linestyle='', ms=6, label=name)
            ax.legend(frameon=True, framealpha=0.9)
            ax.plot(ra, best, color='gray')
            ax.set_title('Algorithm Performance')
            ax.set_ylabel('Score')
            ax.set_xlabel('Model No.')
        plt.savefig(os.path.join(self.metric_path[(modus, task_key, metric_key)], 'train_history.svg'))

        # save pkl, consumes to much storage
        # with open(os.path.join(output_dir, 'clf.pickle'), 'wb') as f:
        # pickle.dump(automl, f, pickle.HIGHEST_PROTOCOL)

        with open(os.path.join(self.metric_path[(modus, task_key, metric_key)], 'stats.txt'), 'w') as f:
            with redirect_stdout(f):
                print(self.model[(modus, task_key, metric_key)].sprint_statistics())

        # with open(os.path.join(output_dir, 'leaderboard.txt'), 'w') as f:
        # b = automl.leaderboard()
        # with redirect_stdout(f):
        # print(lb.to_string())

        with open(os.path.join(self.metric_path[(modus, task_key, metric_key)], 'metric_results.txt'), 'w') as f:
            with redirect_stdout(f):
                print(self.get_metric_result(modus, task_key, metric_key).to_string(index=False))

        with open(os.path.join(self.metric_path[(modus, task_key, metric_key)], 'model.txt'), 'w') as f:
            with redirect_stdout(f):
                print(self.model[(modus, task_key, metric_key)].show_models())

        # save model
        """
        with open(os.path.join(self.split_path, f'autosklearn_model_{self.scorer}'), 'wb') as f:
            pickle.dump(self.model, f)
        """

    def get_metric_result(self, modus, task_key, metric_key):
        results = pd.DataFrame.from_dict(self.model[(modus, task_key, metric_key)].cv_results_)
        # results = results[results['status'] == "Success"]
        cols = ['rank_test_scores', 'param_regressor:__choice__',
                'mean_test_score']
        cols.extend([key for key in self.model[(modus, task_key, metric_key)].cv_results_.keys()
                     if key.startswith('metric_')])
        return results[cols]

        # --- FRAMEWORK SPECIFIC END ----- #2 -------------------------------------
