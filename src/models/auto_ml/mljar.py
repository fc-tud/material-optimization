#!/usr/bin/env python
# coding: utf-8

import os
from datetime import datetime
import json
import shutil

from src.models.auto_ml.base_auto_ml import BaseModelAutoML
from src.models.helpers.quantile_reg import pinball_loss, partial
import config

from sklearn.model_selection import GroupShuffleSplit
from supervised import AutoML


class MLjar(BaseModelAutoML):
    def __init__(self, path, name):
        super().__init__(path, name)
        self.model_name = 'MLjar'

    def define_scorer(self, metric_key, metric):
        if metric[0] == 'quantile':
            quantile = metric[1]
            self.scorer[metric_key] = partial(pinball_loss, quantile=quantile)
            # scorer = pinball_loss
        elif metric == 'mse':
            self.scorer[metric_key] = 'mse'
        elif metric == 'r2':
            self.scorer[metric_key] = 'r2'
        else:
            self.scorer[metric_key] = 'auto'

    def train(self, modus, task_key, metric_key, X_data, y_data):
        self.create_task_dirs(modus, task_key)
        self.create_metric_dirs(modus, task_key, metric_key)
        # --- FRAMEWORK SPECIFIC START --- #1 -------------------------------------

        if self.group_split_col:
            val_strategy = {"validation_type": "custom"}
            gss = GroupShuffleSplit(n_splits=constants.INNER_SPLITS, train_size=.8, random_state=constants.SEED)
            val_indices = []
            for i, (train_index, test_index) in enumerate(gss.split(X_data, groups=self.groups_inner)):
                val_indices.append((train_index, test_index))
        else:
            val_strategy = {'validation_type': 'kfold',
                            'k_folds': config.INNER_SPLITS,
                            "shuffle": True,
                            'random_seed': config.SEED}

        self.model[(modus, task_key, metric_key)] = AutoML(mode='Compete',
                                                           ml_task='regression',
                                                           total_time_limit=config.MAX_TIME_MINUTES*60,
                                                           validation_strategy=val_strategy,
                                                           n_jobs=config.NUM_CORES,
                                                           random_state=config.SEED,
                                                           eval_metric=self.scorer[metric_key],
                                                           results_path=self.metric_path[(modus, task_key, metric_key)]
                                                           )

        # --- FRAMEWORK SPECIFIC END ----- #1 -------------------------------------

        print(datetime.now())
        print('Training over {} min started'.format(constants.MAX_TIME_MINUTES))
        if self.group_split_col:
            self.model[(modus, task_key, metric_key)].fit(X_data, y_data, cv=val_indices)
        else:
            self.model[(modus, task_key, metric_key)].fit(X_data, y_data)
        print(datetime.now())
        self.clean_dir()
        # save model
        """
        with open(os.path.join(self.split_path, f'autosklearn_model_{self.scorer}'), 'wb') as f:
            pickle.dump(self.model, f)
        """

    def clean_dir(self):
        for key, path in self.metric_path.items():
            print(key)
            print(path)
            folder_list = os.listdir(path)
            folder_list = [n for n in folder_list if (n.split('_')[0].isdigit())]
            print(folder_list)
            if not folder_list:
                continue
            with open(os.path.join(path, 'params.json'), 'r') as file:
                params = json.load(file)
            pred_model_list = params['load_on_predict']
            del_list = [item for item in folder_list if item not in pred_model_list]
            print(del_list)
            for folder in del_list:
                shutil.rmtree(os.path.join(path, folder))

                    # TBD add model file
        # --- FRAMEWORK SPECIFIC END ----- #2 -------------------------------------

