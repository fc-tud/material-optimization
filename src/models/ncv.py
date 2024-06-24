#!/usr/bin/env python
# coding: utf-8

import os
import pandas as pd
import numpy as np
from sklearn.model_selection import ShuffleSplit, GroupShuffleSplit
from sklearn.preprocessing import LabelEncoder

import config
from src.models.helpers.data_handling import regression_results, splitting_functions


def ncv(path, name, model_key):
    # Import the chosen Model
    mod = __import__(f"src.models."
                     f"{config.model_dict[model_key]['dir']}."
                     f"{config.model_dict[model_key]['script']}", fromlist=['object'])
    model = getattr(mod, config.model_dict[model_key]['class'])
    print(name)
    model = model(path, name)
    model.read_data()
    print('Split-mode:', config.SPLIT_MODE)
    model.create_output_dir()
    print(model.output_dir)
    # Model to GPU if possible
    if config.model_dict[model_key]['dir'] == 'pytorch':
        model.to(model.device)
        print('Device: ', model.device)
    model.save_exp_def()

    # Def "outer loop CV"
    i = 1
    split_func = splitting_functions[config.SPLIT_MODE]
    splits = split_func(model, config.OUTER_SPLITS)
    scores = pd.DataFrame(columns=['quantile_score',
                                   'r2',
                                   'rmse',
                                   'mae',
                                   'mse',
                                   'mape',
                                   'task_key',
                                   'split',
                                   'mode',
                                   'metric',
                                   'evaluation'])
    # ncv_data_index, model.X_data[ncv_data_index], if NCV on specific subset
    # Start "outer loop CV"
    for train_index, test_index in splits:
        print('{:-^60}'.format(f'SPLIT {i}'))

        if model.group_split_col:
            le = LabelEncoder()
            model.groups_inner = le.fit_transform(model.data[model.group_split_col].loc[train_index])

        # Werden die verschiedenen Metriken bei der NCV überhaupt benötigt oder nur beim Finalen Training?
        for metric_key, metric in config.METRICS.items():
            print('{:-^40}'.format(f'Metric: {metric}'))
            if metric[0] == 'quantile':
                quantile = metric[1]
                print(quantile)
            else:
                quantile = config.DEF_QUANTILE_SCORE
            if config.model_dict[model_key]['dir'] == 'auto_ml':
                model.define_scorer(metric_key, metric)
            model.create_sub_dirs(i)

            # Set and save indexes for data splitting
            model.train_index, model.test_index = train_index, test_index
            np.savetxt(os.path.join(model.split_path, 'train_index.txt'), [model.train_index], fmt="%d",
                       delimiter=",")
            np.savetxt(os.path.join(model.split_path, 'test_index.txt'), [model.test_index], fmt="%d",
                       delimiter=",")

            y_pred = model.evaluate(metric_key=metric_key)
            y_true = model.y_data.loc[test_index]
            for n in range(model.num_tasks):
                if config.model_dict[model_key]['dir'] == 'pytorch':
                    print(f'\nTask: {model.y_label[n]}')
                    y_pred = y_pred.set_axis(y_true.index, axis='index')
                    task_result = regression_results(y_true.iloc[:, n].dropna(),
                                                     y_pred.iloc[:, n][y_true.iloc[:, n].notna()],
                                                     quantile)
                    task_result.extend([model.y_label[n], i, model.model_name, metric, config.SPLIT_MODE])
                    scores.loc[len(scores)] = task_result
                if config.model_dict[model_key]['dir'] == 'auto_ml':
                    for mode in ['STL'] + config.MTL_LIST:
                        print(f'Task: {model.y_label[n]}, Mode: {mode}')
                        task_result = regression_results(
                            y_true.iloc[:, n].dropna(),
                            y_pred.loc[:, f'{model.y_label[n]}_{mode}'][y_true.iloc[:, n].notna()],
                            quantile)
                        task_result.extend([model.y_label[n], i, mode, metric, config.SPLIT_MODE])
                        scores.loc[len(scores)] = task_result

        scores.to_csv(os.path.join(model.output_dir, 'regression_summary.csv'), index=None)
        i += 1
    # Retrain
    if config.RETRAIN:
        if config.model_dict[model_key]['dir'] == 'auto_ml':
            for y_label in model.y_label:
                model.create_sub_dirs('retrain')
                model.train('STL', y_label, metric_key, model.X_data, model.y_data[[y_label]])
        if config.model_dict[model_key]['dir'] == 'pytorch':
            model.create_sub_dirs('retrain')
            # train test index
            _ = model.evaluate(retrain=True)

    model.save_model()
    # gc.collect()
    # print('{:-^20}'.format(f'finished training for {metric}'))
