#!/usr/bin/env python
# coding: utf-8

import os
import config
from datetime import datetime
from src.models.auto_ml.base_auto_ml import BaseModelAutoML
from src.models.helpers.quantile_reg import pinball_loss, partial
import xgboost as xgb
import optuna
import sklearn.metrics
from sklearn.model_selection import ShuffleSplit


class XGBoost(BaseModelAutoML):
    def __init__(self, path, name):
        super().__init__(path, name)
        self.model_name = 'XGBoost'
        self.model_version = config.model_dict[self.model_name]['version']
        self.X_split_train = None
        self.y_split_train = None
        self.metric_key = None
        self.best_trial = None
        self.best_params = None

    @staticmethod
    def define_hpspace(trial):
        params = {'max_depth': trial.suggest_int('max_depth', 2, 10),
                  'learning_rate': trial.suggest_float('learning_rate', 1e-3, 1, log=True),
                  'n_estimators': trial.suggest_int('n_estimators', 250, 1500),
                  'reg_alpha': trial.suggest_float('reg_alpha', 0, 100),
                  'reg_lambda': trial.suggest_float('reg_lambda', 0, 100),
                  }
        return params

    def define_scorer(self, metric_key, metric):
        if metric[0] == 'quantile':
            quantile = metric[1]
            self.scorer[metric_key] = partial(pinball_loss, quantile=quantile)
        elif metric == 'mse':
            self.scorer[metric_key] = 'reg:squarederror'
        elif metric == 'r2':
            self.scorer[metric_key] = 'r2'
        else:
            self.scorer[metric_key] = None

    def train(self, modus, task_key, metric_key, X_data, y_data):
        self.create_task_dirs(modus, task_key)
        self.create_metric_dirs(modus, task_key, metric_key)
        if self.model_version == 'hpo':
            self.X_split_train = X_data
            self.y_split_train = y_data
            self.metric_key = metric_key
            self.tune(modus, task_key)
        self.retrain(modus, task_key, metric_key, X_data, y_data)

    def tune(self, modus, task_key):
        study_name = f"{self.name}_{self.model_name}_{'_'.join(self.split_path.split(os.sep)[-2:])}_{task_key}"
        storage = f"sqlite:///{os.path.join('..', 'workdir', 'db.sqlite3')}"
        study = optuna.create_study(
            direction="minimize",
            sampler=optuna.samplers.TPESampler(),
            pruner=optuna.pruners.MedianPruner(),
            study_name=study_name,
            storage=storage
        )

        study.optimize(self.objective, n_trials=config.OPTUNA_TRAILS, n_jobs=config.NUM_CORES)
        print("Number of finished trials: {}".format(len(study.trials)))

        self.best_params = study.best_params

        best_score = study.best_value
        print(f"Best score: {best_score}\n")
        print(f"Optimized parameters: {self.best_params}\n")
        print("Number of finished trials: {}".format(len(study.trials)))

        print("Best trial:")
        self.best_trial = study.best_trial

        print("  Value: {}".format(self.best_trial.value))

        print("  Params: ")
        for key, value in self.best_trial.params.items():
            print("    {}: {}".format(key, value))

        # TBD: save search to csv
        with open(os.path.join(self.task_path[(modus, task_key)], f'best_params.txt'), 'w') as f:
            f.write(str(self.best_params))
            f.write(str(best_score))

    def objective(self, trial):
        # Hyperparameters
        params = self.define_hpspace(trial)
        # Inner Loop CV
        rs = ShuffleSplit(n_splits=config.INNER_SPLITS, train_size=0.8, random_state=config.SEED)
        cost_run = []

        for i, (split_train_index, split_val_index) in enumerate(rs.split(self.X_split_train)):
            X_train, y_train = self.X_split_train.iloc[split_train_index], self.y_split_train.iloc[split_train_index]
            X_val, y_val = self.X_split_train.iloc[split_val_index], self.y_split_train.iloc[split_val_index]

            # Training per Split
            model = self.train_(params, X_train, y_train)
            # Evaluation
            predict = model.predict(X_val)
            rmse = sklearn.metrics.mean_squared_error(y_val, predict)
            cost_run.append(rmse)
            trial.report(sum(cost_run) / len(cost_run), i)
            if trial.should_prune():
                raise optuna.exceptions.TrialPruned()
        return sum(cost_run) / len(cost_run)

    def train_(self, params, X_train, y_train):
        # --- FRAMEWORK SPECIFIC START --- #1 -------------------------------------q

        model = xgb.XGBRegressor(**params,
                                 objective=self.scorer[self.metric_key],
                                 n_jobs=1,
                                 # n_jobs=config.NUM_CORES,
                                 # tree_method='gpu_hist',
                                 random_state=config.SEED
                                 )

        # --- FRAMEWORK SPECIFIC END ----- #1 -------------------------------------
        model.fit(X_train, y_train)
        return model

    def retrain(self, modus, task_key, metric_key, X_data, y_data):
        if self.model_version == 'hpo':
            # --- FRAMEWORK SPECIFIC START --- #1 -------------------------------------
            self.model[(modus, task_key, metric_key)] = xgb.XGBRegressor(**self.best_params,
                                                                         objective=self.scorer[metric_key],
                                                                         n_jobs=1,
                                                                         # n_jobs=config.NUM_CORES,
                                                                         # tree_method='gpu_hist',
                                                                         random_state=config.SEED
                                                                         )

            # --- FRAMEWORK SPECIFIC END ----- #1 -------------------------------------
        else:
            # --- FRAMEWORK SPECIFIC START --- #1 -------------------------------------
            self.model[(modus, task_key, metric_key)] = xgb.XGBRegressor(objective=self.scorer[metric_key],
                                                                         n_jobs=1,
                                                                         # n_jobs=config.NUM_CORES,
                                                                         # tree_method='gpu_hist',
                                                                         random_state=config.SEED,
                                                                         )

            # print(self.model[(modus, task_key, metric_key)].evals_result())
            # --- FRAMEWORK SPECIFIC END ----- #1 -------------------------------------

        print(datetime.now())
        print(f'Training in {modus} of {task_key} over {config.MAX_TIME_MINUTES} min started')
        self.model[(modus, task_key, metric_key)].fit(X_data, y_data)
        print(datetime.now())
