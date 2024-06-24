import os

# Data
DATA_FOLDER = os.path.join('..', 'data', 'run')
NAME_DATA = 'data.csv'


# CV
INNER_SPLITS = 10
DEFAULT_TRAIN_SIZE = 0.75
OUTER_SPLITS = 5
SPLIT_MODE = 'random'  # 'random', 'PCA', 'extra', 'pareto'
SPLIT_OPTIONS = {}, # {}, {}, {'extra_split': {'label': -1}}, {}
RETRAIN = True

# Computing
# Metrics
METRICS = dict(mean='mse')
DEF_QUANTILE_SCORE = 0.5

# Modles
NUM_CORES = 8
# Pytorch
OPTUNA_TRAILS = 200
# AutoML
MAX_TIME_MINUTES = 15
MTL_LIST = []

SEED = 1

# Optimization
optimizer_dict = {'SciOptimizer': {'scipy_parallel': {'num_jobs': 20, 'iterations': 20},
                                   'scipy': {},
                                   'skopt': {},
                                   'brute-force': {},
                                   'script': 'sci_kit_optimizer'},

                  'NGOptimizer':  {'NGOpt': {'num_jobs': 1, 'iterations': 300000},
                                   'PSO': {'num_jobs': 1, 'iterations': 300000},
                                   'OnePlusOne': {'num_jobs': 1, 'iterations': 300000},
                                   'CMA': {'num_jobs': 1, 'iterations': 300000},
                                   'DE': {'num_jobs': 1, 'iterations': 300000},
                                   'TBPSA': {'num_jobs': 1, 'iterations': 300000},
                                   'TPDE': {'num_jobs': 1, 'iterations': 300000},
                                   'NoisyDE': {'num_jobs': 1, 'iterations': 300000},
                                   'Bayes': {'num_jobs': 1, 'iterations': 1250},
                                   'ScrHammersleySearch': {'num_jobs': 1, 'iterations': 300000},
                                   'RandomSearch': {'num_jobs': 1, 'iterations': 300000},
                                   'script': 'ng_optimizer'}}


# Model library
model_dict = {'FFNN_mtl': {'dir': 'pytorch', 'script': 'ffnn_mtl', 'class': 'FFNN'},
              'FFNN_stl': {'dir': 'pytorch', 'script': 'ffnn_mtl', 'class': 'FFNN'},
              'AutoSklearn': {'dir': 'auto_ml', 'script': 'autosklearn', 'class': 'AutoSklearn'},
              'XGBoost': {'dir': 'auto_ml', 'script': 'xgboost', 'class': 'XGBoost', 'version': 'hpo'},
              'CatBoost': {'dir': 'auto_ml', 'script': 'catboost', 'class': 'CatBoost', 'version': 'hpo'},
              'SIM-TRC': {'dir': 'sim_trc', 'script': 'SimTRC', 'class': 'SimTRC'},
              'SIM-PAN': {'dir': 'sim_pan', 'script': 'SimPAN', 'class': 'SimPAN'}}
