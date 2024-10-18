import sys
sys.path.append('..')
import os
import pickle
import importlib
import numpy as np
from itertools import product
import concurrent.futures

from config import model_dict

OPTIMIZER = 'NGOptimizer'
OPTIMIZER_NAME = 'CMA'
USE_PARALLEL_VERSION = True  # Change this based on your condition

MODEL = 'ML-MODEL'  # ML-MODEL, SIM-PAN, SIM-TRC
ML_MODEL = 'AutoSklearn'
USE_CASE = 'SIM-TRC'

CASE_LIST = np.arange(5, 81, 1)  # Range of Area


def import_sim_model():
    # Import the chosen Model
    mod = __import__(f"src.models.{model_dict[MODEL]['dir']}.{model_dict[MODEL]['script']}", fromlist=['object'])
    model = getattr(mod, model_dict[MODEL]['class'])
    return model


def import_optimizer():
    # Import the chosen Use-Case + Optimizer Combi
    optimizer_ = 'use_case_sim_trc'
    opt = __import__(f"src.optimization.{optimizer_}", fromlist=['object'])
    optimizer = getattr(opt, 'UseCaseTRC')
    return optimizer


def process_task(task):
    print(39 * '-')
    print(f'Task: {USE_CASE}, Sub-Task: {task}')
    # if task not in done_tasks:
    name = f'{USE_CASE}__{MODEL}__{task}'
    # files = check_dataset(path=None, name=name)
    main(path=None, name=name, task=task)
    print(79 * '^')


def run_parallel(combinations):
    n_workers = 20
    with concurrent.futures.ProcessPoolExecutor(max_workers=n_workers) as executor:
        executor.map(process_task, combinations)
    print(79 * '^')


def main(path, name, task):
    optimizer = import_optimizer()
    if MODEL == 'SIM-TRC':
        model = import_sim_model()
        print(name)
        model = model(path, name)
        # model.read_data()
    elif MODEL == 'ML-MODEL':
        storage_path = os.path.join('..', 'ml_models_bin', USE_CASE, ML_MODEL, 'model.pkl')
        model = pickle.load(open(storage_path, 'rb'))
    else:
        print('No valid Model definition')
        return
    optimizer = optimizer(model=model, cat_features=[], opt_name=OPTIMIZER_NAME, task=task, pca=False)
    optimizer.create_output_dir()
    if MODEL == 'SIM-TRC':
        optimizer.model.create_output_dir(mode = 'optimization', task = optimizer.output_dir)
    optimizer.optimize()
    optimizer.save_results()


if __name__ == '__main__':
    if USE_PARALLEL_VERSION:
        run_parallel(CASE_LIST)
    else:
        for task in CASE_LIST:
            process_task(task)
