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
USE_PARALLEL_VERSION = False  # Change this based on your condition

MODEL = 'ML-MODEL'  # ML-MODEL, SIM-PAN, SIM-TRC
ML_MODEL = 'AutoSklearn'
USE_CASE = 'SIM-PAN'

# Define the search spaces for the targets
wca_space = np.linspace(160, 160, 1)
q_space = np.linspace(80, 200, 40)
sigma_space = np.linspace(6, 6, 1)

# Create combinations with one variable fixed at None
all_combinations = list(product(wca_space.tolist(), q_space.tolist(), [None] + sigma_space.tolist()))
# Filter combinations where exactly two variables are not None
filtered_combinations = [combo for combo in all_combinations if sum(x is not None for x in combo) == 2]


# Define a custom key function for sorting
def custom_key(combination):
    return tuple(x is not None for x in combination)


# Sort the filtered combinations
combinations = sorted(filtered_combinations, key=custom_key)


def import_sim_model():
    # Import the chosen Model
    mod = __import__(f"src.models.{model_dict[MODEL]['dir']}.{model_dict[MODEL]['script']}", fromlist=['object'])
    model = getattr(mod, model_dict[MODEL]['class'])
    return model


def import_optimizer():
    # Import the chosen Use-Case + Optimizer Combi
    optimizer_ = 'use_case_sim_pan'
    opt = __import__(f"src.optimization.{optimizer_}", fromlist=['object'])
    optimizer = getattr(opt, 'UseCasePAN')
    return optimizer


def main(path, name, task):
    optimizer = import_optimizer()
    if MODEL == 'SIM-PAN':
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
    optimizer.optimize()
    optimizer.save_results()


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


if __name__ == '__main__':
    # Your condition to choose between parallel and single version
    if USE_PARALLEL_VERSION:
        run_parallel(combinations)
    else:
        for task in combinations:
            process_task(task)
    """
    n_workers = 1
    with concurrent.futures.ProcessPoolExecutor(max_workers=n_workers) as executor:
        # Use executor.map to distribute the tasks among workers
        executor.map(process_task, combinations)
    print(79 * '^')


    for task in combinations:
        print(39 * '-')
        print(f'Task: {USE_CASE}, Sub-Task: {task}')
        # if task not in done_tasks:
        name = f'{USE_CASE}__{MODEL}__{task}'
        # files = check_dataset(path=None, name=name)
        main(path=None, name=name, task=task)
        print(79 * '^')
    """
