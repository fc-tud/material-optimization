import sys
import os
import pickle
from optimization import constants
from models.helpers.data_handling import check_dataset
import numpy as np
from optimizer_sim_trc import Optimizer

# Import the best performing Framework
MODEL = 'sim'
mod = __import__(f'frameworks.'models.{MODEL}.object', fromlist=['object']')
Model = getattr(mod, 'Model')

USE_CASE = 'sim-trc'
OPTIMIZER = 'optimizer_sim_trc'
#opt = __import__(f'optimization.{OPTIMIZER}', fromlist=['object'])
#Optimizer = getattr(opt, 'Optimizer')

CASE_LIST = np.arange(5, 81, 5) # Range of Area


def main(path, name, task):
    print(name)
    model = Model(path, name)
    model.read_data()
    optimizer = Optimizer(model=model, cat_features=[], task=task, pca=False)
    storage_path = os.path.join('optimization', 'models', optimizer.model.name, f'{MODEL}.pkl')
    if MODEL != 'dummy':
        if os.path.exists(storage_path):
            print('\nLoad retrained Model')
            optimizer.model = pickle.load(open(storage_path, 'rb'))
        else:
            model.create_output_dir()
            optimizer.retrain()
    optimizer.create_output_dir()
    optimizer.optimize()
    optimizer.save_results()


if __name__ == '__main__':
    for task in CASE_LIST:
        print(39 * '-')
        print(f'Task: {USE_CASE}, Sub-Task: {task}')
        #if task not in done_tasks:
        path = os.path.join(constants.DATA_FOLDER, dataset)
        name = f'{USE_CASE}_{MODEL}'
        files = check_dataset(path)
        main(path, name, task)
        print(79 * '^')
