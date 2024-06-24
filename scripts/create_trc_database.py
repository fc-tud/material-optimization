import sys
sys.path.append('..')

import numpy as np
import os
from src.models.sim_trc.helpers import random_sampling, lhc_sampling, constant_sampling
from src.models.sim_trc.SimTRC import SimTRC
import time

DIR = os.path.join('X:\\', 'workdir', 'optimization', 'SIM-TRC')
I = 0

loaded_points = np.loadtxt(os.path.join(DIR ,f'sim_val_{I}.txt'), dtype=str, delimiter=';')
# Convert numpy array to nested Python list
RUNS = [[int(float(value)) for value in row] for row in loaded_points]
# Define runs manualy
#RUNS = [[9,9,9,4,4,4], [9,9,9,5,5,5], [10,10,10,3,3,3], [11,11,11,3,3,3], [10,10,10,4,4,4], [11,11,11,4,4,4], [10,10,10,5,5,5], [11,11,11,5,5,5]]

NUM_JOBS = 15  # 30 auf MCKnecht 5 zu viel, 20 auf Win-Vm zu viel
SAMPLING = constant_sampling

if __name__ == "__main__":
    print(RUNS)
    SimTRC = SimTRC(path=None, name='SimTRC')
    run_folder = SimTRC.create_output_dir(mode='database', task = SAMPLING)
    SimTRC.create_search_space(RUNS)
    SimTRC.run_parallel_jobs(NUM_JOBS)
    SimTRC.get_results()
    SimTRC.build_df()
