import sys
sys.path.append('..')

from src.models.sim_pan.helpers import random_sampling, lhc_sampling
from src.models.sim_pan.SimPAN import SimPAN
import time

MAX_RUNS = 500
SAMPLING = lhc_sampling


if __name__ == "__main__":
    SimPAN = SimPAN(path=None, name='SimPAN')
    run_folder = SimPAN.create_output_dir(SAMPLING, MAX_RUNS)
    SimPAN.create_search_space(MAX_RUNS)
    SimPAN.build_df()
