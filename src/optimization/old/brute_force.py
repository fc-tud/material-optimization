import numpy as np
from optimization.optimizer import SciOptimizer
from scipy.spatial.distance import cdist, pdist
from . import constants
import pybamm
from optimization.timeout import timeout

para_list = ['Electrode height [m]', 'Electrode width [m]',
             'Initial inner SEI thickness [m]', 'Initial outer SEI thickness [m]',
             'Negative current collector thickness [m]', 'Positive current collector thickness [m]',
             'Negative electrode thickness [m]', 'Positive electrode thickness [m]'
            ]


import signal
import time

class Timeout(Exception):
    pass

def handler(sig, frame):
    raise Timeout

signal.signal(signal.SIGALRM, handler)  # register interest in SIGALRM events



class Optimizer(SciOptimizer):
    def __init__(self, model, cat_features, task,  opt_name='brute-force', pca=True):
        super().__init__(model, cat_features, task, opt_name, pca)
        self.opt_name = opt_name
        self.scaler = None
        self.scaled_X = None
        self.PCA = pca
        self.PCA_model = None
        self.PCA_result = None
        self.parameter_values = pybamm.ParameterValues("Chen2020")
        self.model_dfn = pybamm.lithium_ion.DFN()

    def def_dimensions(self):
        super().def_dimensions()
        self.dimensions[1] = (self.dimensions[1][0], self.task)
        print(self.dimensions)


    def performance(self, x_input):
        for para, value in zip(para_list, x_input):
            self.parameter_values[para] = value
        sim = pybamm.Simulation(self.model_dfn, parameter_values=self.parameter_values)
        try:
            sim.solve([0, 3600])
            solution = sim.solution
            y = solution["Terminal voltage [V]"].entries[-1].item()
        except pybamm.SolverError:
            y = 0
        return y

    def calc_pca_dist(self, x_input):
        x_input = np.array(x_input)
        x_scaled = self.scaler.transform(x_input.reshape(1, -1))
        pca_input = self.PCA.transform(x_scaled)
        dist = cdist(self.PCA_result, pca_input, 'euclidean')
        dist_sum = sum(np.sort(dist, axis=0)[:10])
        return dist_sum.item() / 10

    def score(self, x_input):
        perform_score = 0
        signal.alarm(2)  # timeout in 2 seconds
        try:
            perform_score = self.performance(x_input)
        except Timeout:
            print('took too long')
        pca_dist = self.calc_pca_dist(x_input)
        self.opt_results.loc[len(self.opt_results)] = [x_input, perform_score, perform_score, pca_dist]
        return -perform_score
