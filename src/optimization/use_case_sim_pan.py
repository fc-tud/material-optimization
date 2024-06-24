import sys
sys.path.append('..')
import os
import numpy as np
from scipy.spatial.distance import cdist, pdist
from datetime import datetime

import config
from src.models.sim_trc.helpers import define_input
from scripts.opti_sim_pan import OPTIMIZER
from src.models.sim_pan.SimPAN import SimPAN

opt = __import__(f"src.optimization.{config.optimizer_dict[OPTIMIZER]['script']}", fromlist=['object'])
Optimizer = getattr(opt, OPTIMIZER)


class UseCasePAN(Optimizer):
    def __init__(self, model, cat_features, task,  opt_name, pca=True):
        super().__init__(model, cat_features, task, opt_name, pca)
        self.sim_model = SimPAN(path=None, name='SimPAN')
        self.opt_name = opt_name
        self.output_dir = None
        self.sim_dir = None
        self.dimensions = None
        self.scaler = None
        self.scaled_X = None
        self.PCA = pca
        self.PCA_model = None
        self.PCA_result = None

    def performance(self, x_input):
        perform_score = 0
        result = self.model.predict(x_input)

        # Get perform_score from the current task
        perform_score = [val1 for val1, val2 in zip(result, self.task) if val2 is None][0]
        # Check if all elements in result are larger than the corresponding elements in task
        if any(val1 < val2 if val2 is not None else False for val1, val2 in zip(result, self.task)):
            perform_score = 0.0
            # Summe der Abstände zu target (Damit gibt es eine Richtung,
            # hin zu dem Bereich wo die target Werte erfüllt sind).
            # Wenn Bedingung erfüllt, ist dies aber immer kleiner (besser)
           
            for val1, val2 in zip(result, self.task):
                if val2 is not None and val2 > val1:
                    diff = -abs(val2 - val1)
                    perform_score += diff

        # pca_dist = self.calc_pca_dist(x_input)
        # self.opt_results.loc[len(self.opt_results)] = [x_input, perform_score, perform_score, pca_dist]
        return perform_score

    def score(self, x_input):
        perform_score = 0
        perform_score = self.performance(x_input)
        # pca_dist = self.calc_pca_dist(x_input)
        # self.opt_results.loc[len(self.opt_results)] = [x_input, perform_score, perform_score, pca_dist]
        return -perform_score
