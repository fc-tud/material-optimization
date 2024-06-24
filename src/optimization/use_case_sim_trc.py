import sys
sys.path.append('..')
import numpy as np
import os
from scipy.spatial.distance import cdist, pdist
from datetime import datetime

import config
from scripts.opti_sim_trc import OPTIMIZER
from src.models.sim_trc.SimTRC import SimTRC


opt = __import__(f"src.optimization.{config.optimizer_dict[OPTIMIZER]['script']}", fromlist=['object'])
Optimizer = getattr(opt, OPTIMIZER)


class UseCaseTRC(Optimizer):
    def __init__(self, model, cat_features, task,  opt_name, pca=True):
        super().__init__(model, cat_features, task, opt_name, pca)
        self.sim_model = SimTRC(path=None, name='SimTRC')
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
        x_input = self.sim_model.adjust_y_dimension(x_input, 0)
        area = self.sim_model.area_from_dim(x_input)
        if area > self.task:
            return self.task - area
        result = self.model.predict(x_input)[0]
        return result

    def score(self, x_input):
        perform_score = 0
        perform_score = self.performance(x_input)
        # pca_dist = self.calc_pca_dist(x_input)
        # self.opt_results.loc[len(self.opt_results)] = [x_input, perform_score, perform_score, pca_dist]
        return -perform_score
