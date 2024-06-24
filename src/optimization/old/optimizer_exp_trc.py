import numpy as np
from frameworks.optimization.sci_kit_optimizer import SciOptimizer
from scipy.spatial.distance import cdist, pdist
from skopt import space
from frameworks.models.sim_trc.helpers import define_input
import os
from datetime import datetime
import subprocess
from . import constants


para_list = ['x1[m]', 'x2[m]', 'x3[m]', 'y1[m]', 'y2[m]', 'y3[m]']


class Optimizer(SciOptimizer):
    def __init__(self, model, cat_features, task,  opt_name='skopt', pca=True):
        super().__init__(model, cat_features, task, opt_name, pca)
        self.opt_name = opt_name
        self.output_dir = None
        self.sim_dir = None
        self.scaler = None
        self.scaled_X = None
        self.PCA = pca
        self.PCA_model = None
        self.PCA_result = None

    def create_run_dir(self):
        time_run = datetime.now().strftime('%Y_%m_%d__%H_%M_%S')
        self.output_dir = os.path.join('..', 'workdir_opt', 'TRC_sim', '{name}_{time}'.format(name=self.opt_name, time=time_run))
        if not os.path.exists(self.output_dir):
            os.makedirs(self.output_dir)

    def def_dimensions(self):
        # anpassen da Dimensions nur int sein können:
        self.dimensions = [space.Integer(1, 9), space.Integer(1, 9), space.Integer(1, 9),
                           space.Integer(0, 3), space.Integer(1, 3), space.Integer(0, 3)]
                           
    def area_from_dim(self, x_input):
        area = 0
        #print('Input:',x_input)
        for i in range(3):
            sub_area = x_input[i]*x_input[i+3]
            area += sub_area
        #print('Area:', area)
        return area

    def performance(self, x_input):
        area = self.area_from_dim(x_input)
        if area > self.task:
            return 0
        result = self.model.predict(x_input, self.output_dir)
        return result
            
            
    def calc_pca_dist(self, x_input):
        x_input = np.array(x_input)
        x_scaled = self.scaler.transform(x_input.reshape(1, -1))
        pca_input = self.PCA.transform(x_scaled)
        dist = cdist(self.PCA_result, pca_input, 'euclidean')
        dist_sum = sum(np.sort(dist, axis=0)[:10])
        return dist_sum.item() / 10


    def score(self, x_input):
        perform_score = 0
        perform_score = self.performance(x_input)
        # pca_dist = self.calc_pca_dist(x_input)
        # self.opt_results.loc[len(self.opt_results)] = [x_input, perform_score, perform_score, pca_dist]
        return -perform_score
