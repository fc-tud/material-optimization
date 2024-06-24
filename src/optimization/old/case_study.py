import numpy as np
from optimization.optimizer import SciOptimizer
from scipy.spatial.distance import cdist, pdist
from . import constants


class Optimizer(SciOptimizer):
    def __init__(self, model, cat_features, task,  opt_name='skopt', pca=True):
        super().__init__(model, cat_features, task, opt_name, pca)
        self.opt_name = opt_name
        self.scaler = None
        self.scaled_X = None
        self.PCA = pca
        self.PCA_model = None
        self.PCA_result = None

    def def_dimensions(self):
        super().def_dimensions()
        self.dimensions[1] = (self.dimensions[1][0], self.task)
        print(self.dimensions)

    def performance(self, x_input):
        self.list_to_df(x_input)
        y = self.model.predict_df(self.df, 'mean')
        return y.item()

    def calc_pca_dist(self, x_input):
        x_input = np.array(x_input)
        x_scaled = self.scaler.transform(x_input.reshape(1, -1))
        pca_input = self.PCA.transform(x_scaled)
        dist = cdist(self.PCA_result, pca_input, 'euclidean')
        dist_sum = sum(np.sort(dist, axis=0)[:10])
        return dist_sum.item() / 10

    def score(self, x_input):
        perform_score = self.performance(x_input)
        perform_score_norm = self.scaler_y.transform(np.array(perform_score).reshape(-1, 1))
        pca_dist = self.calc_pca_dist(x_input)
        score = perform_score_norm-(pca_dist/self.pca_dist_data)*constants.EP_FACTOR
        self.opt_results.loc[len(self.opt_results)] = [x_input, score.item(), perform_score, pca_dist]
        # return -perform_score (for optimization without PCA)
        return -score.item()
