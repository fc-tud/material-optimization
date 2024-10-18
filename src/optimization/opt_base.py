from datetime import datetime
import os
import pandas as pd
import numpy as np
import pickle
from contextlib import redirect_stdout
from scipy.spatial.distance import cdist, pdist
import config
from sklearn.preprocessing import StandardScaler, MinMaxScaler
from sklearn.decomposition import PCA


class BaseOptimizer:
    def __init__(self, model, cat_features, task, opt_name, pca):
        self.model = model
        self.opt_name = opt_name
        self.opt_class = None
        self.opt_time = None
        self.optimizer = None
        self.opt_results = pd.DataFrame(columns=['x', 'score'])
        self.callback_df = pd.DataFrame()
        self.output_dir = None
        self.dimensions = []
        self.scaler = None
        self.scaler_y = None
        self.scaled_X = None
        self.PCA = pca
        self.PCA_model = None
        self.PCA_result = None
        self.pca_dist_data = None
        # self.df = pd.DataFrame(columns=self.model.X_cols)
        self.list_cat_features = cat_features
        self.task = task

    def create_output_dir(self):
        np.random.seed(config.SEED)
        time_run = datetime.now().strftime('%Y_%m_%d__%H_%M_%S')
        self.output_dir = os.path.join('..', 'workdir', 'optimization', self.model.name.split('__')[0],
                                       self.model.model_name, f'{self.opt_class}_{self.opt_name}',
                                       f'{self.task}_{time_run}')
        if not os.path.exists(self.output_dir):
            os.makedirs(self.output_dir)

    def def_dimensions(self):
        dim_cat = [self.model.X[n].value_counts().keys().to_list() for n in self.list_cat_features]
        model_data = self.model.X
        for features in dim_cat:
            for i in features:
                model_data = model_data.drop(columns=[i])
        n = len(model_data.columns)
        minimum = [model_data.iloc[:, i].min() for i in range(0, n)]
        maximum = [model_data.iloc[:, i].max() for i in range(0, n)]
        dim_num = list(zip(minimum, maximum))
        self.dimensions = dim_num + dim_cat

    def list_to_df(self, x_input):
        self.df.loc[0] = 0
        x_input_nr = [num for num in x_input if not isinstance(num, str)]
        for n in range(len(x_input_nr)):
            self.df.iat[0, n] = x_input_nr[n]
        x_input_str = [string for string in x_input if isinstance(string, str)]
        for string in x_input_str:
            self.df.at[0, string] = 1

    def scale_data(self):
        self.scaler = StandardScaler()
        self.scaled_X = self.scaler.fit_transform(np.array(self.model.X.fillna(0)))
        self.scaler_y = MinMaxScaler()
        self.scaler_y = self.scaler_y.fit(np.array(self.model.y.fillna(0)))

    def calc_pca(self):
        self.scale_data()
        self.PCA = PCA()
        self.PCA_result = self.PCA.fit_transform(self.scaled_X)
        self.pca_dist_data = cdist(self.PCA_result, self.PCA_result, 'euclidean')
        self.pca_dist_data = np.partition(self.pca_dist_data, 10)
        self.pca_dist_data = np.sum(self.pca_dist_data[:, :10], axis=1)
        self.pca_dist_data = np.quantile(self.pca_dist_data, q=0.9) / 10

    def calc_pca_dist(self, x_input):
        x_input = np.array(x_input)
        x_scaled = self.scaler.transform(x_input.reshape(1, -1))
        pca_input = self.PCA.transform(x_scaled)
        dist = cdist(self.PCA_result, pca_input, 'euclidean')
        dist_sum = sum(np.sort(dist, axis=0)[:10])
        return dist_sum.item() / 10

    def retrain(self):
        for metric_key, metric in self.model.metric.items():
            self.model.define_scorer(metric_key, metric)
            self.model.create_sub_dirs(0, metric_key, metric)
            self.model.train(metric_key)
        save_model_path = os.path.join('optimization', 'models', self.model.name)
        if not os.path.exists(save_model_path):
            os.makedirs(save_model_path)
        pickle.dump(self.model, open(os.path.join(save_model_path, f'{self.model.model_name}.pkl'), 'wb'))

    def save_results(self):
        metadata_dict = {'time': self.opt_time,
                         'x': self.optimizer.x,
                         'task': self.task,
                         'performance_score': -self.performance(self.optimizer.x),
                         'model_output': self.model.predict(self.optimizer.x),
                         'optimizer_class': self.opt_class,
                         'optimizer_name': self.opt_name,
                         'optimizer_options': self.opt_options,
                         'model_class': self.model.model_class,
                         'model_name': self.model.model_name
                         }
        if self.model.model_class == 'ML-MODEL':
            metadata_dict['sim_output'] = self.sim_model.predict(self.optimizer.x)
        pickle.dump(metadata_dict, open(os.path.join(self.output_dir, 'metadata_results.pkl'), 'wb'))

        # self.callback_df.to_csv(os.path.join(self.output_dir, 'optimization_results.csv'), index=False)
        self.callback_df.to_pickle(os.path.join(self.output_dir, 'optimization_results.pkl'))
        with open(os.path.join(self.output_dir, 'results.txt'), 'w') as f:
            with redirect_stdout(f):
                print("--- %.1f seconds ---" % self.opt_time)
                print(self.optimizer.x, -self.optimizer.fun)
                print('Performance score:', -self.performance(self.optimizer.x))
                print(f"Model output: {self.model.predict(self.optimizer.x)}")
                if self.model.model_class == 'ML-MODEL':
                    print(f"Sim output: {self.sim_model.predict(self.optimizer.x)}")


