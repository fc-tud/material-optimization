import os
import time
from scipy.optimize import minimize, brute
from skopt import gp_minimize
from skopt import Optimizer
from skopt import space
from joblib import Parallel, delayed
import numpy as np
import pandas as pd
from tqdm import tqdm
import config
from src.optimization.opt_base import BaseOptimizer


class SciOptimizer(BaseOptimizer):
    def __init__(self, model, cat_features, task, opt_name, pca):
        super().__init__(model, cat_features, task, opt_name, pca)
        self.opt_class = 'SciOptimizer'
        self.opt_options = config.optimizer_dict[self.opt_class][opt_name]
        self.dimensions = list()
        #self.evaluated_points = set()
        self.evaluated_points_info = dict()
        
    def def_dimensions(self):
        for variable in self.sim_model.space:
            name = variable["name"]
            domain = variable["domain"]
            var_type = variable["type"]

            if var_type == "integer":
                param = space.Integer(low=domain[0], high=domain[1])
            elif var_type == "continuous":
                param = space.Real(low=domain[0], high=domain[1])
            elif var_type == "list":
                param = space.Categorical(domain)
            self.dimensions.append(param)
            
        print('Dimensions', self.dimensions)

    def def_starting_point(self):
        start_point = self.model.X.sample(n=1)
        start_point = start_point.to_numpy()
        return start_point

    def generate_random_initial_guess(self):
        return [np.random.uniform(low=b[0], high=b[1]) for b in self.dimensions]

    # Callback function to write intermediate results to a CSV file
    def callback(self, iteration, x, f):
        input_ = pd.DataFrame(x)
        output_ = pd.DataFrame(f).add_prefix('res_')
        # Update evaluated points information
        for i, point in enumerate(x):
            count, last_iteration = self.get_point_info(point)
            input_.loc[i, 'repetitions'] = count
            input_.loc[i, 'last_iteration'] = last_iteration
        self.callback_df = pd.concat([input_, output_], axis=1)
        #self.callback_df['iter'] = np.repeat(np.arange(0, iteration + 1), self.opt_options['num_jobs'])
        self.callback_df.to_csv(os.path.join(self.output_dir, 'optimization_results.csv'), index=False)

    def optimize(self):
        self.def_dimensions()
        print(self.dimensions)
        start_time = time.monotonic()

        if self.PCA:
            self.calc_pca()
            self.opt_results = pd.DataFrame(columns=['x', 'score', 'performance', 'pca'])

        if self.opt_name == 'scipy_parallel':
            progress_bar = tqdm(total=self.opt_options['iterations'], desc="Optimizing", position=0, leave=True)
            self.optimizer = Optimizer(dimensions=self.dimensions,
                                       random_state=1,
                                       base_estimator='gp')
            for i in range(self.opt_options['iterations']):
                x = self.optimizer.ask(n_points=self.opt_options['num_jobs'])
                for idx in range(len(x)):
                    x[idx]=self.adjust_y_dimension(x[idx], value=1)
                # Remove duplicates in x list
                x = [list(x) for x in np.unique([sub for sub in x], axis=0)]
                # Remove duplicates from previous runs
                x = [point for point in x if not tuple(point) in self.evaluated_points_info]
                y = Parallel(n_jobs=self.opt_options['num_jobs'])(delayed(self.score)(v) for v in x)  # evaluate points in parallel
                # Update evaluated points information
                for point in x:
                    self.evaluated_points_info[tuple(point)] = {'count': self.get_point_info(point)[0] + 1, 'last_iteration': i}
                self.optimizer.tell(x, y)
                progress_bar.update(1)
                self.callback(i, self.optimizer.Xi, self.optimizer.yi)
            self.optimizer = self.optimizer.get_result()

        if self.opt_name == 'scipy':
            # start_point = self.def_starting_point()
            # num_samples = 5  # Adjust the number of samples as needed
            start_point = self.generate_random_initial_guess()

            self.optimizer = minimize(fun=self.score,  # the function to minimize
                                      method='L-BFGS-B',
                                      bounds=self.dimensions,  # the bounds on each dimension of x
                                      x0=start_point,
                                      options=dict(iprint=10,
                                                   maxls=100)
                                      )

        if self.opt_name == 'skopt':
            # starting_point_list = self.def_starting_point()
            # print(starting_point_list)
            self.optimizer = gp_minimize(func=self.score,  # the function to minimize
                                         dimensions=self.dimensions,  # the bounds on each dimension of x
                                         n_calls=50,  # the number of evaluations of f
                                         n_initial_points=5,  # the number of random initialization points
                                         # x0=starting_point_list,
                                         random_state=1,
                                         n_jobs=-1,
                                         verbose=True)

        if self.opt_name == 'brute-force':
            self.optimizer = brute(func=self.score,  # the function to minimize
                                   ranges=self.dimensions,  # the bounds on each dimension of x
                                   Ns=100,
                                   workers=16
                                   )

        self.opt_time = time.monotonic() - start_time
        print("--- %.1f seconds ---" % self.opt_time)
        print(self.optimizer.x, -self.optimizer.fun)
        # print('Performance score:', -self.performance(self.optimizer.x))
        
   
    # Function to check if a point has been evaluated and get repetition count and last iteration
    def get_point_info(self, point):
        info = self.evaluated_points_info.get(tuple(point), {'count': 0, 'last_iteration': -1})
        return info['count'], info['last_iteration']