import os
import time
import pickle
import nevergrad as ng
from nevergrad.optimization import optimizerlib
from concurrent import futures
import numpy as np
import pandas as pd
from tqdm import tqdm
from contextlib import redirect_stdout
from io import StringIO
import warnings
import cma
import config
from src.optimization.opt_base import BaseOptimizer
warnings.simplefilter("ignore", cma.evolution_strategy.InjectionWarning)

MODE = 'sequential'


class NGOptimizer(BaseOptimizer):
    def __init__(self, model, cat_features, task, opt_name, pca):
        super().__init__(model, cat_features, task, opt_name, pca)
        self.opt_class = 'NGOptimizer'
        self.opt_options = config.optimizer_dict[self.opt_class][opt_name]
        self.instrumentation = None
        self.results = dict()

    def define_optimizer(self):
        optimizer_mapping = {
            'NGOpt': ng.optimizers.NGOpt,
            'PSO': ng.optimizers.PSO,
            'OnePlusOne': ng.optimizers.OnePlusOne,
            'DE': ng.optimizers.DE,
            'CMA': ng.optimizers.CMA,
            'TBPSA': ng.optimizers.TBPSA,
            'TPDE': ng.optimizers.TwoPointsDE,
            'NoisyDE': ng.optimizers.NoisyDE,
            'Bayes': ng.optimizers.BO,
            'ScrHammersleySearch': ng.optimizers.ScrHammersleySearch,
            'RandomSearch': ng.optimizers.RandomSearch
            # Add more optimizers as needed
        }
        return optimizer_mapping[self.opt_name](parametrization=self.instrumentation,
                                                budget=self.opt_options['iterations'],
                                                num_workers=self.opt_options['num_jobs'])

    def def_dimensions(self):
        transformed_space = []
        for variable in self.sim_model.space:
            name = variable["name"]
            domain = variable["domain"]
            var_type = variable["type"]

            if var_type == "integer":
                param = ng.p.Scalar(init=np.random.uniform(low=domain[0], high=domain[1])).set_bounds(
                    *domain).set_integer_casting()
            elif var_type == "continuous":
                param = ng.p.Scalar(init=np.random.uniform(low=domain[0], high=domain[1])).set_bounds(*domain)

            transformed_space.append(param)
        self.instrumentation = ng.p.Instrumentation(*transformed_space)
        print('Instrumentation', self.instrumentation)

    def def_starting_point(self):
        start_point = self.model.X.sample(n=1)
        start_point = start_point.to_numpy()
        return start_point

    def generate_random_initial_guess(self):
        return [np.random.uniform(low=b[0], high=b[1]) for b in self.dimensions]

    # Callback function to write intermediate results to a CSV file
    def callback(self, x, f):
        input_ = pd.DataFrame([x])
        output_ = pd.DataFrame([f]).add_prefix('res_')
        df = pd.concat([input_, output_], axis=1)
        df['time'] = time.monotonic()
        self.callback_df = pd.concat([self.callback_df, df], axis=0)
        # df['iter'] = np.repeat(np.arange(0, iteration + 1), self.opt_options['num_jobs'])

    def optimize(self):
        self.def_dimensions()
        # print(self.instrumentation)
        start_time = time.monotonic()

        if self.PCA:
            self.calc_pca()
            self.opt_results = pd.DataFrame(columns=['x', 'score', 'performance', 'pca'])

        self.optimizer = self.define_optimizer()
        progress_bar = tqdm(total=self.optimizer.budget, desc="Optimizing", position=0, leave=True)

        # Sequential Verison:
        if MODE == 'sequential':
            for _ in range(self.optimizer.budget):
                x = self.optimizer.ask()
                loss = self.objfunc(x)
                self.optimizer.tell(x, loss)
                self.callback([*x.args], loss)
                progress_bar.update(1)
            # self.optimizer.minimize(self.objfunc)

        if MODE == 'parallel_ng':
            def func(*x):
                loss = self.score(list(x))
                self.callback(list(x), loss)
                progress_bar.update(1)
                return loss

            with futures.ThreadPoolExecutor(self.opt_options['num_jobs']) as executor:
                # with futures.ProcessPoolExecutor(max_workers=optimizer.num_workers) as executor: - Testen!
                recommendation = self.optimizer.minimize(func, executor=executor, batch_mode=False)

        if MODE == 'parallel_own':
            # Redirect stdout to suppress optimizer output - Alternative
            """
            with open('/dev/null', 'w') as fnull:
                with redirect_stdout(fnull):
                # Your code here
            """
            # Create a buffer to redirect stdout
            # with StringIO() as buf, redirect_stdout(buf):
            for x in range(int(self.optimizer.budget/self.opt_options['num_jobs'])):
                with futures.ProcessPoolExecutor(max_workers=self.opt_options['num_jobs']) as executor:
                    futures_list = [executor.submit(self.worker,  self.optimizer.ask()) for _ in range(self.opt_options['num_jobs'])]
                    for future in futures.as_completed(futures_list):
                        x, value = future.result()
                        progress_bar.update(1)

        # Close the progress bar
        progress_bar.close()
        self.callback_df.reset_index(drop=True, inplace=True) 
        # self.results['x'] = list(self.optimizer.provide_recommendation().args)
        min_col = self.callback_df['res_0'].idxmin()
        self.results['x'] = list(self.callback_df.drop(['res_0', 'time'], axis=1).loc[min_col])
        # print(self.results['x'])
        self.results['y'] = self.score(self.results['x'])
        self.opt_time = time.monotonic() - start_time
        print("--- %.1f seconds ---" % self.opt_time)
        print(self.results['x'])
        print(f"X: {self.results['x']}, Y: {self.results['y']}")

    def objfunc(self, x):
        loss = self.score([*x.args])
        return loss

    def worker(self, *x):
        value = self.objfunc(*x)
        self.optimizer.tell(*x, value)
        self.callback([*x], value)
        return *x, value

    def save_results(self):
        metadata_dict = {'time': self.opt_time,
                         'x': self.results['x'],
                         'task': self.task,
                         'performance_score': self.results['y'],
                         'model_output': self.model.predict(np.array(self.results['x'])),
                         'optimizer_class': self.opt_class,
                         'optimizer_name': self.opt_name,
                         'optimizer_options': self.opt_options,
                         'model_class': self.model.model_class,
                         'model_name': self.model.model_name
                         }
        if (self.model.model_class == 'ML-MODEL') & (self.sim_model.model_name == 'SimPAN'):
            metadata_dict['sim_output'] = self.sim_model.predict(self.results['x'])
        if self.sim_model.model_name == 'SimTRC':
            metadata_dict['area'] = self.sim_model.area_from_dim(self.results['x'])

        pickle.dump(metadata_dict, open(os.path.join(self.output_dir, 'metadata_results.pkl'), 'wb'))

        # self.callback_df.to_csv(os.path.join(self.output_dir, 'optimization_results.csv'), index=False)
        self.callback_df.to_pickle(os.path.join(self.output_dir, 'optimization_results.pkl'))
        with open(os.path.join(self.output_dir, 'results.txt'), 'w') as f:
            with redirect_stdout(f):
                print("--- %.1f seconds ---" % self.opt_time)
                print(f"X: {self.results['x']}")
                print(f"Y: {self.results['y']}")
                print(f"Model output: {self.model.predict(np.array(self.results['x']))}")
                print(f"Optimizer_recommendation: {self.optimizer.provide_recommendation()}")
                print(f"Optimizer_x_recommendation: {list(self.optimizer.provide_recommendation().args)}")
                if (self.model.model_class == 'ML-MODEL') & (self.sim_model.model_name == 'SimPAN'):
                    print(f"Sim output: {self.sim_model.predict(self.results['x'])}")
                if self.sim_model.model_name == 'SimTRC':
                    print(f"Area: {metadata_dict['area']}")
