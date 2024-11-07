#!/usr/bin/env python
# coding: utf-8

import sys
import os
import pandas as pd
import numpy as np
import time
import random
from datetime import datetime
import pickle
import concurrent.futures
import uuid
import multiprocessing
import collections
import matplotlib.pyplot as plt
from subprocess import STDOUT, check_output
import subprocess
import psutil
from src.models.ModelBase import ModelBase
from src.models.sim_trc.helpers import define_input


class SimTRC:
    def __init__(self, path, name):
        self.model_name = 'SimTRC'
        self.model_class = 'SIM-MODEL'
        self.name = name
        self.path = path
        self.output_dir = None
        self.sampling = None
        self.run_dir = None
        self.sampling = None
        self.search_space = None
        self.result = collections.defaultdict(dict)
        self.space = [{"name": "x_0", 'domain': (1, 9), "type": 'integer'},
                      {"name": "x_1", 'domain': (1, 9), "type": 'integer'},
                      {"name": "x_2", 'domain': (1, 9), "type": 'integer'},
                      {"name": "y_0", 'domain': (0, 3), "type": 'integer'},
                      {"name": "y_1", 'domain': (1, 3), "type": 'integer'},
                      {"name": "y_2", 'domain': (0, 3), "type": 'integer'},
                      ]
        self.X_cols = [dim['name'] for dim in self.space]

    def create_output_dir(self, mode, task):
        time_run = datetime.now().strftime('%Y_%m_%d__%H_%M_%S')
        if mode == 'database':
            self.sampling = task
            self.output_dir = os.path.join('..', 'workdir', 'database', 'SIM-TRC', f'{self.sampling.__name__}__{time_run}')
        elif mode == 'optimization':
            self.output_dir = task
        else:
            sys.exit('mode must be database or optimization')
        if not os.path.exists(self.output_dir):
            os.makedirs(self.output_dir)
        return self.output_dir

    def create_search_space(self, samples):
        # Check samples for allready run simulations
        done_tasks = pd.DataFrame(columns = self.X_cols + ['f_res'])
        for i, x_input in enumerate(samples):
            y_pred = self.check_db_runs(x_input)
            if y_pred:
                done_tasks = pd.concat([done_tasks, pd.DataFrame([x_input + [y_pred]], columns = self.X_cols + ['f_res'])])
        samples = [x_input for x_input in samples if x_input not in done_tasks[self.X_cols].values.tolist()]
        done_tasks.to_csv(os.path.join(self.output_dir, 'done_tasks.csv'), index=None)
        self.search_space = self.sampling(samples, self.space)
        np.save(os.path.join(self.output_dir, 'search_space.npy'), self.search_space)
        # save metadata
        metadata_dict = {'size': samples,
                         'sampling': self.sampling.__name__,
                         'space': self.space
                         }
        pickle.dump(metadata_dict, open(os.path.join(self.output_dir, 'metadata_data.pkl'), 'wb'))
        return self.search_space

    def create_sub_dirs(self, num, input_dim):
        sim_dir = os.path.join(self.output_dir, f'sim_{num}')
        if not os.path.exists(sim_dir):
            os.makedirs(sim_dir)
            np.save(os.path.join(sim_dir, 'search_space.npy'), input_dim)
        return sim_dir

    def run_sim(self, workdir):
        print(f"Start Sim {os.path.basename(workdir)} - {datetime.now().strftime('%Y_%m_%d__%H_%M_%S')}")
        path_to_py = os.path.join('..', 'src', 'models', 'sim_trc', 'run_sim_trc.py')
        subprocess.run(['python', path_to_py, f'-f{workdir}'])

    def run_parallel_jobs(self, num_jobs):
        sim_dir_list = []
        for n in range(len(self.search_space)):
            sim_dir = self.create_sub_dirs(n, self.search_space[n])
            sim_dir_list.append(sim_dir)
        with concurrent.futures.ProcessPoolExecutor(max_workers=num_jobs) as executor:
            executor.map(self.run_sim, sim_dir_list)
        
    def results_from_sim_folder(self, sim_folder):
        paths = {
            'force': {"filename": "PCantilevera.sum", "columns": ['time', 'f1', 'f2', 'f3']},
            'displacement': {"filename": "PCantilevera.dis", "columns": ['time', 'd1', 'd2', 'd3']}
        }
        for data_type, info in paths.items():
            full_path = os.path.join(sim_folder, info["filename"])
            data = pd.read_csv(full_path, names=info["columns"], sep=r'\s+')
            self.result[sim_folder][data_type] = data
            self.result[sim_folder][data_type]['{}_res'.format(data_type[0])] = (data.iloc[:, 1:] ** 2).sum(axis=1) ** 0.5
        return self.result[sim_folder]['force'], self.result[sim_folder]['displacement']
            
    def plot_result_from_sim_folder(self, sim_folder):
        for y in ['force', 'displacement']:
            for col in self.result[sim_folder][y].columns:
                if col != 'time':
                    plt.scatter(self.result[sim_folder][y]['time'], self.result[sim_folder][y][col], label=col)
            # Add labels and title
            plt.xlabel('Time')
            plt.ylabel(y)
            plt.title(f'{y} over time')
            plt.legend()
            # Show the plot
            plt.show()
    
    def get_results(self, output_dir=None):
        if output_dir:
            self.output_dir = os.path.join('..', 'workdir', output_dir)
        sim_folders = [sim_folder for sim_folder in os.listdir(self.output_dir) if os.path.isdir(os.path.join(self.output_dir, sim_folder))]
        for sim_folder in sim_folders:
            self.results_from_sim_folder(os.path.join(self.output_dir, sim_folder))
        # Save results dict as pickle    
        with open(os.path.join(self.output_dir, 'result_dict.pkl'), 'wb') as pickle_file:
            pickle.dump(self.result, pickle_file)
            
    def build_df(self, output_dir=None):
        if output_dir:
            self.output_dir = os.path.join('..', 'workdir', output_dir)
            self.search_space = np.load(os.path.join(self.output_dir, 'search_space.npy'))
            # Load the object from the pickle file
            with open(os.path.join(self.output_dir, 'result_dict.pkl'), 'rb') as pickle_file:
                self.result = pickle.load(pickle_file)
            
        df_input = pd.DataFrame()
        for i in range(len(self.search_space)):
            df_input.at[i, 'id'] = i
            for j in range(3):
                x = self.search_space[i][:,j][1]*2
                df_input.at[i, f'x_{j}'] = x
            for j in range(3):
                y = self.search_space[i][:, j][3] * 2
                df_input.at[i, f'y_{j}'] = y
                    
        for output in ['displacement', 'force']:
            df = pd.DataFrame(columns=['id'])
            for key, value in self.result.items():
                if len(self.result[key][output]) == 101:
                    # Find the index of the maximum value in f_res
                    max_index = self.result[key]['force']['f_res'].idxmax()
                    #print(max_index)
                    #print(result.result[output].loc[[max_index]])
                    df = pd.concat([df, self.result[key][output].loc[[max_index]]])
                else:
                    df = pd.concat([df, pd.DataFrame(index=[0])])
                df['id'].iat[len(df)-1] = int(key.split('_')[-1])
            df = pd.merge(df_input, df, on='id', how='inner')
            df.to_csv(os.path.join(self.output_dir, f'{output}.csv'), index=None)

    def predict(self, x_input):
        # Set the timeout in seconds (4 hours)
        timeout_seconds = 12 * 60 * 60
        print(x_input)
        # Check if x_input was already evaluated
        y_pred = self.check_db_runs(x_input)
        # Return done run if it exists
        if y_pred:
            return [y_pred]

        x_input = define_input(x_input, self.space)
        print(x_input)
        # Create output dirs
        sim_folders = [sim_folder for sim_folder in os.listdir(self.output_dir) if os.path.isdir(os.path.join(self.output_dir, sim_folder))]
        num = f"{len(sim_folders)}_{str(uuid.uuid4())[:8]}"
        sim_dir = self.create_sub_dirs(num, x_input)

        # command to run the simulation script
        self.run_sim(sim_dir)
        try:
            # Use Popen without timeout
            process = subprocess.Popen(command, stdout=subprocess.PIPE, stderr=subprocess.PIPE, text=True)

            # Wait for the process to finish or timeout
            process.wait(timeout=timeout_seconds)
        except subprocess.TimeoutExpired:
            print('Simulation not finished successfully (timeout reached).')

            # Terminate the process and its children using psutil
            parent = psutil.Process(process.pid)
            for child in parent.children(recursive=True):
                child.terminate()
            parent.terminate()
            parent.wait()
            return [0]
        
        try:
            force, displacement = self.results_from_sim_folder(sim_dir)
        except FileNotFoundError:
            print('Simulation not finished successful')
            return [0]
        """    
        for key, df in {'displacement':displacement, 'force':force}.items():
            df[f'{key[0]}_res'] = (df[f'{key[0]}1']**2 + df[f'{key[0]}2']**2 + df[f'{key[0]}3']**2)**0.5 
        """
        if len(force) == 101:
            max_index = force['f_res'].idxmax()
            y_pred = force['f_res'].loc[max_index]
        else:
            y_pred = 0
        print([y_pred])
        return [y_pred]

    def adjust_y_dimension(self, x_input, value=0):
        for i in range(3):
            if x_input[i+3] == 0:
                x_input[i] = value
        return x_input

    def area_from_dim(self, x_input):
        area = 0
        for i in range(3):
            sub_area = x_input[i]*x_input[i+3]
            area += sub_area
        return area

    def check_db_runs(self, x_input):
        print('We are checking')
        x_input = self.adjust_y_dimension(x_input, 0)
        # load db
        db_path = os.path.join('..', 'results', 'database', 'SIM-TRC', 'all_runs.csv')
        if os.path.isfile(db_path):
            df = pd.read_csv(db_path)
        x_input = pd.DataFrame([x_input], columns=self.X_cols)
        print(x_input)
        merged_df = pd.merge(df, x_input, on=self.X_cols, how='inner')
        print(merged_df)
        if not merged_df.empty:
            print('Got Value from DB', merged_df['f_res'])
            return merged_df.iloc[0]['f_res']
        else:
            return None

