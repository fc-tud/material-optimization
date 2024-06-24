#!/usr/bin/env python
# coding: utf-8

import os
import pandas as pd
import numpy as np
import time
from datetime import datetime
import pickle
import csv
import concurrent.futures
import uuid
import multiprocessing
import collections
import matplotlib.pyplot as plt
import subprocess
from src.models.ModelBase import ModelBase


class SimPAN:
    def __init__(self, path, name):
        self.model_name = 'SimPAN'
        self.model_class = 'SIM-MODEL'
        self.name = name
        self.path = path
        self.output_dir = None
        self.sampling = None
        self.run_dir = None
        self.sampling = None
        self.search_space = None
        self.result = collections.defaultdict(dict)
        self.space = [{"name": "PS:PAN ratio", 'domain': (0.3 / 0.7, 0.7 / 0.7), "type": 'continuous'},
                      {"name": "Feed rate(mL/h)", 'domain': (0.7 / 2, 2 / 2), "type": 'continuous'},
                      {"name": "Distance(cm)", 'domain': (12 / 24, 24 / 24), "type": 'continuous'},
                      {"name": "Mass fraction of solute", 'domain': (12 / 18, 18 / 18), "type": 'continuous'},
                      {"name": "Mass fraction of SiO2 in solute ", 'domain': (0 / 20, 20 / 20), "type": 'continuous'},
                      {"name": "Applied voltage(kV)", 'domain': (16 / 28, 28 / 28), "type": 'continuous'},
                      {"name": "Inner diameter(mm)", 'domain': (0.41 / 1.32, 1.32 / 1.32), "type": 'continuous'}
                      ]

    def create_output_dir(self, sampling, samples):
        # Dir to save runs for creating a database
        # define sampling method
        self.sampling = sampling
        time_run = datetime.now().strftime('%Y_%m_%d__%H_%M_%S')
        self.output_dir = os.path.join('..', 'workdir', 'database', 'SIM-PAN', f'{sampling.__name__}__{samples}__{time_run}')
        if not os.path.exists(self.output_dir):
            os.makedirs(self.output_dir)
        return self.output_dir

    def create_search_space(self, samples):
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

    def build_df(self, output_dir=None):
        if output_dir:
            self.output_dir = os.path.join('workdir', output_dir)
            self.search_space = np.load(os.path.join(self.output_dir, 'search_space.npy'))

        columns_x = [item["name"] for item in self.space]
        df_input = pd.DataFrame(self.search_space, columns=columns_x)
        columns_y = ['wca', 'q', 'sigma']
        df_output = pd.DataFrame(self.predict(df_input), columns=columns_y)
        df = pd.merge(df_input, df_output, left_index=True, right_index=True, how='inner')
        df.to_csv(os.path.join(self.output_dir, 'data.csv'), sep=';', index=None)
        # Write the list to the text file
        with open(os.path.join(self.output_dir, 'X.txt'), 'w', newline='') as file:
            writer = csv.writer(file)
            writer.writerow(columns_x)
        with open(os.path.join(self.output_dir, 'y.txt'), 'w', newline='') as file:
            writer = csv.writer(file)
            writer.writerow(columns_y)

    def predict(self, x_input):
        if isinstance(x_input, pd.DataFrame):
            y_pred = self.func(x_input)
            return y_pred
        # time~0.0002s
        x_input = np.array(x_input).reshape(1, -1)
        y_pred = self.func(x_input)[0]
        return y_pred

    def func(self, x_input):
        if isinstance(x_input, pd.DataFrame):
            x_input = np.array(x_input)
        v1 = x_input[:, 0].reshape(-1, 1)
        v2 = x_input[:, 1].reshape(-1, 1)
        v3 = x_input[:, 2].reshape(-1, 1)
        v4 = x_input[:, 3].reshape(-1, 1)
        v5 = x_input[:, 4].reshape(-1, 1)
        v6 = x_input[:, 5].reshape(-1, 1)
        v7 = x_input[:, 6].reshape(-1, 1)

        wca = -197.0928 - 78.3309 * v1 + 98.6355 * v2 + 300.0701 * v3 + 89.8360 * v4 \
            + 208.2343 * v5 + 332.9341 * v6 + 135.6621 * v7 - 11.0715 * v1 * v2 \
            + 201.8934 * v1 * v3 + 17.1270 * v1 * v4 + 2.5198 * v1 * v5 \
            - 109.3922 * v1 * v6 + 30.1607 * v1 * v7 - 46.1790 * v2 * v3 \
            + 19.2888 * v2 * v4 - 102.9493 * v2 * v5 - 19.1245 * v2 * v6 \
            + 53.6297 * v2 * v7 - 73.0649 * v3 * v4 - 37.7181 * v3 * v5 \
            - 219.1268 * v3 * v6 - 55.3704 * v3 * v7 + 3.8778 * v4 * v5 - 6.9252 * v4 * v6 \
            - 105.1650 * v4 * v7 - 34.3181 * v5 * v6 - 36.3892 * v5 * v7 \
            - 82.3222 * v6 * v7 - 16.7536 * v1 * v1 - 45.6507 * v2 * v2 - 91.4134 * v3 * v3 \
            - 76.8701 * v5 * v5
        wca = np.array(wca).reshape(-1, 1)

        q = -212.8531 + 245.7998 * v1 - 127.3395 * v2 + 305.8461 * v3 + 638.1605 * v4 \
            + 301.2118 * v5 - 451.3796 * v6 - 115.5485 * v7 + 42.8351 * v1 * v2 \
            + 262.3775 * v1 * v3 - 103.5274 * v1 * v4 - 196.1568 * v1 * v5 \
            - 394.7975 * v1 * v6 - 176.3341 * v1 * v7 + 74.8291 * v2 * v3 \
            + 4.1557 * v2 * v4 - 133.8683 * v2 * v5 + 65.8711 * v2 * v6 \
            - 42.6911 * v2 * v7 - 323.9363 * v3 * v4 - 107.3983 * v3 * v5 \
            - 323.2353 * v3 * v6 + 46.9172 * v3 * v7 - 144.4199 * v4 * v5 \
            + 272.3729 * v4 * v6 + 49.0799 * v4 * v7 + 318.4706 * v5 * v6 \
            - 236.2498 * v5 * v7 + 252.4848 * v6 * v7 - 286.0182 * v4 * v4 \
            + 393.5992 * v6 * v6
        q = np.array(q).reshape(-1, 1)

        sigma = 7.7696 + 15.4344 * v1 - 10.6190 * v2 - 17.9367 * v3 + 17.1385 * v4 + 2.5026 * v5 \
            - 24.3010 * v6 + 10.6058 * v7 - 1.2041 * v1 * v2 - 37.2207 * v1 * v3 \
            - 3.2265 * v1 * v4 + 7.3121 * v1 * v5 + 52.3994 * v1 * v6 + 9.7485 * v1 * v7 \
            - 15.9371 * v2 * v3 - 1.1706 * v2 * v4 - 2.6297 * v2 * v5 \
            + 7.0225 * v2 * v6 - 1.4938 * v2 * v7 + 30.2786 * v3 * v4 + 14.5061 * v3 * v5 \
            + 48.5021 * v3 * v6 - 11.4857 * v3 * v7 - 3.1381 * v4 * v5 \
            - 14.9747 * v4 * v6 + 4.5204 * v4 * v7 - 17.6907 * v5 * v6 - 19.2489 * v5 * v7 \
            - 9.8219 * v6 * v7 - 18.7356 * v1 * v1 + 12.1928 * v2 * v2 - 17.5460 * v4 * v4 \
            + 5.4997 * v5 * v5 - 26.2718 * v6 * v6
        sigma = np.array(sigma).reshape(-1, 1)

        y = np.concatenate([wca, q, sigma], axis=1) # * (-1)  # non-scaled
        return y