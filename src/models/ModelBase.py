#!/usr/bin/env python
# coding: utf-8

import os
import pandas as pd
import numpy as np
from datetime import datetime
import pickle
from sklearn.preprocessing import LabelEncoder, MinMaxScaler, PowerTransformer

import config


class ModelBase:
    def __init__(self, path, name):
        self.path = path
        self.name = name
        self.model_name = ''
        self.start_time = datetime.now().strftime('%Y_%m_%d__%H_%M_%S')
        self.output_dir = None
        self.split_path = None
        self.dirs = None
        self.files = None
        self.scaler_X = None
        self.scaler_y = None
        self.data = None
        self.X_data = pd.DataFrame()  # df from csv
        self.y_data = pd.DataFrame()  # df from csv
        self.X = pd.DataFrame()  # df scaled
        self.y = pd.DataFrame()  # df scaled
        self.X_cols = None
        self.y_label = None
        self.train_size = None
        self.group_split_col = None
        self.groups_outer = None
        self.groups_inner = None
        self.train_index = None
        self.test_index = None
        self.model = None
        self.num_inputs = None
        self.input_size = None
        self.num_tasks = None

    def get_X_and_y_and_train_size(self):
        try:
            with open(os.path.join(self.path, 'X.txt'), 'r') as f:
                self.X_cols = f.read().rstrip('\n').split(',')
        except FileNotFoundError:
            print('X.txt not found!')
            self.X_cols = None

        try:
            with open(os.path.join(self.path, 'y.txt'), 'r') as f:
                self.y_label = f.read().rstrip('\n').split(',')
        except FileNotFoundError:
            print('y.txt not found!')
            self.y_label = None

        try:
            with open(os.path.join(self.path, 'train_size.txt'), 'r') as f:
                self.train_size = float(f.read().rstrip('\n'))
                if self.train_size > 1 :
                    self.train_size = int(self.train_size)
                print('train_size for outer loop = {}'.format(self.train_size))
        except FileNotFoundError:
            print('train_size.txt not found! train_size is set to {}'.format(config.DEFAULT_TRAIN_SIZE))
            self.train_size = config.DEFAULT_TRAIN_SIZE

        try:
            with open(os.path.join(self.path, self.name, 'group_split.txt'), 'r') as f:
                self.group_split_col = f.read().rstrip('\n')
                print(f'Group Split with group {self.group_split_col}')
        except FileNotFoundError:
            pass


    def read_data(self):

        def fix_categorical(df):
            # prevent unknown input type error for string input types
            stringcols = df.select_dtypes(include='object').columns
            df[stringcols] = df[stringcols].astype('category')
            return df

        self.get_X_and_y_and_train_size()

        self.data = pd.read_csv(os.path.join(self.path, config.NAME_DATA), delimiter=';')
        self.data = fix_categorical(self.data)
        self.X_data = self.data[self.X_cols]
        self.num_inputs = self.X_data.shape[1]
        self.input_size = self.X_data.shape[0]
        self.y_data = self.data[self.y_label]
        self.num_tasks = self.y_data.shape[1]
        # self.train_index = self.X.index

        if self.group_split_col:
            le = LabelEncoder()
            self.groups_outer = le.fit_transform(self.data[self.group_split_col])

    def scale(self):
        self.scaler_X = MinMaxScaler()
        self.scaler_X = self.scaler_X.fit(self.X_data.loc[self.train_index])
        self.X = pd.DataFrame(self.scaler_X.transform(self.X_data), columns=self.X_data.columns, index=self.X_data.index)
        self.scaler_y = MinMaxScaler()
        self.scaler_y = self.scaler_y.fit(self.y_data.loc[self.train_index])
        self.y = pd.DataFrame(self.scaler_y.transform(self.y_data[self.y_data.columns]), columns=self.y_data.columns, index=self.y_data.index)

    def rescale(self, preds):
        preds[preds.columns] = self.scaler_y.inverse_transform(preds[preds.columns])
        return preds

    def create_output_dir(self):
        time_run = datetime.now().strftime('%Y_%m_%d__%H_%M_%S')
        self.output_dir = os.path.join('..', 'workdir', 'models', self.name, self.model_name,
                                       f'{time_run}')
        if not os.path.exists(self.output_dir):
            os.makedirs(self.output_dir)

    def create_sub_dirs(self, i):
        self.split_path = os.path.join(self.output_dir, 'split_{}'.format(i))
        if not os.path.exists(self.split_path):
            os.makedirs(self.split_path)




