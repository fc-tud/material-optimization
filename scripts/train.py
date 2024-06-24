#!/usr/bin/env python
# coding: utf-8

import sys
sys.path.append('..')
import gc
import argparse
import os

import config
from src.models.ncv import ncv
from src.models.helpers.data_handling import check_dataset


parser = argparse.ArgumentParser()
parser.add_argument("-m", "--model", nargs='?')
args = parser.parse_args()


if __name__ == '__main__':
    for dataset in os.listdir(config.DATA_FOLDER):
        path = os.path.join(config.DATA_FOLDER, dataset)
        files = check_dataset(path)

        print(79 * 'v')
        print(f'Dataset name: {dataset}')
        print()
        name = dataset
        ncv(path, name, args.model)
        gc.collect()
    print(79 * '^')
