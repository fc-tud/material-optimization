# Optimizing Material-Designs with ML

The code in this repository accompanies the paper _“Navigating Design Space: Machine Learning for Multi-Objective Optimal Material Designs with Comprehensive Performance Evaluation“_.


## Getting Started 

A basic linux machine with an installation of anaconda is able to run the code.
With the comand:
```
python train.py -m <Model>
```
The training for the defined model starts for all datasets in the folder `data/run`, with the configurations set in the `config.py` file.

With the comand:
```
python opti_sim_pan.py
```
and 
```
python opti_sim_trc.py
```
The optimization for the respective use case with the in the script defined model and optimizer, with the configurations set in the `config.py` file.
The `model.pkl` saved in the folder `ml_models_bin/Use-Case/Model-name` is used as the model.
The saved `model.pkl` used in the study are originally stored in the corresponding folders.

We recommend starting the script from a `tmux` session. 

## Console Output
### Model Training
```
vvvvvvvvvvvvvvvvvvvvvvvvvvvvvvvvvvvvvvvvvvvvvvvvvvvvvvvvvvvvvvvvvvvvvvvvvvvvvvv
Dataset name: dataset-name, type: (full/split)

---------------------------------------
Task name: 'dataset-name'_'task-name'
train_size for outer loop = X_train-size
---------- SPLIT 1 ---------- (outer split)
*Starting Time*
Training over X_training-time min started

Results from this outer split

---------- SPLIT 2 ---------- (outer split)
*Starting Time*
Training over X_training-time min started

Results from this outer split

[...all outer splits...]


^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^

Next Task

^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^
vvvvvvvvvvvvvvvvvvvvvvvvvvvvvvvvvvvvvvvvvvvvvvvvvvvvvvvvvvvvvvvvvvvvvvvvvvvvvvv

Next Dataset

^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^

```

## Saved Results 

### Model Training
All results are saved in the directory `workdir/models` in the following folder: `dataset-name\y_label\start-time`.  
In this folder is saved the file `regression_summary.csv`, in which all performance metrics for each fold are stored.  
In the folders `split_'x'` the outputs of the AutoML frameworks are stored, which are framework-specific.

### Optimization
All results are saved in the directory `workdir/optimization` in the following folder: `dataset-name\model\optimizer-name\opt-task_start-time`.
In this folder are saved the files:
- `results.txt`: in which all results of the optimization are stored
- `optimization_result.pkl`: in which all optimization steps are stored
- `optimization_result.pkl`: in which all configurations and final results of the optimization are stored


### Plot Results
The code for visualizing the results and reproducing the figures from the paper can be found in the follwoing scripts:
- `evaluate_opti_pan.ipynb`
- `evaluate_opti_trc.ipynb`
- `plot_sim_trc_f_over_dis.ipynb`
- `plot_splitting_strategies.ipynb`


## Datasets and License

The datasets used in the study are stored in the folder `data/storage`