import sys
sys.path.append('..')

import os
import pandas as pd
import numpy as np
import re
import contextlib
import pickle
import copy

from src.models.helpers.data_handling import regression_results

import matplotlib.pyplot as plt
import plotly.graph_objects as go
plt.rc('axes', axisbelow=True) # grid in background


ml_color_dict = {'database':'grey', 'CatBoost': '#6c65b5', 'autosklearn': '#62cfac', 'FFNN_mtl':'#db7707'}
sim_color_dict = {'database':'grey', 'TPDE': '#6c65b5', 'NoisyDE': '#62cfac'}
box_plot_dict = {'sim':'#00305e', 'model':'#62cfac'}

def min_max_scaling(value, min_value, max_value):
    return (value - min_value) / (max_value - min_value)

def get_val_for_str(dictionary, substring):
    for key, value in dictionary.items():
        if key in substring:
            return value
    return None

class Result():
    def __init__(self, path):
         self.optimizer_class = None
         self.optimizer_name = None
         self.optimizer_options = None
         self.x_names = None
         self.y_names = None
         self.df = None
         self.folder_name = os.path.basename(path)
         self.folder_path = path
         self.time = None
         self.model_name = None
         self.model_output = None
         self.sim_output = None
         self.task = None
         
    def get_metadata(self):
        pkl_file_path = os.path.join(self.folder_path, 'metadata_results.pkl')
        if os.path.exists(pkl_file_path):
            # open a file, where you stored the pickled data
            file = open(os.path.join(self.folder_path, 'metadata_results.pkl'), 'rb')
            metadata_dict = pickle.load(file)
            attribute_names = ['time', 'optimizer_class', 'optimizer_name', 'optimizer_options', 'model_class', 'model_name', 'model_output', 'task']
            for attr in attribute_names:
                setattr(self, attr, metadata_dict[attr])
            if not isinstance(self.task, tuple):
                self.task = (self.task,)
            if 'sim_output' in metadata_dict:
                self.sim_output = metadata_dict['sim_output']       
        else:
            return
            constraints = self.folder_path.split('(')[1].split(')')[0]
            constraints = constraints.split(',')
            self.task = [float(value) if value.strip().lower() != 'none' else None for value in constraints]
           
            # Read the content of the results file
            with open(os.path.join(self.folder_path, 'results.txt'), 'r') as file:
                file_content = file.read()
            # Extract time using regular expression
            time_match = re.search(r'--- (\d+\.\d+) seconds ---', file_content)
            self.time = float(time_match.group(1)) if time_match else None
        
            # Extract model output using regular expression
            model_output_match = re.search(r'Model output: \[\[(.*?)\]\]', file_content)
            model_output_str = model_output_match.group(1) if model_output_match else None
            # Extract numeric values from the model output string using findall
            self.model_output = [float(value) for value in re.findall(r'[\d.]+', model_output_str)] if model_output_str else None
    
    def get_data(self):
        # Check if the folder contains either a CSV or a pickle file
        csv_file_path = os.path.join(self.folder_path, 'optimization_results.csv')
        pkl_file_path = os.path.join(self.folder_path, 'optimization_results.pkl')
        if os.path.exists(csv_file_path):
            # Read CSV file
            self.df = pd.read_csv(csv_file_path)
            # print(f"Reading CSV file from {self.folder_path}")
        elif os.path.exists(pkl_file_path):
            # Read pickle file
            self.df = pd.read_pickle(pkl_file_path)
            # print(f"Reading pickle file from {self.folder_path}")
        else:
            return
            # print(f"No optimization results file found in {self.folder_path}")
        if len(self.df.index.unique()) != len(self.df):
            self.df = self.df.reset_index(drop=True)
        self.y_names = [s for s in self.df.columns if  isinstance(s, str) and 'res_' in s]
        self.x_names = [item for item in self.df.columns if item not in self.y_names]

    def get_optima(self, n = None):
        if self.df is None:
            self.get_data()
        if n:
            self.df = self.df.iloc[:n]
        #Achtung Callback ist noch mit - Vorzeichen aus der Optimierung!
        # Identify the column with the maximum value
        max_col = self.df[self.y_names[0]].idxmin()
        self.x = self.df[self.x_names].loc[[max_col]]
        self.y = min(self.df[self.y_names[0]])

    def plot_(self):
        #plot y over iter
        self.df['res_max'] = self.df['res_0'].cummin()
        plt.plot(self.df.index, -self.df['res_max'], color='#00305e')
        plt.show()
             

class Evaluation_Base():
    def __init__(self, workdir, use_case, task_list):
        self.workdir = workdir
        self.use_case = use_case
        self.task_list = task_list
        self.database = pd.DataFrame()
        self.all_results = []
        self.df_opti = pd.DataFrame()

    def get_results(self):
        for model in os.listdir(os.path.join(self.workdir, self.use_case)):
            print(model)
            for optimizer in os.listdir(os.path.join(self.workdir, self.use_case, model)):
                print(optimizer)
                for folder in os.listdir(os.path.join(self.workdir, self.use_case, model, optimizer)):
                    # print(folder)
                    folder_path = os.path.join(self.workdir, self.use_case, model, optimizer, folder)
                    results = Result(os.path.join(folder_path))
                    results.get_metadata()
                    results.get_data()
                    if (results.df is None) or (results.df.empty):
                        continue
                    results.get_optima()
                    self.all_results.append(results)
    
    def get_steps_opti_results(self, steps, models):
        # list all results where int results should be added
        # loop trough list and append to self.all_results
        for model in models:
            for result in [result for result in self.all_results if (result.model_name == model)]:
                for step in steps:
                    step_results = copy.deepcopy(result)
                    step_results.optimizer_options['iterations'] = step
                    step_results.get_optima(n=step)
                    self.all_results.append(step_results)

        
    def get_all_optima(self):
        results_list = []
        for result in self.all_results:
            x = result.x
            y = result.y
            x['y']= -y
            for i, value in enumerate(result.task):
                x[f'con_{i}'] = result.task[i]
            for i, model_output in enumerate(result.model_output):
                x[f'model_output_{i}'] = model_output
    
            # fill all Nan in con with the marker opti as this is the optimization target
            subset_columns = [col for col in x.columns if 'con_' in str(col)]
            string_value = 'opti'
            x[subset_columns] = x[subset_columns].fillna(string_value)
            # add metadata 
            attributes_to_add = ['time', 'optimizer_class', 'optimizer_name', 'model_name']
            for attr in attributes_to_add:
                x[attr] = getattr(result, attr)
            if result.sim_output is not None:
                for i, sim_output in enumerate(result.sim_output):
                    x[f'sim_output_{i}'] = sim_output
            x['max_opt_iter']= result.optimizer_options['iterations']
            results_list.append(x)
        #add best iter
        self.df_opti = pd.concat(results_list, axis = 0, ignore_index=True)
        self.df_opti = self.df_opti.sort_values(by=['model_name', 'optimizer_name', 'max_opt_iter'])
        self.df_opti = self.df_opti.reset_index(drop=True)
        self.df_opti = self.df_opti.rename(columns={"index": "best_iter"})


    def load_database(self, ml_model):
        storage_path = os.path.join('..', 'ml_models_bin', self.use_case, ml_model, 'model.pkl')
        model = pickle.load(open(storage_path, 'rb'))
        self.database = model.data.loc[model.train_index]
        self.database['model_name']='database'
        self.database['optimizer_name']='database'
        self.database['max_opt_iter']=0

    
    def rename_opti_cols(self, x_list, y_dict=None):
        # Create a dictionary to map old column names to new column names
        column_name_mapping = {i: dim['name'] for i, dim in enumerate(x_list)}
        # Rename x-columns using the mapping
        self.df_opti.rename(columns=column_name_mapping, inplace=True)
        # Rename y-colums according to labels
        if y_dict:
            self.df_opti.rename(columns=y_dict, inplace=True)
        else:
            for i, y in enumerate(self.task_list):
                self.df_opti.columns = self.df_opti.columns.str.replace(f'_{i}', f'_{y}')
    

class Evaluation():
    def __init__(self, evaluation_base, task_constrains, opti_constrains, pareto_ascending):
        self.base = evaluation_base
        self.task_constrains = task_constrains
        self.opti_constrains = opti_constrains
        self.pareto_ascending = pareto_ascending
        self.df_task = pd.DataFrame()
        self.database = pd.DataFrame()
        self.exp_dict = {}
        self.exp_overview = pd.DataFrame()
        self.opti_traget = [key.split('_')[1] for key, item in self.task_constrains.items() if item == 'opti'][0]
        self.opti_x = list(set(self.base.task_list) - set([key.split('_')[1] for key, item in self.task_constrains.items()]))[0]
        self.scores = pd.DataFrame(columns=['quantile_score', 'r2', 'rmse', 'mae', 'mse', 'mape', 'model', 'task', 'optimizer'])  


    def get_optimisation_task(self):
        self.datapoints = self.base.database.copy()
        for key, item in self.task_constrains.items():
            self.df_task = self.base.df_opti.loc[(self.base.df_opti[key]==item)]
            if item != 'opti':
                self.datapoints = self.datapoints.loc[self.datapoints[key.split('_')[-1]]>=float(item)]
        else:
            self.df_task = self.base.df_opti.copy()
        if self.opti_constrains:
            for key, item in self.opti_constrains.items():
                self.df_task = self.df_task.loc[(self.df_task[key]==item)]
        self.datapoints = self.datapoints[(self.datapoints[self.opti_x] >= 0) & (self.datapoints[self.opti_traget] >= 0)]

    def group_results(self, grouping=None):
        if grouping == None:
            grouping = ['model_name', 'optimizer_name', 'max_opt_iter']
        grouped = self.df_task.groupby(grouping)
        # Create a df to store subsets
        self.exp_overview = pd.DataFrame()
        # Iterate over each unique combination and store the corresponding subset
        for group_name, group_df in grouped:
            exp_dict = {}
            for i, n in enumerate(grouping):
                exp_dict[n] = group_name[i]
            exp_dict['data'] = group_df.reset_index(drop=True)
            self.exp_overview = pd.concat([self.exp_overview, pd.DataFrame([exp_dict])], axis=0, ignore_index = True)
  
    def get_opti_datapoints(self):
        datapoints = self.datapoints.sort_values(by=[self.opti_x], ascending=self.pareto_ascending)      
        datapoints[f'cummax_{self.opti_traget}']= datapoints[self.opti_traget].cummax()
        datapoints = datapoints[datapoints[f'cummax_{self.opti_traget}'] == datapoints[self.opti_traget]]
        datapoints = datapoints[~datapoints.duplicated(subset=self.opti_x, keep='last')]
                
        # Duplicate the last row with the constant 'y' value
        if self.pareto_ascending:
            min_row = datapoints.iloc[[0]].copy()
            max_row = datapoints.iloc[[-1]].copy()
            max_row[self.opti_traget] = 0
        if not self.pareto_ascending:
            min_row = datapoints.iloc[[-1]].copy()
            max_row = datapoints.iloc[[0]].copy()
            max_row[self.opti_traget] = 0
        min_row[self.opti_x ] = 0 #self.df_task[f'con_{self.opti_x}'].min()
        max_row[self.opti_x ] = self.df_task[f'con_{self.opti_x}'].max()
        datapoints = pd.concat([min_row, datapoints, max_row], ignore_index=True)
        for domain in ['model_output_', 'sim_output_', 'con_']:
            for y in self.base.task_list:
                datapoints.loc[:,domain+y] = datapoints[y]
                self.datapoints.loc[:,domain+y] = self.datapoints[y]
        datapoints['y'] = 0
        exp_dict = {'model_name':'database', 'optimizer_name':'database', 'max_opt_iter':0, 'data': datapoints}
        self.exp_overview = pd.concat([self.exp_overview, pd.DataFrame([exp_dict])], axis=0, ignore_index = True)

    def calc_error(self, task, exclude=[]):
        # surpress printing
        if task == 'all':
            df = self.base.df_opti
        if task == 'task':
            df = self.df_task
        self.scores = pd.DataFrame(columns=self.scores.columns)
        task_list = list(set(self.base.task_list).difference(exclude))
        #filtered_rows = self.exp_overview[~self.exp_overview.model_name.str.contains('sim|database', case=False)]
        with open(os.devnull, "w") as f, contextlib.redirect_stdout(f):    
            for index, exp in self.exp_overview.loc[self.filter_index(model_substrings=['Sim', 'database'], inclusive=False)].iterrows():
                for task in task_list:
                    exp['data'] = exp['data'].dropna(subset=[f'sim_output_{task}', f'model_output_{task}'])
                    model_pred_power = regression_results(exp['data'][[f'sim_output_{task}']], exp['data'][[f'model_output_{task}']])
                    model_pred_power.extend([exp['model_name'], task, exp['optimizer_name']])
                    self.scores.loc[len(self.scores)] = model_pred_power

    def calc_cum_max_optimizer(self, cols=None):
        if cols is None:
            cols = ['y']
        # Calc cum max for each model + (database)
        for index, exp in self.exp_overview.iterrows():
            df = exp['data']
            for col in cols:
                df = df.sort_values(by=[f'con_{self.opti_x}'], ascending=self.pareto_ascending)
                df[f'cummax_{col}'] = df[col].cummax().clip(lower=0)
                self.exp_overview.at[index, 'data'] = df
                #self.exp_dict[key] = df
                #self.df_task.loc[df.index, f'cummax_{col}'] = df[f'cummax_{col}'] 
               
    def calc_cum_max(self, cols=None):
        if cols is None:
            cols = [f'model_output_{self.opti_traget}', f'sim_output_{self.opti_traget}']
        # Calc cum max for each model + (database)
        for index, exp in self.exp_overview.iterrows():
            df = exp['data']
            for col in cols:
                model_type = col.split('_')[0]
                for con in df[f'con_{self.opti_x}'].unique():
                    df = df.sort_values(by=[f'con_{self.opti_x}'], ascending=self.pareto_ascending)
                    if self.pareto_ascending:
                        df.loc[df[f'con_{self.opti_x}']==con, f'cummax_{col}'] = df.loc[df[f'{model_type}_output_{self.opti_x}']<=con, col].clip(lower=0).max()
                    else:
                        df.loc[df[f'con_{self.opti_x}']==con, f'cummax_{col}'] = df.loc[df[f'{model_type}_output_{self.opti_x}']>=con, col].clip(lower=0).max()
            self.exp_overview.at[index, 'data'] = df

    def calc_normalized_score(self, cols=None, sim='all'):
        if cols is None:
            cols = [f'model_output_{self.opti_traget}', f'sim_output_{self.opti_traget}']
        data = self.get_df(model_substrings=['database'])
        # Option if df_opti or df shall be used (optimal sim results with same opti options --> df, best results over all options --> df_opti)
        if sim == 'all':
            sim = self.base.df_opti.loc[(self.base.df_opt['model_name'].str.contains('Sim')) & (self.base.df_opt[f'con_{self.opti_x}']!='opti')]
            sim.astype({f'con_{self.opti_x}': 'float'})
        if sim == 'task':
            sim = self.get_df(model_substrings=['Sim'])
        
        for index, exp in self.exp_overview.loc[self.filter_index(model_substrings=['Sim', 'database'], inclusive=False)].iterrows():
            df = exp['data']
            for con in df[f'con_{self.opti_x}'].unique():
                for col in cols:
                    data_column = data[f'{self.opti_traget}']
                    sim_column = sim[f'model_output_{self.opti_traget}']
                    
                    if self.pareto_ascending:
                        data_query = data[f'con_{self.opti_x}'] <= con
                        sim_query = sim[f'con_{self.opti_x}'] <= con
                    else:
                        data_query = data[f'con_{self.opti_x}'] >= con
                        sim_query = sim[f'con_{self.opti_x}'] >= con
            
                    data_value = np.nanmax([data_column.loc[data_query].max(), 0])
                    max_value = sim_column.loc[sim_query].clip(lower=0).max()
                    df.loc[df[f'con_{self.opti_x}']==con, f'normed_{col}'] = min_max_scaling(df.loc[df[f'con_{self.opti_x}']==con, f'cummax_{col}'], data_value, max_value)
                    #df[f'normed_{col}'].where(df[f'normed_{col}'] < 1, 1, inplace=True)
            df.replace([np.inf, -np.inf], np.nan, inplace=True)
            self.exp_overview.at[index, 'data'] = df
 
    def calc_opti_score(self, cols=None, con_col = 'con_q', models=['Sim']):
        if cols is None:
            cols = [f'model_output_{self.opti_traget}']
        # Option if df_opti or df shall be used (optimal sim results with same opti options --> df, best results over all options --> df_opti)
        sim = self.get_df(model_substrings=models)
        for index, exp in self.exp_overview.iterrows():
            df = exp['data']
            for con in df[con_col].unique():
                for col in cols:
                    #min_value = sim[f'y'].loc[(sim[f'con_{self.opti_x}']>=con)].clip(lower=0).min()
                    min_value = 0
                    if self.pareto_ascending:
                        max_value = sim[f'y'].loc[(sim[f'con_{self.opti_x}']<=con)].clip(lower=0).max()
                    else:
                        max_value = sim[f'y'].loc[(sim[f'con_{self.opti_x}']>=con)].clip(lower=0).max()
                    if max_value == 0:
                        df.loc[df[f'con_{self.opti_x}']==con, f'normed_opti_{col}'] = 0
                    else:
                        # Use cummax?
                        df.loc[df[f'con_{self.opti_x}']==con, f'normed_opti_{col}'] = min_max_scaling(df.loc[df[f'con_{self.opti_x}']==con, col], min_value, max_value)
            df.replace([np.inf, -np.inf], np.nan, inplace=True)
            self.exp_overview.at[index, 'data'] = df

    def plot_2d_pareto(self, df=None, x=None, y_list=None, label=None, color_dict=ml_color_dict, save_fig = None):
        if df is None:
            df =  self.get_df()
            
        df = df[(df[x] >= 0) & (df[y_list[0]] >= 0)]
        datapoints =  self.datapoints[(self.datapoints[x.split('_')[-1]] >= 0) & (self.datapoints[y_list[0].split('_')[-1]] >= 0)]
        params = {}
        fig, ax = plt.subplots()
        
        if self.pareto_ascending:
            drawstyle = 'steps-post'
        else:
            drawstyle = 'steps-pre'

        dict_ = {f'cummax_model_output_{self.opti_traget}':1.0 , f'cummax_sim_output_{self.opti_traget}':1}

        ax.plot(datapoints[x.split('_')[-1]], 
                 datapoints[y_list[0].split('_')[-1]],
                 c='grey',
                 #label = 'database',
                 marker='.',
                 linestyle=''
                )
 
        for y in y_list:
            for key, color in color_dict.items():
                if key == 'database':
                    params['markevery']=slice(1, -1)
                else:
                    params['markevery']=None
                params['alpha'] = dict_[y]
                ax.plot(df[x].loc[df[label]==key],
                         df[y].loc[df[label]==key],
                         c=color,
                         label = key,
                         marker='.',
                         drawstyle='steps-post',
                         **params)
        #ax.axhline(y=self.base.database.sigma.max(), color='grey', linestyle='--', alpha=0.5, label = 'max database')
        #ax.axvline(x=self.base.database.q.max(), color='grey', linestyle='--', alpha=0.5)
        ax.legend()
        ax.grid()
        if save_fig:
            ax.savefig(os.path.join('..', 'plots', USE_CASE, save_fig))
        #plt.plot()
        return fig

    def plot_summarize(self, df=None, targets=None, group='model_name', plot_dict=box_plot_dict, print_error=False, save_fig=False, y_range=None):
        if targets is None:
            targets = [f'normed_sim_output_{self.opti_traget}', f'normed_model_output_{self.opti_traget}']

        if df is None:
            # remove database and Sim for ploting  
            df_plot =  self.get_df(model_substrings=['Sim', 'database'], inclusive = False)
        else:
            df_plot = df.copy()
        layout = go.Layout(yaxis=dict(title='Normed Performance Gain', title_font={'size': 20}, tickfont={'size': 16},linecolor='black', gridcolor='#cccccc',
                                     zeroline=True, zerolinecolor='#cccccc', zerolinewidth=1),
                           xaxis=dict(title=None, linecolor='black', title_font={'size': 20}, tickfont={'size': 16}),
                           boxmode='group',
                           plot_bgcolor='white',
                           legend=dict(font_size=18, orientation="h", yanchor="top", y=1.25, xanchor='center', x=0.5)
        )
    
        fig = go.Figure(layout=layout)
        if print_error:
            for model in df_plot.model_name.unique():
                optimizer = df_plot['optimizer_name'].loc[df_plot['model_name']==model].unique()[0]
                RMSE = round(self.scores.loc[(self.scores['model'] == model) & (self.scores['task'] == targets[0].split('_')[-1]) 
                                            & (self.scores['optimizer'] == optimizer), 'rmse'].item(),3)
                df_plot.loc[df_plot['model_name']== model, 'model_name'] = df_plot.loc[df_plot['model_name']==model, 'model_name'] + '<br>' + f"{RMSE}"
            
        for target in targets:
            fig.add_trace(go.Box(
                y=df_plot[target],
                x=df_plot[group],
                name=target,#f"{' '.join(target.split('_')[0:2])}",
                marker_color= get_val_for_str(plot_dict, target),
                #marker_color=#plot_dict[target.split('_')[1]],
                # line_color='black',
                line_width=1.5,
                #boxpoints='all'
            ))
    
        #fig.add_hline(y=1, line_color="black")
        fig.update_traces(whiskerwidth=1, selector=dict(type='box'))
        fig.update_traces(quartilemethod="exclusive")  # or "inclusive", or "linear" by default
        if y_range:
            fig.update_layout(yaxis_range=y_range)
        if save_fig:
            fig.write_image(save_fig)
        #fig.show()
        return fig

    def filter_index(self, model_substrings=None, optimizer_substrings=None, max_iters=None, inclusive=True):
        """
        Function to retrieve relevant dataframes from a nested dictionary based on substrings in model names, optimizer names, and max_iter values.
        Parameters:
            nested_dfs (dict): Nested df containing model_name, optimizer_name, and max_iter and the data as df
            model_substrings (list): List of substrings to match in model names.
            optimizer_substrings (list): List of substrings to match in optimizer names.
            max_iters (list): List of max_iter values to filter the dataframes.
        Returns:
            df: concatination of all filtered df
        """
        index_list = []
        # Iterate through the nested dictionary
        for index, exp in self.exp_overview.iterrows():
            match = []
            # Check if any substring in model_substrings matches the model name
            if model_substrings:
                #print(exp['model_name'])
                #print(any(substring in exp['model_name'] for substring in model_substrings))
                match.append(any(substring in exp['model_name'] for substring in model_substrings))
            # Check if any substring in optimizer_substrings matches the optimizer name
            if optimizer_substrings:
                match.append(any(substring in exp['optimizer_name'] for substring in optimizer_substrings))
            # Check if the max iter matches any value in max_iters
            if max_iters:
                match.append(exp['max_opt_iter'] in max_iters)
            # If all conditions are met, append the dataframe to the list of relevant dataframes
            if inclusive:
                # Include the dataframe if any condition is met
                include = all(match)
            else:
                # Exclude the dataframe if any condition is met
                include = not all(match)
            if include:
                index_list.append(index)        
        return index_list
    
    def get_df(self, model_substrings=None, optimizer_substrings=None, max_iters=None, inclusive=True):
        index = self.filter_index(model_substrings, optimizer_substrings, max_iters, inclusive)
        relevant_dfs = [exp['data'] for index, exp in self.exp_overview.loc[index].iterrows()]
        if relevant_dfs:
            relevant_dfs = pd.concat(relevant_dfs, axis=0).reset_index(drop=True)
        else:
            relevant_dfs = pd.DataFrame()
        return relevant_dfs



def split_dataframe(df, num_splits):
    """
    Split a DataFrame into sub DataFrames evenly.

    Parameters:
        df (DataFrame): The DataFrame to split.
        num_splits (int): The number of splits to create.

    Returns:
        list: A list containing sub DataFrames.
    """
    total_rows = len(df)
    rows_per_split = total_rows // num_splits
    remainder = total_rows % num_splits
    splits = []
    start_row = 0

    for i in range(num_splits):
        if i < remainder:
            end_row = start_row + rows_per_split + 1
        else:
            end_row = start_row + rows_per_split

        split_df = df.iloc[start_row:end_row]
        splits.append(split_df)

        start_row = end_row

    return splits

