import sys
sys.path.append('..')
import os
import pandas as pd
from src.models.sim_trc.SimTRC import SimTRC


optimization_dir = os.path.join('..', 'workdir', 'optimization', 'SIM-TRC', 'SimTRC')
opti_filename = 'optimization_results.pkl'
database_dir = os.path.join('..', 'workdir', 'database', 'SIM-TRC')
data_filename = 'force.csv'

db_path = os.path.join('..', 'results', 'database', 'SIM-TRC')
sim = SimTRC(None, 'SimTRC')


# Function to search for files recursively
def find_files(root_dir, filename):
    matching_files = []
    for root, dirs, files in os.walk(root_dir):
        for file in files:
            if file == filename:
                matching_files.append(os.path.join(root, file))
    return matching_files


def rename_cols(df, x_list, task_list):
    # Create a dictionary to map old column names to new column names
    column_name_mapping = {i: dim['name'] for i, dim in enumerate(x_list)}
    # Rename x-columns using the mapping
    df.rename(columns=column_name_mapping, inplace=True)
    # Rename y-columns according to labels
    for i, y in enumerate(task_list):
        df.columns = df.columns.str.replace(f'res_{i}', y)
        df[y] = -df[y]
    return df


if __name__ == "__main__":
    db = []
    """
    # get ml train data
    ml_train_data = pd.read_csv(os.path.join('..', 'data', 'storage', 'SIM-TRC', 'data.csv'), sep=';')
    ml_train_data['source'] = 'train_data'
    db.append(ml_train_data)
    """
    # Loop through all optimization dirs, find df_callback, add to list, concat and save
    matching_files_opti = find_files(optimization_dir, opti_filename)
    for file_path in matching_files_opti:
        print(file_path)
        df_callback = pd.read_pickle(file_path)
        df_callback = rename_cols(df_callback, sim.space, ['f_res'])
        df_callback = df_callback.loc[df_callback['f_res'] > 0]
        df_callback['source'] = [file_path.split(os.sep)[-3:-1] for _ in range(len(df_callback))]
        db.append(df_callback)
    matching_files_database = find_files(database_dir, data_filename)
    for file_path in matching_files_database:
        print(file_path)
        df_force = pd.read_csv(file_path)
        df_force['source'] = [file_path.split(os.sep)[-2] for _ in range(len(df_force))]
        db.append(df_force)
    db = pd.concat(db, axis=0, ignore_index=True)
    # clean y = 0 rows
    db[sim.X_cols] = db.apply(lambda row: sim.adjust_y_dimension(row[sim.X_cols].tolist()),
                              axis=1, result_type='expand')
    # get current db
    if os.path.isfile(os.path.join(db_path, 'all_runs.csv')):
        #db_current = pd.read_pickle(os.path.join(db_path, 'all_runs.csv'))
        db_current = pd.read_csv(os.path.join(db_path, 'all_runs.csv'), index_col=0)
        db = pd.concat([db_current, db], axis=0, ignore_index=True)
    # drop duplicated rows
    # only check x_0,x_1,x_2,y_0,y_1,y_2,f_res
    
    # Define a lambda function to round the 'f_res' column to 7 decimal places
    # db['f_res'] = db['f_res'].round(7)
    # Drop duplicates based on specified columns and rounded 'f_res' for checking
    # db.drop_duplicates(subset=['x_0', 'x_1', 'x_2', 'y_0', 'y_1', 'y_2', 'f_res'], inplace=True)
    db.drop_duplicates(subset=['x_0', 'x_1', 'x_2', 'y_0', 'y_1', 'y_2'], inplace=True)
    db.dropna(subset=["f_res"], inplace=True)
    db.reset_index(drop=True, inplace=True)
    db.to_csv(os.path.join(db_path, 'all_runs.csv'))
    db.to_pickle(os.path.join(db_path, 'all_runs.pkl'))
