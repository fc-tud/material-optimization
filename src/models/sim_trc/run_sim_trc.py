import numpy as np
import os
import glob
from datetime import datetime
import argparse
import shutil

parser = argparse.ArgumentParser()
parser.add_argument('-f', '--folder')
args = parser.parse_args()
path_to_exe= os.path.join(os.path.join('..', 'src', 'models', 'sim_trc', 'exeDir'))
REMOVE_LIST = ['fort.12', 'fort.1111', 'fort.1121', 'fort.1161', 'fort.1171', 'fort.6666']


def get_sample_dim(workdir):
    search_space = np.load(os.path.join(workdir, 'search_space.npy'))
    coord_x_inf = search_space[0]
    coord_x_sup = search_space[1] 
    coord_y_inf = search_space[2]
    coord_y_sup = search_space[3]
    return coord_x_inf, coord_x_sup, coord_y_inf, coord_y_sup


def copy_sim_files(workdir):
    shutil.copyfile(os.path.join(path_to_exe, 'Feap86.exe'), os.path.join(workdir, 'Feap86.exe'))
    shutil.copyfile(os.path.join(path_to_exe, 'iCantilever.txt'), os.path.join(workdir, 'iCantilever.txt'))


def del_sim_files():
    # Executed in workdir
    os.remove(os.path.join('Feap86.exe'))
    os.remove(os.path.join('iCantilever.txt'))

    def del_files(file):
        # Executed in workdir
        if isinstance(file, str):
            if os.path.exists(file):
                os.remove(os.path.join(file))
        if isinstance(file, list):
            for f in file:
                if os.path.exists(f):
                    os.remove(f)

    def del_files_by_extension(file_extension):
        # Construct the search pattern for files in the specified format
        workdir = os.getcwd()
        search_pattern = os.path.join(workdir, f'*{file_extension}')

        # Use glob to get a list of files matching the pattern
        files_to_remove = glob.glob(search_pattern)
        # print('Files to remove', files_to_remove)

        # Remove each file in the list
        for file_path in files_to_remove:
            try:
                os.remove(file_path)
                # print(f'Removed: {file_path}')
            except Exception as e:
                print(f'Error removing {file_path}: {e}')
    del_files(REMOVE_LIST)
    del_files_by_extension('.vtu')


def run_sim(workdir):
    ''' three block and one plate'''
    
    # count run number for optimization
    """
    if num == None:
        sim_folders = [name for name in os.listdir(os.path.join('workdir', run_folder)) if os.path.isdir(os.path.join('workdir', run_folder, name))]
        num = len(sim_folders)
    """
    # set file path blend.txt, tplot_disp.txt 
    # workdir = create_output_dir(run_folder, num)
    mesh_flname = 'mesh.txt'
    mesh_path = os.path.join(workdir, mesh_flname)

    plot_flname = 'plot.txt'
    plot_path = os.path.join(workdir, plot_flname)

    beam_length = 18

    # set material number for blocks
    mat_num = [1, 1, 1, 2]
    # set block height
    coord_z_inf = [0, 0, 0, beam_length]    # lowest z-coordinate per block
    coord_z_sup = [beam_length, beam_length, beam_length, beam_length + 0.5] # highest z-coordinate per block

    # initialize x and y arrays
    # save_search_space(run_folder, workdir, num, MODE)
    coord_x_inf, coord_x_sup, coord_y_inf, coord_y_sup = get_sample_dim(workdir)

    delta_y = (coord_y_sup[1] - coord_y_inf[1]) / 2 # half of thickness of center block


    with open(mesh_path, 'w') as file_mesh:
        # helper arrays
        delta_fac = [-1, 0, 1, 0]
        h_fac = [-1, -0.5, 0, -.5]
        # x, y, in M will be the front face. z from inside to outside
        for i in range(4):
            if coord_y_sup[i] != 0:
                num_node = i * 8
                h = coord_y_sup[i] - coord_y_inf[i]
                # compute global minimum y-coordinate of block
                global_y_sup = delta_fac[i] * delta_y + h_fac[i] * (coord_y_sup[i] - coord_y_inf[i])
                material = mat_num[i]

                # define mesh
                content = f'''SNODe	! definition of "super-nodes" defining the edges of the structure
        {1 + num_node}	{coord_x_sup[i]}    {global_y_sup + h}   {coord_z_inf[i]}
        {2 + num_node} 	{coord_x_inf[i]}    {global_y_sup + h}   {coord_z_inf[i]}
        {3 + num_node} 	{coord_x_inf[i]} 	{global_y_sup}       {coord_z_inf[i]}
        {4 + num_node} 	{coord_x_sup[i]}    {global_y_sup}       {coord_z_inf[i]}
        {5 + num_node} 	{coord_x_sup[i]}    {global_y_sup + h}   {coord_z_sup[i]}
        {6 + num_node} 	{coord_x_inf[i]}	{global_y_sup + h}   {coord_z_sup[i]}
        {7 + num_node} 	{coord_x_inf[i]} 	{global_y_sup}       {coord_z_sup[i]}
        {8 + num_node} 	{coord_x_sup[i]}    {global_y_sup}       {coord_z_sup[i]}

    BLENd ! definition of mesh via blend command
        SOLId,{(coord_x_sup[i] - coord_x_inf[i]) / 0.5},{(coord_y_sup[i] - coord_y_inf[i]) / 0.5}, {(coord_z_sup[i] - coord_z_inf[i]) / 0.5}	! solid (3d) mesh, with a elements in 1-direction and b elements in 2- and 3-direction
        BRICk,8		! 8-node brick elements
        MATE,{material}		! constitutive law definition
        {1 + num_node} {2 + num_node} {3 + num_node} {4 + num_node} {5 + num_node} {6 + num_node} {7 + num_node} {8 + num_node} ! definition of nodes that make up the block

    '''
                file_mesh.write(content)

        # calculation of load scaling factor, necessary for convergence of simulation, dependent on surface are in x-y-plane
        fac2 = (81-0.2)/80
        tot_a = np.sum([(coord_x_sup[i]-coord_x_inf[i])*(coord_y_sup[i]-coord_y_inf[i]) for i in range(3)])
        fac = tot_a * (1-fac2) + fac2

        # set boundary condition and loads
        content = f'''
    EBOUndary ! edge boundary conditions 
        3 0 1 1 1 ! in direction 3, at coordinate 0, hold all 3 displacement DOFS

    CBOUndary ! nodal boundary conditions 
        node  2.5    2.5    {coord_z_sup[3]}  1   1   1	

    CDISPlacement
        node  2.5    2.5    {coord_z_sup[3]}  {np.around(beam_length*0.0015*fac,6)*2}   {np.around(beam_length*0.0015*fac,6)*2}   {np.around(beam_length*0.0015*fac,6)*0.2}	
    '''
        file_mesh.write(content)

    # write tplot displacement command (record displacement result each time point) to tplot_disp.txt
    with open(plot_path, 'w') as file_plot:
        # define "plot" output (txt files)
        content = f'''
    BATCh
        TPLOt ! output commands for forces/stresses/displacements...
    END
    DISP,,1  2.5	2.5	{coord_z_sup[3]} ! displacement in 1-direction of point with coordinate (a,0,0)
    DISP,,2  2.5	2.5	{coord_z_sup[3]} ! displacement in 2-direction of point with coordinate (a,0,0)
    DISP,,3  2.5  2.5	{coord_z_sup[3]} ! displacement in 3-direction of point with coordinate (a,0,0)
    SUMS,1,3,0,0.0001   ! sum of reaction forces in 1-direction for coordinate direction 3 with value 0
    SUMS,2,3,0,0.0001   ! sum of reaction forces in 2-direction for coordinate direction 3 with value 0
    SUMS,3,3,0,0.0001   ! sum of reaction forces in 3-direction for coordinate direction 3 with value 0
    '''
        file_plot.write(content)

    # run simulation
    copy_sim_files(workdir)
    os.chdir(workdir)
    os.system('Feap86.exe -iiCantilever.txt')
    del_sim_files()

if __name__ == "__main__":
    run_sim(workdir=args.folder)
