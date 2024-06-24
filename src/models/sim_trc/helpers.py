from scipy.stats import qmc
import numpy as np
import os


def random_sampling(samples, bounds):
    sample_space = []
    l_bounds = [feat['domain'][0] for feat in bounds]
    u_bounds = [feat['domain'][1] for feat in bounds]
    # For inclusion of upper limit
    u_bounds = [item + 1 for item in u_bounds]
    for i in range(samples):
        coord_x_inf, coord_x_sup, coord_y_inf, coord_y_sup = initilize_boundaries(u_bounds)
        # randomly sample dimensions for all three blocks
        for i in range(3):
            x_dim = np.random.randint(l_bounds[i], u_bounds[i])
            y_dim = np.random.randint(l_bounds[i+3], u_bounds[i+3])
            coord_x_inf[i] = - x_dim / 2
            coord_x_sup[i] = x_dim / 2
            coord_y_inf[i] = -y_dim / 2
            coord_y_sup[i] = y_dim / 2
        sample_space.append([coord_x_inf, coord_x_sup, coord_y_inf, coord_y_sup])
    return np.stack(sample_space)


def lhc_sampling(samples, bounds):
    # lhc sample dimensions for all features
    l_bounds = [feat['domain'][0] for feat in bounds]
    u_bounds = [feat['domain'][1] for feat in bounds]
    # For inclusion of upper limit
    u_bounds = [item + 1 for item in u_bounds]
    sampler = qmc.LatinHypercube(d=len(l_bounds))
    sample = sampler.random(n=samples)
    sample_scaled = qmc.scale(sample, l_bounds, u_bounds)
    # SIM TRC Specific Code
    sample_scaled = sample_scaled.astype(int)
    print(sample_scaled)
    # dimensions for all three blocks
    sample_space = []
    for sample in range(samples):
        coord_x_inf, coord_x_sup, coord_y_inf, coord_y_sup = initilize_boundaries(u_bounds)
        for i in range(3):
            x_dim = sample_scaled[sample][i]
            y_dim = sample_scaled[sample][i + 3]
            coord_x_inf[i] = - x_dim / 2
            coord_x_sup[i] = x_dim / 2
            coord_y_inf[i] = -y_dim / 2
            coord_y_sup[i] = y_dim / 2
        sample_space.append([coord_x_inf, coord_x_sup, coord_y_inf, coord_y_sup])
    return np.stack(sample_space)


def constant_sampling(list_, bounds):
    sample_space = []
    u_bounds = [feat['domain'][1] for feat in bounds]
    for sample in range(len(list_)):
        coord_x_inf, coord_x_sup, coord_y_inf, coord_y_sup = initilize_boundaries(u_bounds)
        for i in range(3):
            x_dim = list_[sample][i]
            y_dim = list_[sample][i+3]
            coord_x_inf[i] = - x_dim / 2
            coord_x_sup[i] = x_dim / 2
            coord_y_inf[i] = -y_dim / 2
            coord_y_sup[i] = y_dim / 2
        sample_space.append([coord_x_inf, coord_x_sup, coord_y_inf, coord_y_sup])
    return np.stack(sample_space)


def define_input(x_input, bounds):
    u_bounds = [feat['domain'][1] for feat in bounds]
    coord_x_inf, coord_x_sup, coord_y_inf, coord_y_sup = initilize_boundaries(u_bounds)
    # set dimensions for all three blocks
    # sample_space = []
    for i in range(3):
        x_dim = x_input[i]
        y_dim = x_input[i + 3]
        coord_x_inf[i] = - x_dim / 2
        coord_x_sup[i] = x_dim / 2
        coord_y_inf[i] = -y_dim / 2
        coord_y_sup[i] = y_dim / 2
    # print(np.stack([coord_x_inf, coord_x_sup, coord_y_inf, coord_y_sup]))
    return np.stack([coord_x_inf, coord_x_sup, coord_y_inf, coord_y_sup])


def initilize_boundaries(u_bounds):
    x_boundaries = u_bounds[0:3]
    y_boundaries = u_bounds[3:6]
    # initialize x and y arrays
    coord_x_inf = np.zeros(4)
    coord_x_sup = np.zeros(4)
    coord_y_inf = np.zeros(4)
    coord_y_sup = np.zeros(4)

    # define top plate (spans maximum possible area)
    coord_x_inf[-1] = - np.max(x_boundaries) / 2
    coord_x_sup[-1] = np.max(x_boundaries) / 2
    coord_y_inf[-1] = - sum(y_boundaries) / 2
    coord_y_sup[-1] = sum(y_boundaries) / 2

    return coord_x_inf, coord_x_sup, coord_y_inf, coord_y_sup
