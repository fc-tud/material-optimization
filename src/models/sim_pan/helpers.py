from scipy.stats import qmc
import numpy as np

# set length boundaries for Features
# l_bounds = np.array([0.3/0.7, 0.7/2, 0.5, 12/18, 0, 16/28, 0.41/1.32])
# u_bounds = np.array([1,           2,   1,     1, 1,     1, 1])


def random_sampling(samples, bounds):
    sample_space = []
    for feat in bounds:
        bound = np.random.uniform(feat['domain'][0], feat['domain'][1], samples)
        sample_space.append(bound)
    return np.stack(sample_space, axis=1)


def lhc_sampling(samples, bounds):
    # lhc sample dimensions for all features
    l_bounds = [feat['domain'][0] for feat in bounds]
    u_bounds = [feat['domain'][1] for feat in bounds]
    sampler = qmc.LatinHypercube(d=len(l_bounds))
    sample = sampler.random(n=samples)
    sample_scaled = qmc.scale(sample, l_bounds, u_bounds)
    # sample_scaled = sample_scaled.astype(int)
    return sample_scaled
    

def define_input(x_input):
    return x_input
 
