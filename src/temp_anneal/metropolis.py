import numpy as np
import tqdm


def metropolis(p, qdraw, xinit, nsamp=10000):
    samples=np.empty((nsamp,2))
    x_prev = xinit
    accepted = 0
    for i in tqdm.tqdm(range(nsamp), position=0):
        x_star = qdraw(x_prev)
        p_star = p(x_star)
        p_prev = p(x_prev)
        pdfratio = p_star / p_prev
        if np.random.uniform() < min(1, pdfratio):
            samples[i] = x_star
            x_prev = x_star
            accepted += 1
        else:
            samples[i]= x_prev
            
    return samples, accepted
