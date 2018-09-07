from amc_utils import *
import numpy as np

def amc(O, O_inv, mu_true, thresh=0.2, nonzeros=3):
    dim = np.shape(O)[0]
    iterative_deps_mask = samplegrid(dim,dim,nonzeros)
    unexpected = dim**2
    prev_unexpected = dim**2
    unexpected_nonzero_counts = []
    try:
        C_synth = O - np.outer(mu_true, mu_true)
        J_synth = np.linalg.pinv(C_synth)
    except:
        print("Failed to invert J in guess and check before loop")
        return np.zeros(np.shape(mu_true)), iterative_deps_mask
    J_distances = []
    mu_distances = []
    max_vals = []
    num_iters = 0
    while(True):
        #print("iter: ", num_iters)
        num_iters = num_iters + 1
        starttime = time.time()
        mu = solveMatrixCompletionWithMu(O_inv,O,iterative_deps_mask)
        max_val, max_ind, J_clean = find_largest(O,mu,dim,iterative_deps_mask,thresh)
        #print("MAX VAL: ", max_val)
        #print("max_ind: ", max_ind)
        max_vals.append(max_val)
        #if max_val < 1e-6: return J_distances, mu_distances, max_vals, mu, num_iters
        if max_val < 1e-6: 
            #print("iterative_deps_mask: ", iterative_deps_mask)
            return mu, iterative_deps_mask

        J_dist = np.linalg.norm(J_synth-J_clean,2)/(1.0*dim**2)
        J_distances.append(J_dist)
        mu_dist = np.linalg.norm(mu-mu_true,2)/(1.0*dim)
        mu_distances.append(mu_dist)
        iterative_deps_mask.append(max_ind)
        #print("MAX IND: ", max_ind)
    #return J_distances, mu_distances, max_vals, mu, num_iters
    return mu, iterative_deps_mask

def amc_analysis(O, O_inv, mu_true, thresh=0.2, nonzeros=3):
    dim = np.shape(O)[0]
    iterative_deps_mask = samplegrid(dim,dim,nonzeros)
    unexpected = dim**2
    prev_unexpected = dim**2
    unexpected_nonzero_counts = []
    try:
        C_synth = O - np.outer(mu_true, mu_true)
        J_synth = np.linalg.pinv(C_synth)
    except:
        print("Failed to invert J in guess and check before loop")
        return np.zeros(np.shape(mu_true)), iterative_deps_mask
    J_distances = []
    mu_distances = []
    max_vals = []
    second_largest_vals = []
    num_iters = 0
    while(True):
        #print("iter: ", num_iters)
        num_iters = num_iters + 1
        starttime = time.time()
        mu = solveMatrixCompletionWithMu(O_inv,O,iterative_deps_mask)
        largest, second_largest, largest_val, second_largest_val, J_clean = find_largest_analysis(O,mu,dim,iterative_deps_mask,thresh)
        #print("largest_val: ", largest_val)
        #print("2nd largest val: ", second_largest_val)
        max_vals.append(largest_val)
        second_largest_vals.append(second_largest_val)

        J_dist = np.linalg.norm(J_synth-J_clean,2)/(1.0*dim**2)
        J_distances.append(J_dist)
        mu_dist = np.linalg.norm(mu-mu_true,2)/(1.0*dim)
        mu_distances.append(mu_dist)
        
        #if max_val < 1e-6: return J_distances, mu_distances, max_vals, mu, num_iters
        if largest_val < 1e-6: 
            #print("iterative_deps_mask: ", iterative_deps_mask)
            return mu, iterative_deps_mask, max_vals, second_largest_vals, mu_distances

        iterative_deps_mask.append(largest)
        #print("MAX IND: ", max_ind)
    #return J_distances, mu_distances, max_vals, mu, num_iters
    return mu, iterative_deps_mask, max_vals, second_largest_vals, mu_distances
