import numpy as np
import cvxpy as cp
import copy
import time
from math import exp
from scipy.stats import kendalltau
import collections

def compare_deps_old(deps_hat, deps_true):
    # expects two lists of tuples
    diff = len(set(deps_true) - set(deps_hat))
    return 1. - (diff/float(len(set(deps_true))))

def compare_deps(deps_hat, deps_true):
    """ Calculates the F1 Score between dep sets """
    deps_true = set(deps_true)
    deps_hat = set(deps_hat)

    tp = len(deps_true & deps_hat)
    fp = len(deps_hat) - tp 
    fn = len(deps_true) - tp

    precision=float(tp)/(tp+fp)
    recall=float(tp)/(tp+fn)
    return 2*((precision*recall)/(precision+recall))

def compare_ranking(mu_true,mu_hat):
    rank_true = np.argsort(mu_true)
    rank_hat = np.argsort(mu_hat)
    tau,pval =  kendalltau(rank_true,rank_hat)
    return tau

def get_deps_from_inverse_sig(J, thresh=0.2):
    deps = []
    for i in range(J.shape[0]):
        for j in range(J.shape[1]):
            if abs(J[i,j]) > thresh:
                deps.append((i,j))
    return deps

def gen_accs_high_acc(m, mu_normal=0.6):
    mu_high = 1 - 1/float(m-1)*5*(1-mu_normal)
    return mu_high

def gen_accs_high_acc_ratio(m, a, b, mu_normal=0.6):
    mu_high = 1 - float((b-a)*(1-mu_normal))/float(a*(m-1))
    return mu_high

def return_acc_vector(m, mu_normal, mu_high):
    mu = [mu_normal for _ in range(m)]
    mu[-1] = mu_high
    return mu

def find_largest(O,mu,dim,mask,thresh):
    prod = np.outer(mu, mu)
    #print("PROD SHAPE: ", prod.shape)
    #print("O SHAPE: ", O.shape)
    C = O - prod
    #print("C SHAPE: ", C.shape)
    try:
        J = np.linalg.pinv(C)
    except:
        print("Failed to invert K in find largest")
        #print("C: ", C)
        #print("O: ", O)
        #print("mu mu^T: ", prod)
        J = np.zeros((dim,dim),dtype=float)
        return 0, (-1,-1), J

    max_val = 0
    max_ind = (-1,-1)
    #print("J SHAPE")
    #print(J.shape)
    J_clean = copy.deepcopy(J)
    for i in range(dim):
        for j in range(dim):
            if abs(J[i,j]) <= thresh:
                J_clean[i,j] = 0
            if (i,j) not in mask and abs(J_clean[i,j]) > max_val:
                max_val = abs(J_clean[i,j])
                max_ind = (i,j)
    return max_val, max_ind, J_clean

def find_largest_analysis(O,mu,dim,mask,thresh):
    prod = np.outer(mu, mu)
    #print("PROD SHAPE: ", prod.shape)
    #print("O SHAPE: ", O.shape)
    C = O - prod
    #print("C SHAPE: ", C.shape)
    try:
        J = np.linalg.pinv(C)
    except:
        print("Failed to invert K in find largest")
        #print("C: ", C)
        #print("O: ", O)
        #print("mu mu^T: ", prod)
        J = np.zeros((dim,dim),dtype=float)
        return 0, (-1,-1), J

    #print("J SHAPE")
    #print(J.shape)
    J_h = copy.deepcopy(J)
    for i in range(dim):
        for j in range(dim):
            if abs(J[i,j]) <= thresh or (i,j) in mask:
                J_h[i,j] = 0
    J_h = np.absolute(J_h)
    #print("J_h: ", J_h)

    J_clean = copy.deepcopy(J)
    for i in range(dim):
        for j in range(dim):
            if abs(J[i,j]) <= thresh:
                J_clean[i,j] = 0
    #print("J_clean: ", J_clean)

    #sorted_indices = J_clean.argsort()
    sorted_indices = np.dstack(np.unravel_index(np.argsort(J_h.ravel()), J_h.shape))
    sorted_indices = sorted_indices[0].tolist()
    #import ipdb; ipdb.set_trace()
    largest = tuple(sorted_indices[-1])
    second_largest = tuple(sorted_indices[-2])
    assert(J_h[largest[0],largest[1]] == J_h.max())
    assert(J_h[largest[0],largest[1]]  >= J_h[second_largest[0],second_largest[1]])
    
    #ipdb.set_trace()
    # change this logic
    # argmax and not in mask

    return largest, second_largest, J_h[largest[0],largest[1]], J_h[second_largest[0],second_largest[1]], J_clean

def solveMatrixCompletion(O_inv, deps):
    #print("deps: ", deps)
    try: 
        set(deps)
    except:
        assert(0==1,"NOT HASHABLE")


    zeros_set = []
    for i in range(O_inv.shape[0]):
        for j in range(O_inv.shape[1]):
            zeros_set.append((i,j))
    zeros_set = set(zeros_set)
    zeros_set = zeros_set - set(deps)
    zeros_set = list(zeros_set)
    
    #form q
    q = np.zeros((len(zeros_set),),dtype=float)
    M = np.zeros((len(zeros_set),O_inv.shape[0]),dtype=float)
    for ix, z in enumerate(zeros_set):
        M[ix, z[0]] = 1
        M[ix, z[1]] = 1
    #print(M)
    #print(M.shape)
    for ix, z in enumerate(zeros_set):
        #print(z)
        q[ix] = np.log(O_inv[z[0],z[1]]**2)
    #print(len(zeros_set))
        
    l = np.linalg.pinv(M) @ q
    #print(M@l)
    #print(q)
    return l

def calculate_empirical_mu(z,O):
    c = 1 + z.dot(O.dot(z.T)) # check this
    mu = 1/np.sqrt(c)*O.dot(z.T)
    return mu

def solveMatrixCompletionWithMu(O_inv, O, deps):
    l = solveMatrixCompletion(O_inv,deps)
    l = np.exp(l)
    z_rec = np.sqrt(l)
    mu = calculate_empirical_mu(z_rec,O)
    return mu

def samplegrid(w, h, n):
    sampled = []
    for i in range(w):
        sampled.append((i,i))
    return sampled
