import subprocess
import time

import numpy as np

SLEEP = 10
CTR_LIM = 12
procs_lim = 1
l_vals = range(33)
d_vals = [100]  # np.arange(10, 250, 10)
script = 'train_model.py'
datapath = 'cdr_data/'
n_runs = 3  # (for labelling functions)
save_file = 'cdr_training_{}.log'


def num_procs_open(procs):
    k = 0
    for p in procs:
        k += (p.poll() is None)
    return k


if __name__ == '__main__':
    procs = []
    j = 0
    # loop over d
    for d in d_vals:
        j += 1
        proc_args = [
            'python', script,
            '--hidden_dim', str(d),
            '--lfs_indices', ','.join([str(x) for x in l_vals]),
            '--datapath', datapath,
            '--save_file', save_file.format('d')
        ]
        print("Launching model {0}".format(j))
        p = subprocess.Popen(proc_args)
        procs.append(p)
        ctr = 0
        while True:
            k = num_procs_open(procs)
            ctr += 1
            if ctr >= CTR_LIM:
                ctr = 0
                print('{0} processes still running'.format(k))
            if num_procs_open(procs) >= procs_lim:
                time.sleep(SLEEP)
            else:
                break
    # loop over l
    for l in l_vals:
        break
        for _ in range(n_runs):
            l_indices = np.random.choice(l_vals, l+1, replace=False)
            j += 1
            proc_args = [
                'python', script,
                '--hidden_dim', str(100),
                '--lfs_indices', ','.join([str(x) for x in l_indices]),
                '--datapath', datapath,
                '--save_file', save_file.format('l')]
            print("Launching model {0}".format(j))
            p = subprocess.Popen(proc_args)
            procs.append(p)
            ctr = 0
            while True:
                k = num_procs_open(procs)
                ctr += 1
                if ctr >= CTR_LIM:
                    ctr = 0
                    print('{0} processes still running'.format(k))
                if num_procs_open(procs) >= procs_lim:
                    time.sleep(SLEEP)
                else:
                    break

    n = len(procs)
    ctr = 0
    while True:
        k = num_procs_open(procs)
        if k == 0:
            break
        ctr += 1
        if ctr >= CTR_LIM:
            ctr = 0
            print('{0} processes still running'.format(k))
        time.sleep(SLEEP)
