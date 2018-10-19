import argparse
import os
import pickle as pkl

import numpy as np

from snorkel import SnorkelSession
from snorkel.annotations import load_gold_labels
from snorkel.learning import GenerativeModel
from snorkel.learning.pytorch import LSTM
from snorkel.learning.structure import DependencySelector
from snorkel.models import candidate_subclass


parser = argparse.ArgumentParser(description='Train an LSTM model on noisy labels with varying parameters.')
parser.add_argument('--lfs_indices', type=str, default='1', help='indices of labelling functions to use in generative'
                                                                 'model.')
parser.add_argument('--hidden_dim', type=int, default=50, help='dimension of the hidden dimension in the end model.')
parser.add_argument('--datapath', type=str, default='./data')
parser.add_argument('--save_file', type=str, default='cdr_d.log')

args = parser.parse_args()

train_kwargs = {
    'lr':              0.01,
    'embedding_dim':   None,
    'hidden_dim':      None,
    'n_epochs':        20,
    'dropout':         0.5,
    'rebalance':       0.25,
    'print_freq':      5,
    'seed':            1701
}


def load_label_matrix(datapath, label_indices, split):
    file_path = os.path.join(datapath, '{}_label_matrix'.format(split))
    L_mat = pkl.load(open(file_path, 'rb'))
    # mask labelling functions
    return L_mat[:, label_indices]


def load_features(session):
    ChemicalDisease = candidate_subclass('ChemicalDisease', ['chemical', 'disease'])
    train_cands = session.query(ChemicalDisease).filter(ChemicalDisease.split == 0).all()
    dev_cands = session.query(ChemicalDisease).filter(ChemicalDisease.split == 1).all()
    test_cands = session.query(ChemicalDisease).filter(ChemicalDisease.split == 2).all()
    return train_cands, dev_cands, test_cands


def log_odd(alpha):
    return np.log(alpha / (1. - alpha))


def get_metrics(error_metrics, eps=10e-8):
    tp, fp, tn, fn = len(error_metrics[0]), len(error_metrics[1]), len(error_metrics[2]), len(error_metrics[3])
    acc = float(tp + tn) / float(tp + fp + tn + fn + eps)
    prec = float(tp) / (tp + fp + eps)
    rec = float(tp) / (tp + fn + eps)
    f1 = 2 * prec * rec / (prec + rec + eps)
    return round(f1, 4), round(acc, 4)


def log_result(log_path, log_message):
    if not os.path.exists(log_path):
        log_file = open(log_path, 'w')
    else:
        log_file = open(log_path, 'a')
    log_file.write(';'.join([str(x) for x in log_message]) + '\n')
    log_file.close()


if __name__ == '__main__':
    args = parser.parse_args()
    session = SnorkelSession()
    # get data
    train_cands, dev_cands, test_cands = load_features(session)
    L_gold_dev = load_gold_labels(session, annotator_name='gold', split=1)
    L_gold_test = load_gold_labels(session, annotator_name='gold', split=2)
    lfs_indices = [int(x) for x in args.lfs_indices.split(',')]
    L_train = load_label_matrix(args.datapath, lfs_indices, split='train')
    L_test = load_label_matrix(args.datapath, lfs_indices, split='test')
    # train generative model
    ds = DependencySelector()
    deps = ds.select(L_train, threshold=0.1)
    gen_model = GenerativeModel(lf_propensity=True)
    gen_model.train(L_train, deps=deps, decay=0.95, step_size=0.1/L_train.shape[0], reg_param=0.0)
    gen_model.train(L_train, epochs=100, decay=0.95, step_size=0.1 / L_train.shape[0], reg_param=1e-6)
    train_marginals = gen_model.marginals(L_train)
    # train end model
    train_kwargs['embedding_dim'] = args.hidden_dim
    train_kwargs['hidden_dim'] = args.hidden_dim
    lstm = LSTM(n_threads=None)
    lstm.train(train_cands, train_marginals, X_dev=dev_cands, Y_dev=L_gold_dev, **train_kwargs)
    # compute metrics
    stats = gen_model.learned_lf_stats()
    lfs_accuracies = stats['Accuracy']
    alpha_min = np.min(lfs_accuracies)
    alpha_max = np.max(lfs_accuracies)
    r_L = round(log_odd(alpha_min) / log_odd(alpha_max), 4)
    gm_errors = gen_model.error_analysis(session, L_test, L_gold_test)
    dm_errors = lstm.error_analysis(session, test_cands, L_gold_test)
    gm_f1, gm_acc = get_metrics(gm_errors)
    dm_f1, dm_acc = get_metrics(dm_errors)
    # log metrics [l, d, r_L, alpha_L, gm_accuracy, dm_accuracy]
    log_message = [len(lfs_indices), args.hidden_dim, r_L, round(alpha_max, 4)]
    log_message += [gm_f1, gm_acc, dm_f1, dm_acc]
    log_result(os.path.join(args.datapath, args.save_file), log_message)
