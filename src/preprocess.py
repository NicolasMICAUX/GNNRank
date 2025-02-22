import os
import pickle as pk
import random
from typing import List, Tuple

import numpy as np
import numpy.random as rnd
import scipy.sparse as sp

from extract_network import extract_network
from generate_data import to_dataset_no_split

from utils import ERO


def load_data_from_memory(root) -> List:
    """If a dataset as already been packaged as a pickle file, load it from memory.
    :param root: The path to the pickle file.
    :return: The data object.
    """
    data = pk.load(open(root, 'rb'))
    if not os.path.isdir(root):
        try:
            os.makedirs(root)
        except FileExistsError:
            pass
    return [data]


def load_real_data(dataset: str) -> Tuple:
    """Load a real-world (not synthetic) dataset.
    This dataset is an adjacency matrix, stored in a npz file.
    :param dataset: The name of the dataset.
    :return: The adjacency matrix, the node features, and the labels.
    """
    x, y = None, None
    A = sp.load_npz(os.path.join(os.path.dirname(os.path.realpath(
        __file__)), '../data/' + dataset + 'adj.npz'))
    # if os.path.isfile(x_path := os.path.join(os.path.dirname(os.path.realpath(
    #     __file__)), '../data/' + dataset + 'x.npy')):
    #     x = np.load(x_path)
    if os.path.isfile(y_path := os.path.join(os.path.dirname(os.path.realpath(
        __file__)), '../data/' + dataset + 'y.npy')):
        y = np.load(y_path)
    return A, x, y


def load_data(args, random_seed):
    """
    Load data for training, be it synthetic or real-world data.
    :param args: all arguments describing the dataset to load or generate.
    :param random_seed: the random seed to use for data generation.
    :return: labels, train_mask, val_mask, test_mask, node features, Adjacency matrix
    """
    rnd.seed(random_seed)
    random.seed(random_seed)
    label = None
    train_mask = None
    val_mask = None
    test_mask = None
    default_name_base = 'trials' + str(args.num_trials) + 'train_r' + str(int(100 * args.train_ratio)) + 'test_r' + str(
        int(100 * args.test_ratio))
    if args.dataset[:3] == 'ERO':
        default_name_base += 'seed' + str(random_seed)
    save_path = os.path.join(os.path.dirname(os.path.realpath(
        __file__)), '../data/' + args.dataset + default_name_base + '.pk')
    if (not args.regenerate_data) and os.path.exists(save_path):
        print('Loading existing data!')
        data = load_data_from_memory(save_path)[0]
    else:
        print('Generating new data or new data splits!')
        if args.dataset[:3] == 'ERO':
            A, label = ERO(n=args.N, p=args.p, eta=args.eta, style=args.ERO_style)  # label=1D array of size nb_nodes
            A, label = extract_network(A, label)  # A and label sizes unchanged
            data = to_dataset_no_split(A, args.K, label, save_path=save_path,
                                       load_only=args.load_only)
        else:
            A, features, label = load_real_data(args.dataset)
            data = to_dataset_no_split(A, args.K, label, save_path=save_path,
                                       load_only=args.load_only, features=features)
    if hasattr(data, 'train_mask'):
        train_mask = data.train_mask.data.numpy().astype('bool_')
        val_mask = data.val_mask.data.numpy().astype('bool_')
        test_mask = data.test_mask.data.numpy().astype('bool_')

    return label, train_mask, val_mask, test_mask, data.x, data.A
