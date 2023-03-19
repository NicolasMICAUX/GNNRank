import os
import pickle as pk

import scipy.sparse as sp
from torch_geometric.data import Data

from utils import hermitian_feature


def to_dataset_no_split(A, num_clusters, label, save_path, load_only=False, features=None) -> Data:
    """
    Convert a graph (adjacency matrix + features) to a torch_geometric.data.Data object.
    :param A: adjacency matrix
    :param num_clusters:
    :param label: "true" ranking
    :param save_path: path to save the data object
    :param load_only: if not load_only, save the data object to save_path fo future use
    :param features: node features
    :return: torch_geometric.data.Data object
    """
    if features is None:
        features = hermitian_feature(A, num_clusters)

    data = Data(x=features, y=label, A=sp.csr_matrix(A))
    if not load_only:
        if not os.path.isdir(os.path.dirname(save_path)):
            try:
                os.makedirs(os.path.dirname(save_path))
            except FileExistsError:
                print('Folder exists for best {}!'.format(os.path.dirname(save_path)))
        pk.dump(data, open(save_path, 'wb'))
    return data
