import scipy.io
import os
import scipy
import torch
import scipy.io
from torch_geometric.data import Data
import numpy as np
from torch_geometric.datasets import Yelp, CoraFull
from utils import sparse_mx_to_torch_sparse_tensor


# blogcatalog
def load_mat_data(shuffle_idx):

    data_name = "blogcatalog"
    path = "../data/blogcatalog/"
    print('Loading dataset ' + data_name + '.mat...')
    mat = scipy.io.loadmat(path + data_name)

    # labels = mat['group']
    # labels = sparse_mx_to_torch_sparse_tensor(labels).to_dense()
    labels = torch.load(path+"blog_clean_lbl.pt")
    print(labels.shape)

    adj = mat['network']
    adj = sparse_mx_to_torch_sparse_tensor(adj).to_dense()
    edge_index = torch.transpose(torch.nonzero(adj), 0, 1).long()
    features = torch.eye(labels.shape[0]).float()

    # groups assignment
    #groups = pd.read_pickle(r'../data/groups.pkl')
    class_order = torch.load(path + "blog_" + shuffle_idx +".pt")
    groups = torch.load(path + "groups" + shuffle_idx[-1] + ".pt")
    # splits is a nested dictionary
    # key: groups, value: (key:train, val, test; value:node_ids)
    #splits = pd.read_pickle(r'../data/groups_splits.pkl')
    splits = torch.load(path + "split_" + shuffle_idx[-1] + ".pt")

    num_nodes = labels.shape[0]

    G = Data(x=features,
             edge_index=edge_index,
             y=labels)
    G.n_id = torch.arange(num_nodes)
    G.splits = splits
    G.groups = groups
    G.class_order = class_order

    return G


# Yelp
def import_yelp(shuffle_idx):

    data_name = "yelp"
    path = "../data/yelp/"
    print('Loading dataset ' + data_name + '...')
    dataset = Yelp(root='../../tmp/Yelp')

    data = dataset[0]
    # removed small classes if there is
    labels = torch.load(path + "yelp_clean_lbl.pt")
    features = data.x
    edge_index = data.edge_index

    class_order = torch.load(path + "yelp_" + shuffle_idx + ".pt")
    groups = torch.load(path + "groups" + shuffle_idx[-1] + ".pt")
    # splits is a nested dictionary
    # key: groups, value: (key:train, val, test; value:node_ids)
    # splits = pd.read_pickle(r'../data/groups_splits.pkl')
    splits = torch.load(path + "split_" + shuffle_idx[-1] + ".pt")

    num_nodes = labels.shape[0]

    G = Data(x=features,
             edge_index=edge_index,
             y=labels)
    G.n_id = torch.arange(num_nodes)
    G.splits = splits
    G.groups = groups
    G.class_order = class_order

    return G


def load_hyper_data(shuffle_idx):

    data_name = "Hyperspheres_10_10_0"
    path = "../data/"
    print('Loading dataset ' + data_name + '...')
    labels = np.genfromtxt(os.path.join(path, data_name, "labels.csv"),
                           skip_header=1, dtype=np.dtype(float), delimiter=',')
    labels = torch.tensor(labels).float()

    features = torch.tensor(np.genfromtxt(os.path.join(path, data_name, "features.csv"),
                                          skip_header=1, dtype=np.dtype(float), delimiter=',')).float()

    edges = torch.tensor(np.genfromtxt(os.path.join(path, data_name, "edges.txt"),
                                       dtype=np.dtype(float), delimiter=',')).long()
    edge_index = torch.transpose(edges, 0, 1)

    class_order = torch.load(path + data_name + shuffle_idx + ".pt")
    groups = torch.load(path + "groups" + shuffle_idx[-1] + ".pt")
    # splits is a nested dictionary
    # key: groups, value: (key:train, val, test; value:node_ids)
    # splits = pd.read_pickle(r'../data/groups_splits.pkl')
    splits = torch.load(path + "split_" + shuffle_idx[-1] + ".pt")

    num_nodes = labels.shape[0]

    G = Data(x=features,
             edge_index=edge_index,
             y=labels)
    G.n_id = torch.arange(num_nodes)
    G.splits = splits
    G.groups = groups
    G.class_order = class_order

    return G


def import_dblp(shuffle_idx):
    path = "../data/dblp/"
    data_name = "dblp"
    print('Loading dataset DBLP...')
    features = torch.FloatTensor(np.genfromtxt(os.path.join(path, "features.txt"), delimiter=",", dtype=np.float64))
    # no label size smaller than 50
    labels = torch.FloatTensor(np.genfromtxt(os.path.join(path, "labels.txt"), delimiter=","))
    edge_list = torch.tensor(np.genfromtxt(os.path.join(path, "dblp.edgelist"))).long()
    edge_list_other_half = torch.hstack((edge_list[:, 1].reshape(-1, 1), edge_list[:, 0].reshape(-1, 1)))
    edge_index = torch.transpose(edge_list, 0, 1)
    edge_index_other_half = torch.transpose(edge_list_other_half, 0, 1)
    edge_index = torch.hstack((edge_index, edge_index_other_half))

    class_order = torch.load(path + data_name + "_" + shuffle_idx + ".pt")
    groups = torch.load(path + "groups" + shuffle_idx[-1] + ".pt")
    # splits is a nested dictionary
    # key: groups, value: (key:train, val, test; value:node_ids)
    # splits = pd.read_pickle(r'../data/groups_splits.pkl')
    splits = torch.load(path + "split_" + shuffle_idx[-1] + ".pt")

    num_nodes = labels.shape[0]

    G = Data(x=features,
             edge_index=edge_index,
             y=labels)
    G.n_id = torch.arange(num_nodes)
    G.splits = splits
    G.groups = groups
    G.class_order = class_order

    return G


def import_pcg(shuffle_idx):
    path = "../data/"
    data_name = "pcg_removed_isolated_nodes"

    print('Loading dataset ' + data_name + '.csv...')

    labels = np.genfromtxt(os.path.join(path, data_name, "labels.csv"),
                           dtype=np.dtype(float), delimiter=',')
    labels = torch.tensor(labels).float()
    features = torch.tensor(np.genfromtxt(os.path.join(path, data_name, "features.csv"),
                                          dtype=np.dtype(float), delimiter=',')).float()

    edges = torch.tensor(np.genfromtxt(os.path.join(path, data_name, "edges_undir.csv"),
                                       dtype=np.dtype(float), delimiter=','))
    edge_index = torch.transpose(edges, 0, 1).long()

    class_order = torch.load(path + data_name + shuffle_idx + ".pt")
    groups = torch.load(path + "groups" + shuffle_idx[-1] + ".pt")
    # splits is a nested dictionary
    # key: groups, value: (key:train, val, test; value:node_ids)
    # splits = pd.read_pickle(r'../data/groups_splits.pkl')
    splits = torch.load(path + "split_" + shuffle_idx[-1] + ".pt")

    num_nodes = labels.shape[0]

    G = Data(x=features,
             edge_index=edge_index,
             y=labels)
    G.n_id = torch.arange(num_nodes)
    G.splits = splits
    G.groups = groups
    G.class_order = class_order

    return G


def import_humloc(shuffle_idx):
    path = "../data/"
    data_name = "HumanGo"
    print('Loading dataset ' + data_name + '.csv...')

    edge_list = torch.tensor(np.genfromtxt(os.path.join(path, data_name, "edge_list.csv"),
                                           skip_header=1, dtype=np.dtype(float), delimiter=','))[:, :2].long()
    edge_list_other_half = torch.hstack((edge_list[:, 1].reshape(-1, 1), edge_list[:, 0].reshape(-1, 1)))
    edge_index = torch.transpose(edge_list, 0, 1)
    edge_index_other_half = torch.transpose(edge_list_other_half, 0, 1)
    edge_index = torch.hstack((edge_index, edge_index_other_half))

    labels = np.genfromtxt(os.path.join(path, data_name, "labels.csv"),
                           dtype=np.dtype(float), delimiter=',')
    labels = torch.tensor(labels).float()

    features = np.genfromtxt(os.path.join(path, data_name, "features.csv"),
                             dtype=np.dtype(float), delimiter=',')

    features = torch.tensor(features).float()

    class_order = torch.load(path + data_name + shuffle_idx + ".pt")
    groups = torch.load(path + "groups" + shuffle_idx[-1] + ".pt")
    # splits is a nested dictionary
    # key: groups, value: (key:train, val, test; value:node_ids)
    # splits = pd.read_pickle(r'../data/groups_splits.pkl')
    splits = torch.load(path + "split_" + shuffle_idx[-1] + ".pt")

    num_nodes = labels.shape[0]

    G = Data(x=features,
             edge_index=edge_index,
             y=labels)
    G.n_id = torch.arange(num_nodes)
    G.splits = splits
    G.groups = groups
    G.class_order = class_order

    return G


def import_eukloc(shuffle_idx):
    path = "../data/"
    data_name = "EukaryoteGo"

    print('Loading dataset ' + data_name + '.csv...')

    edge_list = torch.tensor(np.genfromtxt(os.path.join(path, data_name, "edge_list.csv"),
                                           skip_header=1, dtype=np.dtype(float), delimiter=','))[:, :2].long()
    edge_list_other_half = torch.hstack((edge_list[:, 1].reshape(-1, 1), edge_list[:, 0].reshape(-1, 1)))
    edge_index = torch.transpose(edge_list, 0, 1)
    edge_index_other_half = torch.transpose(edge_list_other_half, 0, 1)
    edge_index = torch.hstack((edge_index, edge_index_other_half))

    labels = np.genfromtxt(os.path.join(path, data_name, "labels.csv"),
                           dtype=np.dtype(float), delimiter=',')
    labels = torch.tensor(labels).float()

    features = torch.tensor(np.genfromtxt(os.path.join(path, data_name, "features.csv"),
                                          dtype=np.dtype(float), delimiter=',')).float()

    class_order = torch.load(path + data_name + shuffle_idx + ".pt")
    groups = torch.load(path + "groups" + shuffle_idx[-1] + ".pt")
    # splits is a nested dictionary
    # key: groups, value: (key:train, val, test; value:node_ids)
    # splits = pd.read_pickle(r'../data/groups_splits.pkl')
    splits = torch.load(path + "split_" + shuffle_idx[-1] + ".pt")

    num_nodes = labels.shape[0]

    G = Data(x=features,
             edge_index=edge_index,
             y=labels)
    G.n_id = torch.arange(num_nodes)
    G.splits = splits
    G.groups = groups
    G.class_order = class_order

    return G


def import_cora(shuffle_idx):

    data_name = "CoraFull"
    path = "../data/CoraFull/"
    print('Loading dataset ' + data_name + '...')
    dataset = CoraFull(root='../../tmp/CoraFull')

    data = dataset[0]
    # removed small classes if there is
    labels = torch.load(path + "cora_clean_lbl.pt")
    features = data.x
    edge_index = data.edge_index

    class_order = torch.load(path + "cora_" + shuffle_idx + ".pt")
    groups = torch.load(path + "groups" + shuffle_idx[-1] + ".pt")
    # splits is a nested dictionary
    # key: groups, value: (key:train, val, test; value:node_ids)
    # splits = pd.read_pickle(r'../data/groups_splits.pkl')
    splits = torch.load(path + "split_" + shuffle_idx[-1] + ".pt")

    num_nodes = labels.shape[0]

    G = Data(x=features,
             edge_index=edge_index,
             y=labels)
    G.n_id = torch.arange(num_nodes)
    G.splits = splits
    G.groups = groups
    G.class_order = class_order

    return G






