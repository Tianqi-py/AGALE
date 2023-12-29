import scipy.io
import pandas as pd
import os
import scipy
import torch
import scipy.io
from torch_geometric.data import Data
import numpy as np
from torch_geometric.datasets import Yelp
from utils import sparse_mx_to_torch_sparse_tensor
from sklearn.utils import shuffle
from sklearn.model_selection import train_test_split
from collections import defaultdict
import itertools

"""
load the data for ClassIL setting
"""


def load_hyper_data(shuffle_idx):

    # n_cls_per_t: number of classes per time step
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
    print("number of edges:", edge_index.shape)

    # groups assignment
    n_cls = labels.shape[1]
    cls_order = shuffle(range(n_cls))

    groups_idx = [tuple(cls_order[i:i+n_cls_per_t]) for i in range(0, n_cls-1, n_cls_per_t)]
    #print("groups index", groups_idx)

    cls = torch.transpose(labels, 0, 1)
    cls_asgn = {}
    for i, label in enumerate(cls):
        cls_asgn[i] = torch.nonzero(label).flatten()

    groups = {}
    for g in groups_idx:
        groups[g] = []
    print(groups)
    for g in groups_idx:
        assert g in groups.keys()
        for c in g:
            groups[g].extend(cls_asgn[c].tolist())
            print(len(groups[g]))
        # note only once the nodes belong to multiple classes
        groups[g] = list(set(groups[g]))
        groups[g].sort()
        print("after set")
        print(len(groups[g]))
        print("###################")

    # splits is a nested dictionary
    # key: groups, value: (key:train, val, test; value:node_ids)
    # each time step, nodes have no more than classes seen so far

    # split the train-val-test within each class
    splits = {}
    for g in list(groups.keys()):
        split = {}
        # get all the nodes in this group
        node_ids = groups[g]

        # split the nodes into train-val-test: 60-10-30
        ids_train, ids_val_test = train_test_split(node_ids, test_size=0.4, random_state=42)
        ids_val, ids_test = train_test_split(ids_val_test, test_size=0.75, random_state=41)
        # write in the dictionary
        split["train"] = ids_train
        split["val"] = ids_val
        split["test"] = ids_test
        splits[g] = split

    print(splits)

    num_nodes = labels.shape[0]

    G = Data(x=features,
             edge_index=edge_index,
             y=labels)

    G.n_id = torch.arange(num_nodes)
    G.splits = splits
    G.groups = groups

    return G


def split_yelp(n_cls_per_t):
    data_name = "yelp"
    print('Loading dataset ' + data_name + '...')
    dataset = Yelp(root='../../tmp/Yelp')

    data = dataset[0]
    labels = data.y
    features = data.x
    edge_index = data.edge_index
    print("number of edges:", edge_index.shape)

    # groups assignment
    n_cls = labels.shape[1]
    cls_order = shuffle(range(n_cls))

    groups_idx = [tuple(cls_order[i:i+n_cls_per_t]) for i in range(0, n_cls-1, n_cls_per_t)]
    #print("groups index", groups_idx)

    cls = torch.transpose(labels, 0, 1)
    cls_asgn = {}
    for i, label in enumerate(cls):
        cls_asgn[i] = torch.nonzero(label).flatten()

    groups = {}
    for g in groups_idx:
        groups[g] = []
    for g in groups_idx:
        assert g in groups.keys()
        for c in g:
            groups[g].extend(cls_asgn[c].tolist())
            print(len(groups[g]))
        # note only once the nodes belong to multiple classes
        groups[g] = list(set(groups[g]))
        groups[g].sort()
        print("after set")
        print(len(groups[g]))
        print("###################")

    # splits is a nested dictionary
    # key: groups, value: (key:train, val, test; value:node_ids)
    # each time step, nodes have no more than classes seen so far

    # split the train-val-test within each class
    splits = {}
    for g in list(groups.keys()):
        split = {}
        # get all the nodes in this group
        node_ids = groups[g]

        # split the nodes into train-val-test: 60-10-30
        ids_train, ids_val_test = train_test_split(node_ids, test_size=0.4, random_state=42)
        ids_val, ids_test = train_test_split(ids_val_test, test_size=0.75, random_state=41)
        # write in the dictionary
        split["train"] = ids_train
        split["val"] = ids_val
        split["test"] = ids_test
        splits[g] = split

    #print(splits)

    num_nodes = labels.shape[0]

    G = Data(x=features,
             edge_index=edge_index,
             y=labels)

    G.n_id = torch.arange(num_nodes)
    G.splits = splits
    G.groups = groups

    return G


def split_blogcatalog(n_cls_per_t, ):
    data_name = "blogcatalog"
    path = "../data/"
    print('Loading dataset ' + data_name + '.mat...')
    mat = scipy.io.loadmat(path + data_name)

    #labels = mat['group']
    #labels = sparse_mx_to_torch_sparse_tensor(labels).to_dense()
    # remove the small class
    labels = torch.load(path+"blog_clean_lbl.pt")


    adj = mat['network']
    adj = sparse_mx_to_torch_sparse_tensor(adj).to_dense()
    edge_index = torch.transpose(torch.nonzero(adj), 0, 1).long()
    features = torch.eye(labels.shape[0]).float()

    # groups assignment
    n_cls = labels.shape[1]
    print(n_cls)
    cls_order1 = torch.load(path + "blog_shuffle1.pt")
    cls_order2 = torch.load(path + "blog_shuffle2.pt")
    cls_order3 = torch.load(path + "blog_shuffle3.pt")

    groups_idx1 = [tuple(cls_order1[i:i+n_cls_per_t]) for i in range(0, n_cls-1, n_cls_per_t)]
    groups_idx2 = [tuple(cls_order2[i:i+n_cls_per_t]) for i in range(0, n_cls-1, n_cls_per_t)]
    groups_idx3 = [tuple(cls_order3[i:i+n_cls_per_t]) for i in range(0, n_cls-1, n_cls_per_t)]
    groups_idxs = []
    groups_idxs.append(groups_idx1)
    groups_idxs.append(groups_idx2)
    groups_idxs.append(groups_idx3)

    cls = torch.transpose(labels, 0, 1)
    cls_asgn = {}
    for i, label in enumerate(cls):
        cls_asgn[i] = torch.nonzero(label).flatten()


    groups1 = {}
    for g in groups_idxs[0]:
        groups1[g] = []

    groups2 = {}
    for g in groups_idxs[1]:
        groups2[g] = []

    groups3 = {}
    for g in groups_idxs[2]:
        groups3[g] = []

    for g in groups_idxs[0]:
        assert g in groups1.keys()
        for c in g:
            groups1[g].extend(cls_asgn[c].tolist())


    for g in groups_idx:
        assert g in groups.keys()
        for c in g:
            groups[g].extend(cls_asgn[c].tolist())
            print(len(groups[g]))
        # note only once the nodes belong to multiple classes
        groups[g] = list(set(groups[g]))
        groups[g].sort()
        print("after set")
        print(len(groups[g]))
        print("###################")

    # splits is a nested dictionary
    # key: groups, value: (key:train, val, test; value:node_ids)
    # each time step, nodes have no more than classes seen so far

    # split the train-val-test within each class
    splits = {}
    for g in list(groups.keys()):
        split = {}
        # get all the nodes in this group
        node_ids = groups[g]

        # split the nodes into train-val-test: 60-10-30
        ids_train, ids_val_test = train_test_split(node_ids, test_size=0.4, random_state=42)
        ids_val, ids_test = train_test_split(ids_val_test, test_size=0.75, random_state=41)
        # write in the dictionary
        split["train"] = ids_train
        split["val"] = ids_val
        split["test"] = ids_test
        splits[g] = split

    print(splits)

    num_nodes = labels.shape[0]

    G = Data(x=features,
             edge_index=edge_index,
             y=labels)

    G.n_id = torch.arange(num_nodes)
    G.splits = splits
    G.groups = groups

    return G

