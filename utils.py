import torch
from torch_geometric.data import Data
import numpy as np
from torch_geometric.utils import mask
from torch_geometric.utils import subgraph


####################################################
import random
import numpy as np
import torch
import dgl
from random import sample
import os
import copy
import errno



def assign_hyp_param(args, params):
    if args.method == 'lwf':
        args.lwf_args = params
    if args.method == 'bare':
        args.bare_args = params
    if args.method == 'gem':
        args.gem_args = params
    if args.method == 'ewc':
        args.ewc_args = params
    if args.method == 'mas':
        args.mas_args = params
    if args.method == 'twp':
        args.twp_args = params
    if args.method in ['jointtrain', 'joint', 'Joint']:
        args.joint_args = params
    if args.method == 'ergnn':
        args.ergnn_args = params


def str2dict(s):
    # accepts a str like " 'k1':v1; ...; 'km':vm ", values (v1,...,vm) can be single values or lists (for hyperparameter tuning)
    output = dict()
    kv_pairs = s.replace(' ', '').replace("'", '').split(';')
    for kv in kv_pairs:
        key = kv.split(':')[0]
        v_ = kv.split(':')[1]
        if '[' in v_:
            # transform list of values
            v_list = v_.replace('[', '').replace(']', '').split(',')
            vs = []
            for v__ in v_list:
                try:
                    # if the parameter is float
                    vs.append(float(v__))
                except:
                    # if the parameter is str
                    vs.append(str(v__))
            output.update({key: vs})
        else:
            try:
                output.update({key: float(v_)})
            except:
                output.update({key: str(v_)})
    return output


def compose_hyper_params(hyp_params):
    hyp_param_list = [{}]
    for hk in hyp_params:
        hyp_param_list_ = []
        hyp_p_current = hyp_params[hk] if isinstance(hyp_params[hk], list) else [hyp_params[hk]]
        for v in hyp_p_current:
            for hk_ in hyp_param_list:
                hk__ = copy.deepcopy(hk_)
                hk__.update({hk: v})
                hyp_param_list_.append(hk__)
        hyp_param_list = hyp_param_list_
    return hyp_param_list


def mkdir_if_missing(directory):
    if not os.path.exists(directory):
        try:
            os.makedirs(directory)
        except OSError as e:
            if e.errno != errno.EEXIST:
                raise


def set_seed(args=None):
    seed = 1 if not args else args.seed

    random.seed(seed)
    np.random.seed(seed)

    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False

    dgl.random.seed(seed)


def remove_illegal_characters(name, replacement='_'):
    # replace any potential illegal characters with 'replacement'
    for c in ['-', '[', ']', '{', '}', "'", ',', ':', ' ']:
        name = name.replace(c, replacement)
    return name
####################################################

def map_edge_index(node_ids, edge_index_complete):
    # input the indices of the nodes of the subgraph in the graph,
    # transform the edge_index into the subgraph index
    num_edge = edge_index_complete.shape[1]
    map_book = {x.item(): i for i, x in enumerate(node_ids)}
    edge_index_mapped = map(lambda node: map_book[node], np.asarray(edge_index_complete.flatten()))
    edge_index = torch.Tensor(list(edge_index_mapped))
    edge_index = torch.reshape(edge_index, (2, num_edge)).long()

    return edge_index


def map_split(node_ids, split):
    # map the split ids into the subgraph
    map_book = {x.item(): i for i, x in enumerate(node_ids)}

    mapped_split = {}
    for key in split.keys():
        mapped_split[key] = list(map(lambda node: map_book[node], np.asarray(split[key])))

    return mapped_split


def find_train_ids_in_G(sub_g):
    # find the train ids of the training node in the original graph G
    node_ids_G = sub_g.n_id_original
    print(node_ids_G)
    # build a map book: subgid: Gid
    map_book = {i: x.item() for i, x in enumerate(node_ids_G)}

    train_ids_subg = sub_g.split["train"]
    train_ids_G = map(lambda node: map_book[node], train_ids_subg)

    return train_ids_G


def get_ids_per_cls_train(sub_g):
    # get the ids of training node in sub_g for each class in one nested list
    # may be duplicated ids cause of the multi-labeled nodes
    labels_subg = sub_g.y
    cls = torch.transpose(labels_subg, 0, 1)

    ids_per_class = []
    for i, label in enumerate(cls):
        ids_per_class.append(torch.nonzero(label).flatten().tolist())
    # among which the training nodes ids in the sub graph
    ids_per_class_train = []
    for ids in ids_per_class:
        ids_train = list(set(ids).intersection(set(sub_g.split["train"])))
        ids_per_class_train.append(ids_train)

    return ids_per_class_train


def map_subg_to_G(ids, sub_g):
    # find the ids of the sub_g in the original graph G
    node_ids_G = sub_g.n_id_original
    # build a map book: {subgid: Gid}
    map_book = {i: x.item() for i, x in enumerate(node_ids_G)}
    # map the ids back into the original graph
    sampled_ids_G = map(lambda node: map_book[node], ids)

    return list(sampled_ids_G)


def prepare_sub_graph(G, key, Cross_Task_Message_Passing=False):
    # prepare subgraph for one task, used both for TaskIL and ClassIL
    # target classes for each task, note for catastrophic forgetting evaluation
    target_classes = list(flatten(key))
    # sorted nodes ids in the group
    node_ids_g = torch.Tensor(G.groups[key]).int().long()

    # build node mask from the ids for the subgraph
    node_mask = mask.index_to_mask(node_ids_g, size=G.num_nodes)

    # allow nodes from other task to pass information to the nodes for this task
    if Cross_Task_Message_Passing:
        # or operation of two boolean lists
        edge_mask = node_mask[G.edge_index[0]] + node_mask[G.edge_index[1]]
        edge_index_g = G.edge_index[:, edge_mask]
        # all nodes in the subgraph(including target nodes and their neighbors) in the original graph
        node_ids_g_all = torch.unique(edge_index_g.flatten()).long()
        # index of target nodes in the graph
        target_ids_g = [i for i, n in enumerate(node_ids_g_all) if n in node_ids_g]

        # !!!!!!!!!!!!!convert target id into the subgraph !!!!!!!!!!!!!!!!!!!
        # evaluate only on the target nodes!!!!!!!!!!!!!!!!!!!!!!!!!!!
        #target_ids_sub =
        # get the edge_index in the subgraph
        edge_index_sub = map_edge_index(node_ids_g_all, edge_index_g)

    # only nodes of this task in the subgraph
    else:
        # edge index in the original graph
        edge_index_g, _ = subgraph(node_ids_g, G.edge_index, None)
        # all neighbors are in the subgraph already
        node_ids_g_all = node_ids_g
        # all nodes are target nodes
        target_ids_g = node_ids_g_all
        # node ids in the subgraph
        target_ids_sub = np.arange(node_ids_g_all.shape[0])
        # edge index in the subgraph
        edge_index_sub = map_edge_index(node_ids_g_all, edge_index_g)

    features = G.x[node_ids_g_all]
    labels = G.y[node_ids_g_all]
    # map the ids to subgraph
    split = map_split(node_ids_g_all, G.splits[key])
    # number of nodes in the subgraph
    num_nodes = node_ids_g_all.shape[0]

    sub_g = Data(x=features,
                 edge_index=edge_index_sub,
                 y=labels)

    # node id in the subgraph
    sub_g.n_id_sub = torch.arange(num_nodes)
    # node id in the original graph
    sub_g.n_id_original = node_ids_g_all
    sub_g.split = split
    sub_g.target_classes = target_classes
    # target ids in the sub graph
    sub_g.target_ids_sub = target_ids_sub
    # target ids in the original graph
    sub_g.taget_ids_g = target_ids_g

    return sub_g


def build_subgraph(node_idx, G, remove_edges=True):
    r"""Returns the induced subgraph of :obj:`(edge_index, edge_attr)`
       containing the nodes in :obj:`subset`.""
    subset, edge_index, mapping, edge_mask = subgraph(node_idx=node_idx,
                                                            edge_index=G.edge_index,
                                                            # re-index nodes from 0
                                                            relabel_nodes=True)"""
    edge_index, _ = subgraph(node_idx, G.edge_index, relabel_nodes=True)

    if remove_edges:
        # graph with no edges run into errors in pyg, manuel add self loop here
        id_sub = torch.arange(len(node_idx))
        print("id_sub")
        edge_index = torch.vstack((id_sub, id_sub))
        sub_g = Data(x=G.x[node_idx],
                     edge_index=edge_index)
    else:
        sub_g = Data(x=G.x[node_idx],
                     edge_index=edge_index)

    return sub_g



def sparse_mx_to_torch_sparse_tensor(sparse_mx):
    """Convert a scipy sparse matrix to a torch sparse tensor."""
    sparse_mx = sparse_mx.tocoo().astype(np.float32)
    indices = torch.from_numpy(
        np.vstack((sparse_mx.row, sparse_mx.col)).astype(np.int64))
    values = torch.from_numpy(sparse_mx.data)
    shape = torch.Size(sparse_mx.shape)
    return torch.sparse.FloatTensor(indices, values, shape)


def flatten(nested_data):
    for i in nested_data:
        if isinstance(i, (list, tuple)):
            for j in flatten(i):
                yield j
        else:
            yield i




