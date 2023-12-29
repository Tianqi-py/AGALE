import torch
import random
import torch.nn as nn
from torch.autograd import Variable


def kaiming_normal_init(m):
    if isinstance(m, torch.nn.Conv2d):
        torch.nn.init.kaiming_normal_(m.weight, nonlinearity='relu')
    elif isinstance(m, torch.nn.Linear):
        torch.nn.init.kaiming_normal_(m.weight, nonlinearity='sigmoid')


class Aggregator(object):

    def __init__(self):
        super(Aggregator, self).__init__()

    def aggregate(self, mask, features):
        aggregate_features = torch.matmul(mask, features)
        return aggregate_features


class SAGEConv(torch.nn.Module):

    def __init__(self, in_size, out_size):
        super(SAGEConv, self).__init__()

        self.in_size = in_size
        self.out_size = out_size
        self.weight = nn.Parameter(torch.Tensor(self.in_size * 2, self.out_size))

        self.init_parameters()

    def init_parameters(self):
        for param in self.parameters():
            nn.init.xavier_uniform_(param)

    def forward(self, x, aggregate_x):
        combine_x = torch.cat([x, aggregate_x], dim=1)
        return nn.functional.relu(torch.matmul(combine_x, self.weight))


class Sampler(object):

    def __init__(self, adj_lists):
        super(Sampler, self).__init__()
        self.adj_lists = adj_lists

    def sample_neighbors(self, nodes, num_sample=10):
        to_neighs = [self.adj_lists[int(node)] for node in nodes]
        if not num_sample is None:
            samp_neighs = [set(random.sample(to_neigh, num_sample)) if len(to_neigh) >= num_sample else to_neigh for
                           to_neigh in to_neighs]
        else:
            samp_neighs = to_neighs

        samp_neighs = [samp_neigh | set([nodes[i]]) for i, samp_neigh in enumerate(samp_neighs)]
        unique_neighs = list(set.union(*samp_neighs))
        unique_neighs_2_idxs = {n: i for i, n in enumerate(unique_neighs)}
        node_idxs = [unique_neighs_2_idxs[node] for node in nodes]

        samp_neighs = [samp_neigh - set([nodes[i]]) for i, samp_neigh in enumerate(samp_neighs)]
        nodes_2_idxs_mask = Variable(torch.zeros(len(samp_neighs), len(unique_neighs_2_idxs)))
        column_indices = [unique_neighs_2_idxs[n] for samp_neigh in samp_neighs for n in samp_neigh]
        row_indices = [i for i in range(len(samp_neighs)) for j in range(len(samp_neighs[i]))]
        nodes_2_idxs_mask[row_indices, column_indices] = 1
        num_neigh = nodes_2_idxs_mask.sum(1, keepdim=True) + 1  # In case 0 degree
        nodes_2_idxs_mask = nodes_2_idxs_mask.div(num_neigh)

        return node_idxs, unique_neighs, nodes_2_idxs_mask


