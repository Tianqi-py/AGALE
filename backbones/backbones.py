import torch
import torch.nn as nn
import torch.nn.functional as F
from torch_geometric.nn import GCNConv, GATConv, SAGEConv
from torch_geometric_temporal.nn.recurrent import EvolveGCNO
from .model_utils import kaiming_normal_init, Aggregator, SAGEConv, Sampler


class GCN(torch.nn.Module):
    def __init__(self, in_channels, hidden_channels, out_channels, args):
        super(GCN, self).__init__()
        self.in_channels = in_channels
        self.hidden_channels = hidden_channels
        self.out_channels = out_channels

        self.conv1 = GCNConv(in_channels, hidden_channels)
        self.conv2 = GCNConv(hidden_channels, out_channels)
        self.apply(kaiming_normal_init)
        self.multi_class = args.multi_class

    def forward(self, x, edge_index):
        x = F.relu(self.conv1(x, edge_index))
        x = self.conv2(x, edge_index)
        # the representations before convert to probabilities
        # used for ergnn sampler
        self.rep = x
        if not self.multi_class:
            m = torch.nn.Sigmoid()
            return m(x)
        # for multi-class dataset, calculate the acc using logits
        else:
            return x

    def add_new_outputs(self, num_new_classes):
        # Add new output units for each new class
        in_channels = self.conv2.in_channels
        out_channels = self.conv2.out_channels + num_new_classes
        new_conv2 = GCNConv(in_channels, out_channels)

        # the parameters trained for the old tasks copied to the new defined layer
        paras = [para for para in self.conv2.parameters()]
        # extend with random values and pack as iter to give to new conv2
        # the in and out dimensions are extent
        bias = torch.randn(out_channels)
        weight = torch.randn(out_channels, in_channels)
        for i, para in enumerate(paras):
            # bias
            if i == 0:
                bias[:self.conv2.out_channels] = para
            # weight
            else:
                weight[:self.conv2.out_channels] = para
        paras_new = {"bias": bias,
                     "lin.weight": weight}
        new_conv2.load_state_dict(paras_new)

        self.conv2 = new_conv2


class GAT(torch.nn.Module):
    def __init__(self, in_channels, hidden_channels, out_channels, heads, args):
        super().__init__()
        self.conv1 = GATConv(in_channels, hidden_channels, heads, dropout=0.0)
        # On the Pubmed dataset, use `heads` output heads in `conv2`.
        self.conv2 = GATConv(hidden_channels * heads, out_channels, heads=1,
                             concat=False, dropout=0.0)
        self.apply(kaiming_normal_init)
        self.multi_class = args.multi_class

    def forward(self, x, edge_index):
        x = F.dropout(x, p=0.0, training=self.training)
        x = F.elu(self.conv1(x, edge_index))
        x = F.dropout(x, p=0.0, training=self.training)
        x = self.conv2(x, edge_index)
        self.rep = x
        if self.multi_class:
            m = torch.nn.Softmax()
        else:
            m = torch.nn.Sigmoid()
        return m(x)


class EvolveGCN(torch.nn.Module):

    def __init__(self, in_channels, hidden_channels, out_channels, layer_name, multi_class=False):
        super(EvolveGCN, self).__init__()

        self.layer_name = layer_name

        self.multi_class = multi_class

        self.lins = torch.nn.ModuleList()
        self.lins[0] = torch.nn.Linear(in_channels, hidden_channels)
        self.lins[1] = torch.nn.Linear(hidden_channels, out_channels)

        self.layers = torch.nn.ModuleList()
        if self.layer_name == "EvolveGCNH":
            print("EvolveGCNH not applicable to the setting, "
                  "because of the pooling operator")

        elif self.layer_name == "EvolveGCNO":
            self.layers.append(EvolveGCNO(in_channels))
            self.layers.append(EvolveGCNO(hidden_channels))

    def forward(self, x, edge_index, edge_weight):
        h = F.relu(self.layers[0](x, edge_index, edge_weight))
        h = F.dropout(self.lins[0](h), 0.0)
        h = self.layers[1](h, edge_index, edge_weight)
        h = self.lins[1](h)

        if self.multi_class:
            m = torch.nn.Softmax()
        else:
            m = torch.nn.Sigmoid()
        return m(h)

# class SAGE(torch.nn.Module):
#     def __init__(self, in_channels, hidden_channels, out_channels):
#         super().__init__()
#         self.convs = torch.nn.ModuleList()
#         self.convs.append(SAGEConv(in_channels, hidden_channels))
#         self.convs.append(SAGEConv(hidden_channels, out_channels))
#
#     def forward(self, x, edge_index):
#         for i, conv in enumerate(self.convs):
#             x = conv(x, edge_index)
#             if i < len(self.convs) - 1:
#                 x = x.relu_()
#                 x = F.dropout(x, p=0.5, training=self.training)
#         return x


class GraphSAGE(nn.Module):

    def __init__(self, layers, in_features, adj_lists, args, multi_class=False):
        super(GraphSAGE, self).__init__()

        self.layers = layers
        self.num_layers = len(layers) - 2
        self.in_features = torch.Tensor(in_features).to(args.device)
        self.adj_lists = adj_lists
        self.num_neg_samples = args.num_neg_samples
        self.device = args.device

        self.convs = nn.ModuleList()
        for i in range(self.num_layers):
            self.convs.append(SAGEConv(layers[i], layers[i + 1]))
        self.sampler = Sampler(adj_lists)
        self.aggregator = Aggregator()

        self.weight = nn.Parameter(torch.Tensor(layers[-2], layers[-1]))

        self.multi_class = multi_class
        if self.multi_class:
            self.xtent = nn.CrossEntropyLoss()
        else:
            self.xent = nn.BCELoss()

        self.init_parameters()


    def init_parameters(self):
        for param in self.parameters():
            nn.init.xavier_uniform_(param)

    def forward(self, nodes):
        layer_nodes, layer_mask = self._generate_layer_nodes(nodes)
        features = self.in_features[layer_nodes[0]]
        for i in range(self.num_layers):
            cur_nodes, mask = layer_nodes[i + 1], layer_mask[i]
            aggregate_features = self.aggregator.aggregate(mask, features)
            features = self.convs[i].forward(x=features[cur_nodes], aggregate_x=aggregate_features)

        if self.multi_class:
            m = torch.nn.Softmax()
        else:
            m = torch.nn.Sigmoid()

        return m(torch.matmul(features, self.weight), 1)

    def loss(self, nodes, labels=None):
        preds = self.forward(nodes)
        return self.xent(preds, labels.squeeze())

    def _generate_layer_nodes(self, nodes):
        layer_nodes = list([nodes])
        layer_mask = list()
        for i in range(self.num_layers):
            nodes_idxs, unique_neighs, mask = self.sampler.sample_neighbors(layer_nodes[0])
            layer_nodes[0] = nodes_idxs
            layer_nodes.insert(0, unique_neighs)
            layer_mask.insert(0, mask.to(self.device))
        return layer_nodes, layer_mask

    def get_embeds(self, nodes):
        layer_nodes, layer_mask = self._generate_layer_nodes(nodes)
        features = self.in_features[layer_nodes[0]]
        for i in range(self.num_layers):
            cur_nodes, mask = layer_nodes[i + 1], layer_mask[i]
            aggregate_features = self.aggregator.aggregate(mask, features)
            features = self.convs[i].forward(x=features[cur_nodes], aggregate_x=aggregate_features)
        return features.data.numpy()

