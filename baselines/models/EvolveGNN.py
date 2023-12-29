import torch
from torch.nn import GRU
from torch_geometric.nn import GCNConv
from metric import BCE_loss, f1_Score, _eval_rocauc, ap_score, CE_loss, accuracy
from utils import build_subgraph
from Sampler import CM_sampler
from utils import get_ids_per_cls_train, map_subg_to_G
import torch.optim as optim
from torch_geometric_temporal.nn.recurrent import EvolveGCNO
import torch.nn.functional as F

class EvolveGNN(torch.nn.Module):

    def __init__(self, in_channels, hidden_channels, out_channels, variant_name, args):
        super(EvolveGNN, self).__init__()

        self.layer_name = variant_name

        self.lins = torch.nn.ModuleList()

        self.lins.append(torch.nn.Linear(in_channels, hidden_channels))
        self.lins.append(torch.nn.Linear(hidden_channels, out_channels))

        if self.layer_name == "EvolveGCNH":
            print("EvolveGCNH not applicable to the setting, "
                  "because of the pooling operator")

        elif self.layer_name == "O":
            self.conv1 = EvolveGCNO(in_channels)
            self.conv2 = EvolveGCNO(hidden_channels)

        self.optimizer = optim.Adam(self.parameters(), lr=args.lr, weight_decay=args.weight_decay)
        self.multi_class = args.multi_class

    def forward(self, x, edge_index, edge_weight=None):

        h = F.relu(self.conv1(x, edge_index, edge_weight))
        h = F.dropout(self.lins[0](h), 0.5)
        h = self.conv2(h, edge_index, edge_weight)
        h = self.lins[1](h)

        if not self.multi_class:
            m = torch.nn.Sigmoid()
            return m(h)
        else:
            return h

    def add_new_outputs(self, num_new_classes):
        # Add new output units for each new class
        # linear layer has in_features and out_features
        in_channels = self.lins[1].in_features
        out_channels = self.lins[1].out_features + num_new_classes
        new_lin = torch.nn.Linear(in_channels, out_channels)

        # the parameters trained for the old tasks copied to the new defined layer
        paras = [para for para in self.lins[1].parameters()]
        # extend with random values and pack as iter to give to new conv2
        # the in and out dimensions are extent
        bias = torch.randn(out_channels)
        weight = torch.randn(out_channels, in_channels)
        for i, para in enumerate(paras):
            # weight first for linear layer!!!
            if i == 0:
                weight[:self.lins[1].out_features] = para
            # bias
            else:
                bias[:self.lins[1].out_features] = para
        paras_new = {"bias": bias,
                     "weight": weight}
        new_lin.load_state_dict(paras_new)

        self.lins[1] = new_lin


    def ClassIL_train(self, sub_g):

        self.train()
        self.zero_grad()
        out = self.forward(sub_g.x, sub_g.edge_index)
        y = sub_g.y

        if not self.multi_class:
            # evaluation on current task
            loss_train = BCE_loss(out[sub_g.split["train"]], y[sub_g.split["train"]])
            micro_train, macro_train = f1_Score(y[sub_g.split["train"]], out[sub_g.split["train"]])
            roc_auc_train = _eval_rocauc(y[sub_g.split["train"]], out[sub_g.split["train"]])
            ap_train = ap_score(y[sub_g.split["train"]], out[sub_g.split["train"]])

            train_metric = {}

            train_metric["micro"] = micro_train
            train_metric["macro"] = macro_train
            train_metric["auroc"] = roc_auc_train
            train_metric["ap"] = ap_train
            train_metric["loss"] = loss_train
        else:
            loss_train = CE_loss(out[sub_g.split["train"]], y[sub_g.split["train"]])
            acc_train = accuracy(out[sub_g.split["train"]], y[sub_g.split["train"]])

            train_metric = {}

            train_metric["loss"] = loss_train
            train_metric["acc"] = acc_train

        loss_train.backward()
        self.optimizer.step()

        return train_metric

    def TaskIL_train(self, sub_g):

        self.train()
        self.zero_grad()
        out = self.forward(sub_g.x, sub_g.edge_index)
        # offsets for the current
        num_classes = sub_g.y.shape[1]
        out = out[:, -num_classes:]

        y = sub_g.y

        if not self.multi_class:
            # evaluation on current task
            loss_train = BCE_loss(out[sub_g.split["train"]], y[sub_g.split["train"]])
            micro_train, macro_train = f1_Score(y[sub_g.split["train"]], out[sub_g.split["train"]])
            roc_auc_train = _eval_rocauc(y[sub_g.split["train"]], out[sub_g.split["train"]])
            ap_train = ap_score(y[sub_g.split["train"]], out[sub_g.split["train"]])


            train_metric = {}

            train_metric["micro"] = micro_train
            train_metric["macro"] = macro_train
            train_metric["auroc"] = roc_auc_train
            train_metric["ap"] = ap_train
            train_metric["loss"] = loss_train

        else:
            loss_train = CE_loss(out[sub_g.split["train"]], y[sub_g.split["train"]])
            acc_train = accuracy(out[sub_g.split["train"]], y[sub_g.split["train"]])

            train_metric = {}

            train_metric["loss"] = loss_train
            train_metric["acc"] = acc_train

        loss_train.backward()
        self.optimizer.step()

        return train_metric

    @torch.no_grad()
    def standard_test(self, sub_g, pre_eval=False, TaskIL=False, t=0, target_classes_groups=None):

        out = self.forward(sub_g.x, sub_g.edge_index)
        y = sub_g.y

        # test on precious task
        if pre_eval:
            # TaskIL
            if TaskIL:
                off1 = 0
                if t == 0:
                    off2 = len(target_classes_groups[0])
                    print("test on t0: off1 and off2", off1, off2)
                else:
                    for i in range(t):
                        off1 = off1 + len(target_classes_groups[i])
                    off2 = off1 + len(target_classes_groups[t])
                    print("test on t", t, ": off1 and off2", off1, off1)
                out = out[:, off1:off2]

            # ClassIL
            else:
                out = out[:, :y.shape[1]]
                print("test on previous task, the shape of true labels in the previous task")

        # test on the current task
        else:
            # TaskIL
            if TaskIL:
                num_target_class = len(sub_g.target_classes)
                out = out[:, -num_target_class:]
            # ClassIl: do nothing

        if not self.multi_class:

            loss_val = BCE_loss(out[sub_g.split["val"]], y[sub_g.split["val"]])
            micro_val, macro_val = f1_Score(y[sub_g.split["val"]], out[sub_g.split["val"]])
            roc_auc_val = _eval_rocauc(y[sub_g.split["val"]], out[sub_g.split["val"]])
            ap_val = ap_score(y[sub_g.split["val"]], out[sub_g.split["val"]])

            micro_test, macro_test = f1_Score(y[sub_g.split["test"]], out[sub_g.split["test"]])
            roc_auc_test = _eval_rocauc(y[sub_g.split["test"]], out[sub_g.split["test"]])
            ap_test = ap_score(y[sub_g.split["test"]], out[sub_g.split["test"]])

            val_metric = {}
            val_metric["micro"] = micro_val
            val_metric["macro"] = macro_val
            val_metric["auroc"] = roc_auc_val
            val_metric["ap"] = ap_val
            val_metric["loss"] = float(loss_val)

            test_metric = {}
            test_metric["micro"] = micro_test
            test_metric["macro"] = macro_test
            test_metric["auroc"] = roc_auc_test
            test_metric["ap"] = ap_test

        else:
            loss_val = CE_loss(out[sub_g.split["val"]], y[sub_g.split["val"]])
            acc_val = accuracy(out[sub_g.split["val"]], y[sub_g.split["val"]])

            loss_test = CE_loss(out[sub_g.split["test"]], y[sub_g.split["test"]])
            acc_test = accuracy(out[sub_g.split["test"]], y[sub_g.split["test"]])

            val_metric = {}
            val_metric["loss"] = loss_val
            val_metric["acc"] = acc_val

            test_metric = {}
            test_metric["loss"] = loss_test
            test_metric["acc"] = acc_test

        return val_metric, test_metric