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

class ERGNN(torch.nn.Module):
    def __init__(self, backbone, budget, d, args):
        super(ERGNN, self).__init__()
        self.backbone = backbone
        self.sampler = CM_sampler(plus=False)
        self.optimizer = optim.Adam(self.backbone.parameters(), lr=args.lr, weight_decay=args.weight_decay)
        # setup memories
        self.current_task = -1
        self.buffer_node_ids = []
        self.buffer_lbl = []
        # buffer size 1000 nodes
        self.budget = int(budget[1])
        self.d_CM = d  # d for CM sampler of ERGNN
        self.aux_g = None
        self.multi_class = args.multi_class

    def backbone_forward(self, sub_g):
        output = self.backbone(sub_g.x, sub_g.edge_index)
        return output

    def add_new_outputs(self, num_new_classes):
        # Add new output units for each new class
        in_channels = self.backbone.conv2.in_channels
        out_channels = self.backbone.conv2.out_channels + num_new_classes
        new_conv2 = GCNConv(in_channels, out_channels)

        # the parameters trained for the old tasks copied to the new defined layer
        paras = [para for para in self.backbone.conv2.parameters()]
        # extend with random values and pack as iter to give to new conv2
        # the in and out dimensions are extent
        bias = torch.randn(out_channels)
        weight = torch.randn(out_channels, in_channels)
        for i, para in enumerate(paras):
            # bias
            if i == 0:
                bias[:self.backbone.conv2.out_channels] = para
            # weight
            else:
                weight[:self.backbone.conv2.out_channels] = para
        paras_new = {"bias": bias,
                     "lin.weight": weight}
        new_conv2.load_state_dict(paras_new)

        self.backbone.conv2 = new_conv2

    def TaskIL_train(self, sub_g, t, target_classes_groups, G):

        self.backbone.train()
        n_nodes = len(sub_g.split["train"])
        buffer_size = len(self.buffer_node_ids)
        # empirical assignment for beta: weight between the current task and the buffer data
        beta = buffer_size / (buffer_size + n_nodes)
        # TaskIL need buffer nodes to be saved in separate subgs
        if not isinstance(self.buffer_node_ids, dict):
            self.buffer_node_ids = {}
            self.buffer_lbl = {}

        # offsets for the current
        num_classes = sub_g.y.shape[1]

        print("number of classes in the current task", num_classes)

        self.backbone.zero_grad()
        out = self.backbone(sub_g.x, sub_g.edge_index)
        # read the target classes output units
        print("the total number of output unit", out.shape)
        out = out[:, -num_classes:]
        print("the shape of the output:", out.shape)
        y = sub_g.y

        if self.multi_class:
            # for multi-class datasets, use cross-entropy loss and acc
            loss_train = CE_loss(out[sub_g.split["train"]], y[sub_g.split["train"]])
            acc_train = accuracy(out[sub_g.split["train"]], y[sub_g.split["train"]])

            loss_val = CE_loss(out[sub_g.split["val"]], y[sub_g.split["val"]])
            acc_val = accuracy(out[sub_g.split["val"]], y[sub_g.split["val"]])

            loss_test = CE_loss(out[sub_g.split["test"]], y[sub_g.split["test"]])
            acc_test = accuracy(out[sub_g.split["test"]], y[sub_g.split["test"]])


            train_metric = {}
            train_metric["loss"] = loss_train
            train_metric["acc"] = acc_train

            val_metric = {}
            val_metric["loss"] = loss_val
            val_metric["acc"] = acc_val

            test_metric = {}
            test_metric["loss"] = loss_test
            test_metric["acc"] = acc_test

        else:
            # multi-label evaluation on current task
            loss_train = BCE_loss(out[sub_g.split["train"]], y[sub_g.split["train"]])
            micro_train, macro_train = f1_Score(y[sub_g.split["train"]], out[sub_g.split["train"]])
            roc_auc_train = _eval_rocauc(y[sub_g.split["train"]], out[sub_g.split["train"]])
            ap_train = ap_score(y[sub_g.split["train"]], out[sub_g.split["train"]])

            loss_val = BCE_loss(out[sub_g.split["val"]], y[sub_g.split["val"]])
            micro_val, macro_val = f1_Score(y[sub_g.split["val"]], out[sub_g.split["val"]])
            roc_auc_val = _eval_rocauc(y[sub_g.split["val"]], out[sub_g.split["val"]])
            ap_val = ap_score(y[sub_g.split["val"]], out[sub_g.split["val"]])

            micro_test, macro_test = f1_Score(y[sub_g.split["test"]], out[sub_g.split["test"]])
            roc_auc_test = _eval_rocauc(y[sub_g.split["test"]], out[sub_g.split["test"]])
            ap_test = ap_score(y[sub_g.split["test"]], out[sub_g.split["test"]])

            train_metric = {}

            train_metric["micro"] = micro_train
            train_metric["macro"] = macro_train
            train_metric["auroc"] = roc_auc_train
            train_metric["ap"] = ap_train

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

        # encountered new task, add new samples to buffer
        if t != self.current_task:
            self.current_task = t
            # new task coming, add the new data to buffer, the added id should be IDs in The original graph G
            ids_per_cls_train = get_ids_per_cls_train(sub_g)  # ids in the sub_g
            sampled_ids = self.sampler(ids_per_cls_train, self.budget, sub_g.x, self.backbone.rep,
                                       self.d_CM)  # in sub_g
            # the labels of the sampled nodes in the sub_g
            lbs_sampled = sub_g.y[sampled_ids]

            # map the sampled ids into original graph
            sampled_ids_G = map_subg_to_G(sampled_ids, sub_g)

            # save the nodes from different tasks along with their labels as separate subgraph
            self.buffer_node_ids[t] = sampled_ids_G
            self.buffer_lbl[t] = lbs_sampled

        # until here: current task loss calculated
        loss = loss_train
        # calculate the loss on buffer nodes
        # note nodes from different time steps are tested on their own target classes
        if t != 0:
            off1 = 0
            off2 = len(target_classes_groups[0])
            for i in range(t):
                if i > 0:
                    off1 = off2
                    off2 = off1 + len(target_classes_groups[i])
                # feed the buffer nodes into the model
                # construct subgraph for the nodes in buffer, subg_buffer only has features
                subg_buffer = build_subgraph(node_idx=self.buffer_node_ids[i],
                                             G=G)  # subgraph constructed by the buffer nodes
                out_buffer = self.backbone(subg_buffer.x, subg_buffer.edge_index)[:, off1:off2]

                # calculate loss on buffer data
                if self.multi_class:
                    loss_buffer = CE_loss(out_buffer, self.buffer_lbl[i])
                else:
                    loss_buffer = BCE_loss(out_buffer, self.buffer_lbl[i])

                # overall loss
                loss = beta * loss + (1 - beta) * loss_buffer

        # back-prop
        loss.backward()
        self.optimizer.step()
        train_metric["loss"] = loss

        return train_metric, val_metric, test_metric


    def ClassIL_train(self, sub_g, G, t):

        self.backbone.train()
        n_nodes = len(sub_g.split["train"])
        buffer_size = len(self.buffer_node_ids)
        # empirical assignment for beta: weight between the current task and the buffer data
        beta = buffer_size / (buffer_size + n_nodes)

        self.backbone.zero_grad()
        out = self.backbone(sub_g.x, sub_g.edge_index)
        y = sub_g.y

        # evaluation on current task
        if self.multi_class:
            # for multi-class datasets, use cross-entropy loss and acc
            loss_train = CE_loss(out[sub_g.split["train"]], y[sub_g.split["train"]])
            acc_train = accuracy(out[sub_g.split["train"]], y[sub_g.split["train"]])

            loss_val = CE_loss(out[sub_g.split["val"]], y[sub_g.split["val"]])
            acc_val = accuracy(out[sub_g.split["val"]], y[sub_g.split["val"]])

            loss_test = CE_loss(out[sub_g.split["test"]], y[sub_g.split["test"]])
            acc_test = accuracy(out[sub_g.split["test"]], y[sub_g.split["test"]])


            train_metric = {}
            train_metric["loss"] = loss_train
            train_metric["acc"] = acc_train

            val_metric = {}
            val_metric["loss"] = loss_val
            val_metric["acc"] = acc_val

            test_metric = {}
            test_metric["loss"] = loss_test
            test_metric["acc"] = acc_test

        else:
            loss_train = BCE_loss(out[sub_g.split["train"]], y[sub_g.split["train"]])
            micro_train, macro_train = f1_Score(y[sub_g.split["train"]], out[sub_g.split["train"]])
            roc_auc_train = _eval_rocauc(y[sub_g.split["train"]], out[sub_g.split["train"]])
            ap_train = ap_score(y[sub_g.split["train"]], out[sub_g.split["train"]])

            loss_val = BCE_loss(out[sub_g.split["val"]], y[sub_g.split["val"]])
            micro_val, macro_val = f1_Score(y[sub_g.split["val"]], out[sub_g.split["val"]])
            roc_auc_val = _eval_rocauc(y[sub_g.split["val"]], out[sub_g.split["val"]])
            ap_val = ap_score(y[sub_g.split["val"]], out[sub_g.split["val"]])

            micro_test, macro_test = f1_Score(y[sub_g.split["test"]], out[sub_g.split["test"]])
            roc_auc_test = _eval_rocauc(y[sub_g.split["test"]], out[sub_g.split["test"]])
            ap_test = ap_score(y[sub_g.split["test"]], out[sub_g.split["test"]])

            train_metric = {}

            train_metric["micro"] = micro_train
            train_metric["macro"] = macro_train
            train_metric["auroc"] = roc_auc_train
            train_metric["ap"] = ap_train

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

        # loss on the buffer data
        loss_buffer = 0
        # encountered new task
        if t != self.current_task:
            self.current_task = t
            print("number of nodes in the buffer until last task")
            print(len(self.buffer_node_ids))
            # new task coming, add the new data to buffer, the added id should be IDs in The original graph G
            ids_per_cls_train = get_ids_per_cls_train(sub_g)  # ids in the sub_g
            sampled_ids = self.sampler(ids_per_cls_train, self.budget, sub_g.x, self.backbone.rep, self.d_CM)  # in sub_g
            # the labels of the sampled nodes in the sub_g
            lbs_sampled = sub_g.y[sampled_ids]

            # map the sampled ids into original graph
            sampled_ids_G = map_subg_to_G(sampled_ids, sub_g)

            # update the buffer
            if t == 0:
                # t=0: add sampled nodes with their labels to the buffer
                self.buffer_node_ids.extend(sampled_ids_G)
                self.buffer_lbl = lbs_sampled

            else:
                # t>0: update the label vector of the sampled nodes from previous time steps
                # extend the existing lbl in the buffer to current cls_seen dimensions
                to_pad_dim = sub_g.y.shape[1] - self.buffer_lbl.shape[1]
                pad_vector = torch.zeros((len(self.buffer_node_ids), to_pad_dim))
                self.buffer_lbl = torch.hstack((self.buffer_lbl, pad_vector))

                # add the sampled nodes to the buffer
                for i, ind in enumerate(sampled_ids_G):  # the index of the node ids(G) in sampled_ids_G  # and node ids(G)
                    # new coming nodes: add to buffer
                    if ind not in self.buffer_node_ids:
                        self.buffer_node_ids.append(ind)
                        # the newly added label vectors have the longest labels
                        self.buffer_lbl = torch.vstack((self.buffer_lbl, lbs_sampled[i]))
                    # existing nodes in the buffer got added again: update the labels
                    else:
                        index = self.buffer_node_ids.index(ind)
                        self.buffer_lbl[index] = lbs_sampled[i]

        # at t0, no data in the buffer, do nothing
        # at t>0, calculate the loss on the buffer data
        if t != 0:
            # construct subgraph for the nodes in buffer, subg_buffer only has features and edges
            # read labels from self.buffer_lbl
            subg_buffer = build_subgraph(node_idx=self.buffer_node_ids, G=G)    #subgraph constructed by the buffer nodes
            # feed the buffer nodes into the model
            out_buffer = self.backbone(subg_buffer.x, subg_buffer.edge_index)

            # calculate loss on buffer data
            if self.multi_class:
                loss_buffer = CE_loss(out_buffer, self.buffer_lbl)
            else:
                # all nodes in the buffer are from training nodes
                loss_buffer = BCE_loss(out_buffer, self.buffer_lbl)

        # overall loss
        loss = beta * loss_train + (1-beta) * loss_buffer
        train_metric["loss"] = loss
        # back-prop
        loss.backward()
        self.optimizer.step()

        return train_metric, val_metric, test_metric


    @torch.no_grad()
    def standard_test(self, sub_g, pre_eval=False, TaskIL=False, t=0, target_classes_groups=None):
        out = self.backbone(sub_g.x, sub_g.edge_index)
        y = sub_g.y

        # test on previous task
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
                    print("test on t", t, ": off1 and off2", off1, off2)
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
            # ClassIL: do nothing

        if self.multi_class:
            # for multi-class datasets, use cross-entropy loss and acc
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

        else:

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

        return val_metric, test_metric