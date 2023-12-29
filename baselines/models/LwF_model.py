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


class LwF_Task_IL(torch.nn.Module):
    def __init__(self, backbone, args):
        super(LwF_Task_IL, self).__init__()
        self.backbone = backbone
        self.multi_class = args.multi_class
    #
    # def freeze_para(self):
    #     # Freeze the weights of the existing output units for old task
    #     for param in self.backbone.conv2.parameters():
    #         param.requires_grad = False
    #     # freeze the shared parameters also
    #     for param in self.backbone.conv1.parameters():
    #         param.requires_grad = False
    #
    # def unfreeze_para(self):
    #     # unfreeze the weights of the existing output units
    #     # the dimension of conv2 would be old units + new units
    #     for param in self.backbone.conv2.parameters():
    #         param.requires_grad = True
    #     for param in self.backbone.conv1.parameters():
    #         param.requires_grad = True

    def add_new_outputs(self, num_new_classes):
        # Add new output units for each new class
        in_channels = self.backbone.conv2.in_channels
        out_channels = self.backbone.conv2.out_channels + num_new_classes
        new_conv2 = GCNConv(in_channels, out_channels)

        # the parameters trained for the old tasks copied to the new defined layer
        paras = [para for para in self.backbone.conv2.parameters()]
        # extend with random values and pack as iter to give to new conv2
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

    def standard_train(self, sub_g, optimizer):
        # t0 training, train and test on the same task
        # subgraph for a task
        self.backbone.train()
        optimizer.zero_grad()
        out = self.backbone(sub_g.x, sub_g.edge_index)

        # only evaluating on the target classes
        # target classes determined by offset
        # out = out[:, sub_g.target_classes]
        y = sub_g.y

        if self.multi_class:
            # for multi-class datasets, use cross-entropy loss and acc
            loss_train = CE_loss(out[sub_g.split["train"]], y[sub_g.split["train"]])
            acc_train = accuracy(out[sub_g.split["train"]], y[sub_g.split["train"]])

            train_metric = {}
            train_metric["loss"] = loss_train
            train_metric["acc"] = acc_train

        else:
            loss = BCE_loss(out[sub_g.split["train"]], y[sub_g.split["train"]])
            micro_train, macro_train = f1_Score(y[sub_g.split["train"]], out[sub_g.split["train"]])
            roc_auc_train = _eval_rocauc(y[sub_g.split["train"]], out[sub_g.split["train"]])
            ap_train = ap_score(y[sub_g.split["train"]], out[sub_g.split["train"]])
            loss.backward()
            optimizer.step()

            train_metric = {}
            train_metric["micro"] = micro_train
            train_metric["macro"] = macro_train
            train_metric["auroc"] = roc_auc_train
            train_metric["ap"] = ap_train
            train_metric["loss"] = float(loss)

        return train_metric

    # def warm_up(self, sub_g, optimizer):
    #     # freeze the para for the old tasks
    #     # train the model on the new task
    #     # subgraph for a task
    #     self.backbone.train()
    #     optimizer.zero_grad()
    #     out = self.backbone(sub_g.x, sub_g.edge_index)
    #
    #     # take the output dimensions for the new task
    #     num_target_class = len(sub_g.target_classes)
    #     out = out[:, -num_target_class:]
    #     y = sub_g.y
    #
    #     loss = BCE_loss(out[sub_g.split["train"]], y[sub_g.split["train"]])
    #     micro_train, macro_train = f1_Score(y[sub_g.split["train"]], out[sub_g.split["train"]])
    #     roc_auc_train = _eval_rocauc(y[sub_g.split["train"]], out[sub_g.split["train"]])
    #     ap_train = ap_score(y[sub_g.split["train"]], out[sub_g.split["train"]])
    #     loss.backward()
    #     optimizer.step()
    #
    #     train_metric = {}
    #     train_metric["micro"] = micro_train
    #     train_metric["macro"] = macro_train
    #     train_metric["auroc"] = roc_auc_train
    #     train_metric["ap"] = ap_train
    #     train_metric["loss"] = float(loss)
    #
    #     return train_metric

    def joint_opt(self, prev_mod, sub_g, optimizer, t, target_classes_groups):

        # subgraph for a task
        self.backbone.train()
        optimizer.zero_grad()
        out = self.backbone(sub_g.x, sub_g.edge_index)

        # loss on the new task
        # target classes determined by offset not index
        num_target_class = len(sub_g.target_classes)
        out_n = out[:, -num_target_class:]
        y = sub_g.y
        if self.multi_class:
            # for multi-class datasets, use cross-entropy loss and acc
            loss_n = CE_loss(out_n[sub_g.split["train"]], y[sub_g.split["train"]])
        else:
            loss_n = BCE_loss(out_n[sub_g.split["train"]], y[sub_g.split["train"]])

        loss = loss_n
        #######################################
        # knowledge distillation
        target = prev_mod.backbone(sub_g.x, sub_g.edge_index)
        off1 = 0
        for tt in range(t):
            off2 = off1 + len(target_classes_groups[tt])
            out_o = out[:, off1:off2]
            target_o = target[:, off1:off2]
            if self.multi_class:
                # for multi-class datasets, use cross-entropy loss and acc
                loss_o = CE_loss(out_o[sub_g.split["train"]], target_o[sub_g.split["train"]])
            else:
                loss_o = BCE_loss(out_o[sub_g.split["train"]], target_o[sub_g.split["train"]])
            loss = loss + loss_o
            off1 = off2

        loss.backward()
        optimizer.step()

        if self.multi_class:
            # for multi-class datasets, use cross-entropy loss and acc
            loss_train = CE_loss(out_n[sub_g.split["train"]], y[sub_g.split["train"]])
            acc_train = accuracy(out_n[sub_g.split["train"]], y[sub_g.split["train"]])

            train_metric = {}
            train_metric["loss"] = loss_train
            train_metric["acc"] = acc_train

        else:

            micro_train, macro_train = f1_Score(y[sub_g.split["train"]], out_n[sub_g.split["train"]])
            roc_auc_train = _eval_rocauc(y[sub_g.split["train"]], out_n[sub_g.split["train"]])
            ap_train = ap_score(y[sub_g.split["train"]], out_n[sub_g.split["train"]])

            train_metric = {}
            train_metric["micro"] = micro_train
            train_metric["macro"] = macro_train
            train_metric["auroc"] = roc_auc_train
            train_metric["ap"] = ap_train
            train_metric["loss"] = float(loss)

        return train_metric

    @torch.no_grad()
    def standard_test(self, sub_g):
        # test on current task
        self.backbone.eval()
        out = self.backbone(sub_g.x, sub_g.edge_index)

        num_target_class = len(sub_g.target_classes)
        out = out[:, -num_target_class:]
        y = sub_g.y

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

        return val_metric, test_metric, out

    @torch.no_grad()
    def eva_pre_tasks(self, sub_g, target_classes_groups, t):

        # test on previous task: sub_g is the corresponding subgraph
        # t is the time step for the previous task
        # get the output
        self.backbone.eval()
        out = self.backbone(sub_g.x, sub_g.edge_index)

        offset1 = 0
        for i in range(t):
            offset1 = offset1 + len(target_classes_groups[i])

        offset2 = len(target_classes_groups[t])

        out = out[:, offset1:offset1 + offset2]
        y = sub_g.y

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

        return val_metric, test_metric, out



class LwF_Class_IL(torch.nn.Module):
    # class il lwf
    def __init__(self, backbone, args):
        super(LwF_Class_IL, self).__init__()
        self.backbone = backbone
        self.multi_class = args.multi_class
    #
    # def freeze_para(self):
    #     # Freeze the weights of the existing output units
    #     for param in self.backbone.conv2.parameters():
    #         param.requires_grad = False
    #         # freeze the shared parameters also
    #     for param in self.backbone.conv1.parameters():
    #         param.requires_grad = False
    #
    # def unfreeze_para(self):
    #     # unfreeze the weights of the existing output units
    #     # the dimension of conv2 would be old units + new units
    #     for param in self.backbone.conv2.parameters():
    #         param.requires_grad = True
    #     for param in self.backbone.conv1.parameters():
    #         param.requires_grad = True

    def add_new_outputs(self, num_new_classes):
        # Add new output units for each new class
        in_channels = self.backbone.conv2.in_channels
        out_channels = self.backbone.conv2.out_channels + num_new_classes
        new_conv2 = GCNConv(in_channels, out_channels)

        # the parameters trained for the old tasks copied to the new defined layer
        paras = [para for para in self.backbone.conv2.parameters()]
        # extend with random values and pack as iter to give to new conv2
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

    def standard_train(self, sub_g, optimizer):
        # t0 training, train and test on the same task
        # subgraph for a task
        self.backbone.train()
        optimizer.zero_grad()
        out = self.backbone(sub_g.x, sub_g.edge_index)

        # only evaluating on the target classes, which determined by offset
        # out = out[:, sub_g.target_classes]
        y = sub_g.y

        if self.multi_class:
            # for multi-class datasets, use cross-entropy loss and acc
            loss_train = CE_loss(out[sub_g.split["train"]], y[sub_g.split["train"]])
            acc_train = accuracy(out[sub_g.split["train"]], y[sub_g.split["train"]])

            train_metric = {}
            train_metric["loss"] = loss_train
            train_metric["acc"] = acc_train

        else:

            loss = BCE_loss(out[sub_g.split["train"]], y[sub_g.split["train"]])
            micro_train, macro_train = f1_Score(y[sub_g.split["train"]], out[sub_g.split["train"]])
            roc_auc_train = _eval_rocauc(y[sub_g.split["train"]], out[sub_g.split["train"]])
            ap_train = ap_score(y[sub_g.split["train"]], out[sub_g.split["train"]])
            loss.backward()
            optimizer.step()

            train_metric = {}
            train_metric["micro"] = micro_train
            train_metric["macro"] = macro_train
            train_metric["auroc"] = roc_auc_train
            train_metric["ap"] = ap_train
            train_metric["loss"] = float(loss)

        return train_metric

    @torch.no_grad()
    def standard_test(self, sub_g):
        # test on current task
        self.backbone.eval()
        out = self.backbone(sub_g.x, sub_g.edge_index)
        y = sub_g.y

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

        return val_metric, test_metric, out
    #
    # def warm_up(self, sub_g, optimizer):
    #     # freeze the para for the old tasks
    #     # train the model on the new task
    #     # subgraph for a task
    #     self.backbone.train()
    #     optimizer.zero_grad()
    #     out = self.backbone(sub_g.x, sub_g.edge_index)
    #
    #     # take the output dimensions for the new task
    #     # new task include all output units: class incremental
    #     print(out.shape)
    #     y = sub_g.y
    #     print(y.shape)
    #
    #     loss = BCE_loss(out[sub_g.split["train"]], y[sub_g.split["train"]])
    #     micro_train, macro_train = f1_Score(y[sub_g.split["train"]], out[sub_g.split["train"]])
    #     roc_auc_train = _eval_rocauc(y[sub_g.split["train"]], out[sub_g.split["train"]])
    #     ap_train = ap_score(y[sub_g.split["train"]], out[sub_g.split["train"]])
    #     loss.backward()
    #     optimizer.step()
    #
    #     train_metric = {}
    #     train_metric["micro"] = micro_train
    #     train_metric["macro"] = macro_train
    #     train_metric["auroc"] = roc_auc_train
    #     train_metric["ap"] = ap_train
    #     train_metric["loss"] = float(loss)
    #
    #     return train_metric

    def joint_opt(self, prev_mod, sub_g, optimizer):

        # subgraph for a task
        self.backbone.train()
        optimizer.zero_grad()
        out = self.backbone(sub_g.x, sub_g.edge_index)

        # loss on the new task
        # target classes determined by offset not index
        # num_target_class = len(sub_g.target_classes)
        out_n = out
        y = sub_g.y

        if self.multi_class:
            loss_n = CE_loss(out_n[sub_g.split["train"]], y[sub_g.split["train"]])
        else:
            loss_n = BCE_loss(out_n[sub_g.split["train"]], y[sub_g.split["train"]])
        loss = loss_n

        # knowledge distillation
        target = prev_mod.backbone(sub_g.x, sub_g.edge_index)

        out_o = out[:, :target.shape[1]]

        if self.multi_class:
            loss_o = CE_loss(out_o[sub_g.split["train"]], target[sub_g.split["train"]])
        else:
            loss_o = BCE_loss(out_o[sub_g.split["train"]], target[sub_g.split["train"]])

        loss = loss + loss_o

        loss.backward()
        optimizer.step()

        # evaluate on the new task,
        # may not always get better, because loss is also calculated on the old tasks
        if self.multi_class:
            # for multi-class datasets, use cross-entropy loss and acc
            loss_train = CE_loss(out_n[sub_g.split["train"]], y[sub_g.split["train"]])
            acc_train = accuracy(out_n[sub_g.split["train"]], y[sub_g.split["train"]])

            train_metric = {}
            train_metric["loss"] = loss_train
            train_metric["acc"] = acc_train

        else:
            micro_train, macro_train = f1_Score(y[sub_g.split["train"]], out_n[sub_g.split["train"]])
            roc_auc_train = _eval_rocauc(y[sub_g.split["train"]], out_n[sub_g.split["train"]])
            ap_train = ap_score(y[sub_g.split["train"]], out_n[sub_g.split["train"]])

            train_metric = {}
            train_metric["micro"] = micro_train
            train_metric["macro"] = macro_train
            train_metric["auroc"] = roc_auc_train
            train_metric["ap"] = ap_train
            train_metric["loss"] = float(loss)

        return train_metric

    @torch.no_grad()
    def eva_pre_tasks(self, sub_g, target_classes_groups, t):

        # test on previous task: sub_g is the corresponding subgraph
        # t is the time step for the previous task
        # get the output
        self.backbone.eval()
        out = self.backbone(sub_g.x, sub_g.edge_index)

        offset1 = 0
        for i in range(t+1):
            offset1 = offset1 + len(target_classes_groups[i])

        # offset2 = len(target_classes_groups[t])
        # print(offset1)
        # print(offset2)

        # out = out[:, offset1:offset1 + offset2]
        out = out[:, :offset1]
        print(out.shape)
        y = sub_g.y
        print(y.shape)

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

        return val_metric, test_metric, out
