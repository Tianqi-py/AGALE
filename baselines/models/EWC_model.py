import torch
from torch.nn import GRU
from torch_geometric.nn import GCNConv
from metric import BCE_loss, f1_Score, _eval_rocauc, ap_score
from utils import build_subgraph
from utils import get_ids_per_cls_train, map_subg_to_G
import torch.optim as optim
from torch_geometric_temporal.nn.recurrent import EvolveGCNO
import torch.nn.functional as F

class EWC_Task_IL(torch.nn.Module):
    def __init__(self, backbone, reg):
        super(EWC_Task_IL, self).__init__()
        self.backbone = backbone
        self.fisher = {}
        self.optpar = {}
        # balance between the old and new task
        self.reg = reg
        self.epochs = 0

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

    def cal_fisher(self, t):
        # at the end of each task, calculate fisher matrix
        self.fisher[t] = []
        self.optpar[t] = []
        print("$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$")
        for p in self.backbone.parameters():
            pd = p.data.clone()
            print(pd.shape)
            pg = p.grad.data.clone().pow(2)
            print(pg.shape)
            self.fisher[t].append(pg)
            self.optpar[t].append(pd)

        # print(self.fisher[0])
        # print(self.optpar[0])
        print("$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$")
        return self.fisher, self.optpar

    def standard_train(self, sub_g, optimizer, t, target_classes_groups):

        self.epochs += 1

        self.backbone.train()
        optimizer.zero_grad()

        out = self.backbone(sub_g.x, sub_g.edge_index)
        # take the output dimensions for the new task
        num_target_class = len(sub_g.target_classes)
        out = out[:, -num_target_class:]
        y = sub_g.y

        # loss on the new task
        loss = BCE_loss(out[sub_g.split["train"]], y[sub_g.split["train"]])
        micro_train, macro_train = f1_Score(y[sub_g.split["train"]], out[sub_g.split["train"]])
        roc_auc_train = _eval_rocauc(y[sub_g.split["train"]], out[sub_g.split["train"]])
        ap_train = ap_score(y[sub_g.split["train"]], out[sub_g.split["train"]])

        train_metric = {}
        train_metric["micro"] = micro_train
        train_metric["macro"] = macro_train
        train_metric["auroc"] = roc_auc_train
        train_metric["ap"] = ap_train

        if t > 0:
            # offset of the label in the last layer
            offset1 = 0
            # there is at least one old task
            for tt in range(t):
                print("%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%")
                print(t)
                print(tt)
                # the value of the paras in the first layer parameters are updated
                # the shape of the paras in the first layer parameters stays the same
                for i, p in enumerate(self.backbone.conv1.parameters()):
                    l = self.reg * self.fisher[tt][i]
                    l = l * (p - self.optpar[tt][i]).pow(2)
                loss += l.sum()
                # the value of the paras in the second layer parameters are updated
                # the shape of the paras in the second layer parameters are also extended
                num_c = len(target_classes_groups[tt])
                offset1 += num_c

                print(offset1)
                for i, p in enumerate(self.backbone.conv2.parameters()):
                    # the shared part with previous time step in the last layer
                    p = p[:offset1]
                    # the index in the fisher matrix is i+2 since first layer has two paras
                    # print("is the value for the first 4 unit get updated?")
                    # print(self.optpar[tt][i+2].shape)
                    # print(self.optpar[tt][i+2]-p)
                    # print("is the fisher value for the first 4 unit get updated?")
                    # print(self.fisher[tt][i+2]-p)
                    l = self.reg * self.fisher[tt][i+2]
                    # print(self.optpar[tt][i+2].shape)
                    l = l * (p - self.optpar[tt][i+2]).pow(2)
                    # print("the value of l")
                    # print(l)
                    loss += l.sum()
                print("%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%")


        loss.backward()
        optimizer.step()
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


class EWC_Class_IL(torch.nn.Module):
    def __init__(self, backbone, reg):
        super(EWC_Class_IL, self).__init__()
        self.backbone = backbone
        self.fisher = {}
        self.optpar = {}
        # balance between the old and new task
        self.reg = reg
        self.epochs = 0

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

    def cal_fisher(self, t):
        # at the end of each task, calculate fisher matrix
        self.fisher[t] = []
        self.optpar[t] = []
        for p in self.backbone.parameters():
            pd = p.data.clone()
            pg = p.grad.data.clone().pow(2)
            self.fisher[t].append(pg)
            self.optpar[t].append(pd)
        return self.fisher, self.optpar

    def standard_train(self, sub_g, optimizer, t, target_classes_groups):
        self.epochs += 1
        self.backbone.train()
        optimizer.zero_grad()

        out = self.backbone(sub_g.x, sub_g.edge_index)
        # print("&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&")
        # print("check the output dimensions")
        # print(out.shape)
        y = sub_g.y
        # print(y.shape)
        # print("&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&")

        # loss on the new task
        loss = BCE_loss(out[sub_g.split["train"]], y[sub_g.split["train"]])
        micro_train, macro_train = f1_Score(y[sub_g.split["train"]], out[sub_g.split["train"]])
        roc_auc_train = _eval_rocauc(y[sub_g.split["train"]], out[sub_g.split["train"]])
        ap_train = ap_score(y[sub_g.split["train"]], out[sub_g.split["train"]])

        train_metric = {}
        train_metric["micro"] = micro_train
        train_metric["macro"] = macro_train
        train_metric["auroc"] = roc_auc_train
        train_metric["ap"] = ap_train

        if t > 0:
            # offset of the label in the last layer
            offset1 = 0
            # there is at least one old task
            for tt in range(t):
                print("%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%")
                #print(t)
                #print(tt)
                # the value of the paras in the first layer parameters are updated
                # the shape of the paras in the first layer parameters stays the same
                for i, p in enumerate(self.backbone.conv1.parameters()):
                    l = self.reg * self.fisher[tt][i]
                    l = l * (p - self.optpar[tt][i]).pow(2)
                    loss += l.sum()
                # the value of the paras in the second layer parameters are updated
                # the shape of the paras in the second layer parameters are also extended
                num_c = len(target_classes_groups[tt])
                offset1 += num_c
                print(offset1)
                for i, p in enumerate(self.backbone.conv2.parameters()):
                    # the shared part with previous time step in the last layer
                    p = p[:offset1]
                    # the index in the fisher matrix is i+2 since first layer has two paras
                    # print("is the value for the first 4 unit get updated?")
                    # print(self.optpar[tt][i+2].shape)
                    # #print(self.optpar[tt][i+2]-p)
                    # print("is the fisher value for the first 4 unit get updated?")
                    # #print(self.fisher[tt][i+2]-p)
                    l = self.reg * self.fisher[tt][i+2]
                    # print(self.optpar[tt][i+2].shape)
                    l = l * (p - self.optpar[tt][i+2]).pow(2)
                    #print("the value of l")
                    #print(l)
                    loss += l.sum()


        loss.backward()
        optimizer.step()
        train_metric["loss"] = float(loss)

        return train_metric

    @torch.no_grad()
    def standard_test(self, sub_g):
        # test on current task
        self.backbone.eval()
        out = self.backbone(sub_g.x, sub_g.edge_index)
        y = sub_g.y

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
        for i in range(t + 1):
            offset1 = offset1 + len(target_classes_groups[i])

        out = out[:, :offset1]
        print(out.shape)
        y = sub_g.y
        print(y.shape)

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