import torch
from torch_geometric.nn import GCNConv
from metric import BCE_loss, f1_Score, _eval_rocauc, ap_score
from utils import build_subgraph
from utils import get_ids_per_cls_train, map_subg_to_G
import torch.optim as optim
from torch_geometric_temporal.nn.recurrent import EvolveGCNO
import torch.nn.functional as F

class MAS(torch.nn.Module):
    def __init__(self, backbone, args, optimizer=None):
        super(MAS, self).__init__()
        self.reg = args.mas_mem_str

        # self.task_manager = task_manager

        # setup network
        self.backbone = backbone

        # setup optimizer
        if optimizer is None:
            print("initialize optimizer")
            self.optimizer = torch.optim.Adam(self.backbone.parameters(), lr=args.lr, weight_decay=args.weight_decay)
        else:
            self.optimizer = optimizer


        # setup memories
        self.current_task = 0
        self.optpar = []
        self.fisher = []
        self.n_seen_examples = 0
        self.mem_mask = None
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

    def ClassIL_train(self, sub_g, t, target_classes_groups, optimizer=None):

        # for t>0, optimize the shared paras and the paras for the new task
        # setup optimizer
        if optimizer is not None:
            self.optimizer = optimizer
        self.epochs += 1
        self.backbone.train()
        self.backbone.zero_grad()

        out = self.backbone(sub_g.x, sub_g.edge_index)
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
            for i, p in enumerate(self.backbone.conv1.parameters()):
                l = self.reg * self.fisher[i]
                l = l * (p - self.optpar[i]).pow(2)
                loss += l.sum()
            # the shape of the paras in the last layer changes
            offset1 = 0
            # the offset from of the last task
            for g in target_classes_groups[:-1]:
                offset1 += len(g)
            print("offset from last task:", offset1)

            for i, p in enumerate(self.backbone.conv2.parameters()):
                # the shared part with previous time step in the last layer
                p = p[:offset1]

                # the index in the fisher matrix is i+2 since first layer has two paras
                print("is the value for the first 2 unit get updated?")
                print(self.optpar[i+2].shape)
                #print(self.optpar[i + 2] - p)
                print("is the fisher value for the first 2 unit get updated?")

                #print(self.fisher[i + 2] - p)
                l = self.reg * self.fisher[i+2]

                print(self.optpar[i+2].shape)
                #print(l.shape)
                #print(self.fisher)
                print(self.fisher[i+2].shape)

                print(p.shape)

                l = l * (p - self.optpar[i+2]).pow(2)
                #print("the value of l")
                #print(l)
                loss += l.sum()

        loss.backward()
        self.optimizer.step()
        train_metric["loss"] = loss

        return train_metric

    def TaskIL_train(self, sub_g, t, target_classes_groups, optimizer=None):

        # for t>0, optimize the shared paras and the paras for the new task
        # setup optimizer
        if optimizer is not None:
            self.optimizer = optimizer
        self.optimizer.zero_grad()

        self.backbone.train()
        self.backbone.zero_grad()

        # offset1, offset2 = self.task_manager.get_label_offset(t)
        out = self.backbone(sub_g.x, sub_g.edge_index)
        num_target_class = len(sub_g.target_classes)
        print(len(sub_g.target_classes))
        print(num_target_class)
        out = out[:, -num_target_class:]
        y = sub_g.y
        print(out.shape)
        print(y.shape)

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
            for i, p in enumerate(self.backbone.conv1.parameters()):
                l = self.reg * self.fisher[i]
                l = l * (p - self.optpar[i]).pow(2)
                loss += l.sum()
            # the shape of the paras in the last layer changes
            offset1 = 0
            # the offset from of the last task
            for g in target_classes_groups[:-1]:
                offset1 += len(g)
            print("offset from last task:", offset1)

            for i, p in enumerate(self.backbone.conv2.parameters()):
                # the shared part with previous time step in the last layer
                p = p[:offset1]

                # the index in the fisher matrix is i+2 since first layer has two paras
                print("is the value for the first 2 unit get updated?")
                print(self.optpar[i+2].shape)
                #print(self.optpar[i + 2] - p)
                print("is the fisher value for the first 2 unit get updated?")

                #print(self.fisher[i + 2] - p)
                l = self.reg * self.fisher[i+2]

                print(self.optpar[i+2].shape)
                #print(l.shape)
                #print(self.fisher)
                print(self.fisher[i+2].shape)

                print(p.shape)

                l = l * (p - self.optpar[i+2]).pow(2)
                #print("the value of l")
                #print(l)
                loss += l.sum()

        loss.backward()
        self.optimizer.step()
        train_metric["loss"] = loss

        return train_metric

    def last_epoch(self, t, sub_g):
        self.optpar = []
        old_fisher = self.fisher
        self.fisher = []
        new_fisher = []

        # at the end of each task, calculate fisher matrix
        n_new_examples = len(sub_g.split["train"])
        print(n_new_examples)
        out = self.backbone(sub_g.x, sub_g.edge_index)
        out = out[sub_g.split["train"], :]

        out.pow_(2)
        loss = out.mean()
        self.backbone.zero_grad()
        loss.backward()

        for p in self.backbone.parameters():
            pd = p.data.clone()
            pg = p.grad.data.clone().pow(2)
            new_fisher.append(pg)
            self.optpar.append(pd)
        print("##############new fisher######################################")
        print(new_fisher[2].shape)
        print("##############new fisher######################################")
        if len(self.fisher) != 0:
            for i, f in enumerate(new_fisher):
                # the second layer paras got extended for the new task
                if i >= 2:
                    # the para dim in the second layer is extended
                    of = old_fisher[i].size(dim=0)
                    # the shared para with old task
                    shared_para = (old_fisher[i] * self.n_seen_examples + new_fisher[i][:of] * n_new_examples) \
                                        / (self.n_seen_examples + n_new_examples)
                    self.fisher.append(shared_para)
                    self.fisher[i].append(new_fisher[of:])
                # the para dim in the first layer does not change
                else:
                    self.fisher[i] = (self.fisher[i] * self.n_seen_examples + new_fisher[i] * n_new_examples) / (
                                      self.n_seen_examples + n_new_examples)

            self.n_seen_examples += n_new_examples
            print("##############self fisher######################################")
            print(self.fisher[0].shape)
            print(self.fisher[2].shape)
            print("##############self fisher######################################")
        else:
            for i, f in enumerate(new_fisher):
                self.fisher.append(new_fisher[i])
            self.n_seen_examples = n_new_examples

        self.current_task = t

        return self.fisher, self.optpar

    def standard_test(self, sub_g, pre_eval=False, TaskIL=False, t=0, target_classes_groups=None):

        self.backbone.eval()
        out = self.backbone(sub_g.x, sub_g.edge_index)
        y = sub_g.y

        # test on previous task
        if pre_eval:
            # TaskIL
            if TaskIL:
                # get the offsets of the target task
                num_cls_in_tsk = len(target_classes_groups[t])
                off = 0
                for tt in range(t):
                    off += len(target_classes_groups[tt])
                out = out[:, off:off+num_cls_in_tsk]

            # ClassIL
            else:
                out = out[:, :y.shape[1]]
                print("test on previous task, the shape of true labels in the previous task")

        # test on the current task
        else:
            if TaskIL:
                num_cls = len(target_classes_groups[-1])
                out = out[:, -num_cls:]
            # do nothing for the ClassIL

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
