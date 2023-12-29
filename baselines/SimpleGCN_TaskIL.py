# task incremental SimpleGCN
import os
import sys
# Add the parent directory of mypackage to the Python path
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))
from data.CGL_DataLoader import *

import torch
import numpy as np
import torch.optim as optim
from args import parse_args
from earlystopping import EarlyStopping
from backbones.backbones import GCN, GAT
from utils import prepare_sub_graph, flatten
from metric import BCE_loss, f1_Score, _eval_rocauc, ap_score

if __name__ == "__main__":
    args = parse_args()
    # load datasets
    try:
        if args.data_name == "blogcatalog":
            G = load_mat_data(args.shuffle_idx)
        elif args.data_name == "Hyperspheres_10_10_0":
            G = load_hyper_data(args.shuffle_idx)
        elif args.data_name == "yelp":
            G = import_yelp(args.shuffle_idx)
        elif args.data_name == "dblp":
            G = import_dblp(args.shuffle_idx)
        elif args.data_name == "pcg_removed_isolated_nodes":
            G = import_pcg(args.shuffle_idx)
        elif args.data_name == "humloc":
            G = import_humloc(args.shuffle_idx)
        elif args.data_name == "eukloc":
            G = import_eukloc(args.shuffle_idx)
    except os.error:
        sys.exit("Can't find the data.")
    # get number of tasks, each for one time step

    time_steps = len(G.groups.keys())

    # initialization
    metric = dict.fromkeys(np.arange(time_steps))
    # target classes in each task
    target_classes_groups = []
    num_class_per_task = []
    train_losses = []
    off_sets_l = []

    # one group arrives at one time step
    for t, key in enumerate(G.groups.keys()):

        print(f'Training for Timestep: {t: 03d}')
        metric[t] = {}

        # prepare subgraph for the task
        sub_g = prepare_sub_graph(G, key, args.Cross_Task_Message_Passing)
        print("New classes in the subgraph: ", sub_g.target_classes)
        print("subgraph size: ", sub_g.x.shape[0])
        print("subgraph edges: ", int(sub_g.edge_index.shape[1]/2))
        target_classes_groups.append(sub_g.target_classes)
        # sub_g.target_classes is a list containing the class indices for the task
        num_class_per_task.append(len(sub_g.target_classes))
        off_sets_l.append(len(sub_g.target_classes))

        # the first time step
        if t == 0:

            # one hot encode with the classes seen in this group
            # only classify among the classes in this group
            sub_g.y = sub_g.y[:, target_classes_groups[0]]

            # initialize the backbone model
            print("intialize backbone model...")
            if args.backbone == "GCN":
                backbone = GCN(in_channels=sub_g.x.shape[1],
                               hidden_channels=args.hidden,
                               out_channels=sub_g.y.shape[1],
                               args=args)

            elif args.backbone == "GAT":
                backbone = GAT(in_channels=sub_g.x.shape[1],
                               hidden_channels=args.hidden,
                               out_channels=sub_g.y.shape[1],
                               heads=3,
                               args=args)
            else:
                # if not specialised, use GCN as backbone
                backbone = GCN(in_channels=sub_g.x.shape[1],
                               hidden_channels=args.hidden,
                               out_channels=sub_g.y.shape[1],
                               args=args)


            print("start training...")
            print("initializing model...")
            model = backbone

            print(model)

            early_stopping = EarlyStopping(model_name=args.model_name,
                                           setting=args.setting,
                                           data_name=args.data_name, split_name=args.shuffle_idx,
                                           patience=args.patience, verbose=True)

            optimizer = optim.Adam(model.parameters(), lr=args.lr, weight_decay=args.weight_decay)

            for epoch in range(1, args.epochs):
                model.train()
                optimizer.zero_grad()

                out = model.forward(sub_g.x, sub_g.edge_index)
                y = sub_g.y

                # evaluation on current task
                loss_train = BCE_loss(out[sub_g.split["train"]], y[sub_g.split["train"]])
                micro_train, macro_train = f1_Score(y[sub_g.split["train"]], out[sub_g.split["train"]])
                roc_auc_train = _eval_rocauc(y[sub_g.split["train"]], out[sub_g.split["train"]])
                ap_train = ap_score(y[sub_g.split["train"]], out[sub_g.split["train"]])

                loss_train.backward()
                optimizer.step()

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
                train_metric["loss"] = loss_train

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

                train_losses.append(train_metric["loss"])

                print(f'Epoch: {epoch:03d}, Loss: {train_metric["loss"]:.5f}, '
                      f'loss val: {val_metric["loss"]:.5f} '
                      f'Train micro: {train_metric["micro"]:.4f}, Train macro: {train_metric["macro"]:.4f}, '
                      f'Val micro: {val_metric["micro"]:.4f}, Val macro: {val_metric["macro"]:.4f}, '
                      f'Test micro: {test_metric["micro"]:.4f}, Test macro: {test_metric["macro"]:.4f}, '
                      f'train ROC-AUC macro: {train_metric["auroc"]:.4f} '
                      f'val ROC-AUC macro: {val_metric["auroc"]:.4f}, '
                      f'test ROC-AUC macro: {test_metric["auroc"]:.4f}, '
                      f'Train Average Precision Score: {train_metric["ap"]:.4f}, '
                      f'Val Average Precision Score: {val_metric["ap"]:.4f}, '
                      f'Test Average Precision Score: {test_metric["ap"]:.4f}, '
                      )
                early_stopping(val_metric["loss"], model)
                if early_stopping.early_stop:
                    print("Early stopping")
                    break

            print("Optimization Finished!")
            # load the last checkpoint with the best model
            model.load_state_dict(torch.load(args.model_name + "_" + args.setting + "_" + args.data_name + "_"
                                             + args.shuffle_idx + '_checkpoint.pt'))
            # test the model on time step 0
            out = model.forward(sub_g.x, sub_g.edge_index)
            y = sub_g.y

            micro_test, macro_test = f1_Score(y[sub_g.split["test"]], out[sub_g.split["test"]])
            roc_auc_test = _eval_rocauc(y[sub_g.split["test"]], out[sub_g.split["test"]])
            ap_test = ap_score(y[sub_g.split["test"]], out[sub_g.split["test"]])

            test_metric = {}
            test_metric["micro"] = micro_test
            test_metric["macro"] = macro_test
            test_metric["auroc"] = roc_auc_test
            test_metric["ap"] = ap_test

            print(
                  f'Test micro: {test_metric["micro"]:.4f}, Test macro: {test_metric["macro"]:.4f}, '
                  f'test ROC-AUC macro: {test_metric["auroc"]:.4f}, '
                  f'Test Average Precision Score: {test_metric["ap"]:.4f}, '
                  )
            print("###############################################################")

        # the time steps i > 0
        else:

            print("target classes for current task", target_classes_groups[t])
            sub_g.y = sub_g.y = sub_g.y[:, target_classes_groups[-1]]

            # sub_g.target_classes is the group of new classes in the current time step
            # generated in utils.py
            print(model)
            print([p.shape for p in model.parameters()])
            print([p for p in model.parameters()])
            model.add_new_outputs(num_new_classes=len(sub_g.target_classes))
            print(model)
            print([p.shape for p in model.parameters()])
            print([p for p in model.parameters()])
            # Update the new output units to minimize the loss on the new task
            early_stopping = EarlyStopping(model_name=args.model_name,
                                           setting=args.setting,
                                           data_name=args.data_name, split_name=args.shuffle_idx,
                                           patience=args.patience, verbose=True)

            for epoch in range(1, args.epochs):

                model.train()
                optimizer.zero_grad()
                out = model.forward(sub_g.x, sub_g.edge_index)
                num_classes = sub_g.y.shape[1]
                out = out[:, -num_classes:]

                y = sub_g.y

                # evaluation on current task
                loss_train = BCE_loss(out[sub_g.split["train"]], y[sub_g.split["train"]])
                micro_train, macro_train = f1_Score(y[sub_g.split["train"]], out[sub_g.split["train"]])
                roc_auc_train = _eval_rocauc(y[sub_g.split["train"]], out[sub_g.split["train"]])
                ap_train = ap_score(y[sub_g.split["train"]], out[sub_g.split["train"]])

                loss_train.backward()
                optimizer.step()

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
                train_metric["loss"] = loss_train

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
                train_losses.append(train_metric["loss"])

                print(f'Epoch: {epoch:03d}, Loss: {train_metric["loss"]:.10f}, '
                      f'loss val: {val_metric["loss"]:.10f} '
                      f'Train micro: {train_metric["micro"]:.4f}, Train macro: {train_metric["macro"]:.4f}, '
                      f'Val micro: {val_metric["micro"]:.4f}, Val macro: {val_metric["macro"]:.4f}, '
                      f'Test micro: {test_metric["micro"]:.4f}, Test macro: {test_metric["macro"]:.4f}, '
                      f'train ROC-AUC macro: {train_metric["auroc"]:.4f} '
                      f'val ROC-AUC macro: {val_metric["auroc"]:.4f}, '
                      f'test ROC-AUC macro: {test_metric["auroc"]:.4f}, '
                      f'Train Average Precision Score: {train_metric["ap"]:.4f}, '
                      f'Val Average Precision Score: {val_metric["ap"]:.4f}, '
                      f'Test Average Precision Score: {test_metric["ap"]:.4f}, '
                      )
                early_stopping(val_metric["loss"], model)
                if early_stopping.early_stop:
                    print("Early stopping")
                    break


            print("Optimization Finished!")
            # load the last checkpoint with the best model
            model.load_state_dict(torch.load(args.model_name + "_" + args.setting + "_" + args.data_name + "_"
                                             + args.shuffle_idx + '_checkpoint.pt'))

            # current time step t, test previous the tasks
            for j, key in enumerate(list(G.splits.keys())[:t + 1]):
                # test on time step j
                sub_g = prepare_sub_graph(G, key, args.Cross_Task_Message_Passing)
                print("***************************************")
                sub_g.y = sub_g.y[:, target_classes_groups[j]]
                print(target_classes_groups[j])
                print("***************************************")

                print("current Timestep:", t)
                print("test on group: ", j, ", target classes: ", sub_g.target_classes)

                out = model.forward(sub_g.x, sub_g.edge_index)
                num_classes = sub_g.y.shape[1]
                off1 = 0
                for k in range(j):
                    off1 = off1 + len(target_classes_groups[k])
                off2 = off1 + len(target_classes_groups[j])
                print("test on t", t, ": off1 and off2", off1, off2)
                out = out[:, off1:off2]
                y = sub_g.y

                micro_test, macro_test = f1_Score(y[sub_g.split["test"]], out[sub_g.split["test"]])
                roc_auc_test = _eval_rocauc(y[sub_g.split["test"]], out[sub_g.split["test"]])
                ap_test = ap_score(y[sub_g.split["test"]], out[sub_g.split["test"]])

                test_metric = {}
                test_metric["micro"] = micro_test
                test_metric["macro"] = macro_test
                test_metric["auroc"] = roc_auc_test
                test_metric["ap"] = ap_test

                metric[t][j] = test_metric

                print(f'Val micro: {val_metric["micro"]:.4f}, Val macro: {val_metric["macro"]:.4f}, '
                      f'Test micro: {test_metric["micro"]:.4f}, Test macro: {test_metric["macro"]:.4f}, '
                      f'val ROC-AUC macro: {val_metric["auroc"]:.4f}, '
                      f'test ROC-AUC macro: {test_metric["auroc"]:.4f}, '
                      f'Val Average Precision Score: {val_metric["ap"]:.4f}, '
                      f'Test Average Precision Score: {test_metric["ap"]:.4f}. '
                      )
            print("###############################################################")