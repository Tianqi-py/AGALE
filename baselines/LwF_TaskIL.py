# task incremental LwF using GCN as backbone model
import os
import sys
# Add the parent directory of mypackage to the Python path
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))
from data.CGL_DataLoader import *

import torch.optim as optim
from models.LwF_model import LwF_Task_IL
from args import parse_args
from earlystopping import EarlyStopping
from backbones.backbones import GCN, GAT
from utils import prepare_sub_graph


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
        elif args.data_name == "CoraFull":
            G = import_cora(args.shuffle_idx)

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
        print("classifying among classes: ", sub_g.target_classes)
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
            # cant know how many classes in total, so read the target clases out now
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

            print("initializing model...")
            model = LwF_Task_IL(backbone=backbone,
                                args=args)
            print(model)
            print("start training...")
            optimizer = optim.Adam(model.parameters(), lr=args.lr, weight_decay=args.weight_decay)
            early_stopping = EarlyStopping(model_name=args.model_name,
                                           setting=args.setting,
                                           data_name=args.data_name, split_name=args.shuffle_idx,
                                           patience=args.patience, verbose=True)

            for epoch in range(1, args.epochs):
                train_metric = model.standard_train(sub_g, optimizer)
                train_losses.append(train_metric["loss"])
                # test on current task
                val_metric, test_metric, _ = model.standard_test(sub_g)

                if args.multi_class:
                    print(f'Epoch: {epoch:03d}, Loss: {train_metric["loss"]:.5f}, '
                          f'loss val: {val_metric["loss"]:.5f}, '
                          f'Train acc: {train_metric["acc"]:.4f}, '
                          f'Val acc: {val_metric["acc"]:.4f}, '
                          f'Test acc: {test_metric["acc"]:.4f}, '
                          )
                else:
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
            val_metric, test_metric, _ = model.standard_test(sub_g)

            if args.multi_class:
                print(
                    f'Val acc: {val_metric["acc"]:.4f}, '
                    f'Test acc: {test_metric["acc"]:.4f}, '
                )
            else:
                print(
                      #f'Val micro: {val_metric["micro"]:.4f}, Val macro: {val_metric["macro"]:.4f}, '
                      f'Test micro: {test_metric["micro"]:.4f}, Test macro: {test_metric["macro"]:.4f}, '
                      #f'val ROC-AUC macro: {val_metric["auroc"]:.4f}, '
                      f'test ROC-AUC macro: {test_metric["auroc"]:.4f}, '
                      #f'Val Average Precision Score: {val_metric["ap"]:.4f}, '
                      f'Test Average Precision Score: {test_metric["ap"]:.4f}, '
                      )
            print("###############################################################")

        # the time steps i > 0
        else:

            # load the previous model and test make predictions for the new data about the old task
            print("###############################################################")
            # load previous model
            # initialize
            if args.backbone == "GCN":
                backbone = GCN(in_channels=sub_g.x.shape[1],
                               hidden_channels=args.hidden,
                               out_channels=model.backbone.conv2.out_channels,
                               args=args)

            elif args.backbone == "GAT":
                backbone = GAT(in_channels=sub_g.x.shape[1],
                               hidden_channels=args.hidden,
                               out_channels=model.backbone.conv2.out_channels,
                               heads=3,
                               args=args)
            else:
                # if not specialised, use GCN as backbone
                backbone = GCN(in_channels=sub_g.x.shape[1],
                               hidden_channels=args.hidden,
                               out_channels=model.backbone.conv2.out_channels,
                               args=args)

            previous_model = LwF_Task_IL(backbone,
                                         args=args)

            previous_model.load_state_dict(torch.load(args.model_name + "_" + args.setting + "_" + args.data_name + "_"
                                             + args.shuffle_idx + '_checkpoint.pt'))
            print("previous model loaded:")
            print(previous_model)
            # predictions regarding the new data for old tasks made by previous model

            sub_g.y = sub_g.y[:, target_classes_groups[-1]]
            print("classes in new task: ", sub_g.y.shape[1])
            # # warm up: train on the new data
            # # Freeze the existing output units and add new units for the new classes
            # model.freeze_para()
            model.add_new_outputs(num_new_classes=len(sub_g.target_classes))
            print("output channels after addition:", model.backbone.conv2.out_channels)
            #
            # # Update the new output units to minimize the loss on the new task
            # optimizer = optim.Adam(filter(lambda p: p.requires_grad, model.parameters()), lr=args.lr,
            #                        weight_decay=args.weight_decay)
            # early_stopping = EarlyStopping(model_name=args.model_name,
            #                                setting=args.setting,
            #                                data_name=args.data_name, split_name=args.shuffle_idx,
            #                                patience=args.patience, verbose=True)
            #
            # # warm-up: train on the new task
            # for epoch in range(1, args.epochs):
            #     train_metric = model.warm_up(sub_g, optimizer)
            #     train_losses.append(train_metric["loss"])
            #     # test on current task
            #     val_metric, test_metric, _ = model.standard_test(sub_g)
            #     print(f'Epoch: {epoch:03d}, Loss: {train_metric["loss"]:.5f}, '
            #           f'loss val: {val_metric["loss"]:.5f} '
            #           f'Train micro: {train_metric["micro"]:.4f}, Train macro: {train_metric["macro"]:.4f}, '
            #           f'Val micro: {val_metric["micro"]:.4f}, Val macro: {val_metric["macro"]:.4f}, '
            #           f'Test micro: {test_metric["micro"]:.4f}, Test macro: {test_metric["macro"]:.4f}, '
            #           f'train ROC-AUC macro: {train_metric["auroc"]:.4f} '
            #           f'val ROC-AUC macro: {val_metric["auroc"]:.4f}, '
            #           f'test ROC-AUC macro: {test_metric["auroc"]:.4f}, '
            #           f'Train Average Precision Score: {train_metric["ap"]:.4f}, '
            #           f'Val Average Precision Score: {val_metric["ap"]:.4f}, '
            #           f'Test Average Precision Score: {test_metric["ap"]:.4f}, '
            #           )
            #     early_stopping(val_metric["loss"], model)
            #     if early_stopping.early_stop:
            #         print("Early stopping")
            #         break
            #
            # print("warm-up phase Finished!")
            print("start joint-optimization")
            # joint optimization
            # optimize on all parameters
            #model.unfreeze_para()
            optimizer = optim.Adam(model.parameters(), lr=args.lr, weight_decay=args.weight_decay)
            early_stopping = EarlyStopping(model_name=args.model_name,
                                           setting=args.setting,
                                           data_name=args.data_name, split_name=args.shuffle_idx,
                                           patience=args.patience, verbose=True)
            for epoch in range(1, args.epochs):
                # joint optimization

                print("previous model", previous_model.backbone.conv2.out_channels)

                train_metric = model.joint_opt(previous_model, sub_g, optimizer, t, target_classes_groups)
                train_losses.append(train_metric["loss"])

                # test on current task
                val_metric, test_metric, _ = model.standard_test(sub_g)

                if args.multi_class:
                    print(f'Epoch: {epoch:03d}, Loss: {train_metric["loss"]:.5f}, '
                          f'loss val: {val_metric["loss"]:.5f}, '
                          f'Train acc: {train_metric["acc"]:.4f}, '
                          f'Val acc: {val_metric["acc"]:.4f}, '
                          f'Test acc: {test_metric["acc"]:.4f}, '
                          )
                else:
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
                # print(model)
                # print(model.backbone.conv2.bias.shape)
                # print(model.backbone.conv2.lin.weight.shape)
                if early_stopping.early_stop:
                    print("Early stopping")
                    break
            print("Optimization Finished!")
            # load the last checkpoint with the best model
            model.load_state_dict(torch.load(args.model_name + "_" + args.setting + "_" + args.data_name + "_"
                                             + args.shuffle_idx + '_checkpoint.pt'))

            # current time step t, test previous the tasks
            for j, key in enumerate(list(G.groups.keys())[:t+1]):
                # test on time step j
                sub_g = prepare_sub_graph(G, key, args.Cross_Task_Message_Passing)
                print("***************************************")
                sub_g.y = sub_g.y[:, target_classes_groups[j]]
                print(target_classes_groups[j])
                print("***************************************")

                print("current Timestep:", t)
                print("test on group: ", j, ", target classes: ", sub_g.target_classes)
                val_metric, test_metric, _ = model.eva_pre_tasks(sub_g, target_classes_groups, j)
                metric[t][j] = test_metric


                if args.multi_class:
                    print(
                          f'Val acc: {val_metric["acc"]:.4f}, '
                          f'Test acc: {test_metric["acc"]:.4f}, '
                          )

                else:
                    print(f'Val micro: {val_metric["micro"]:.4f}, Val macro: {val_metric["macro"]:.4f}, '
                          f'Test micro: {test_metric["micro"]:.4f}, Test macro: {test_metric["macro"]:.4f}, '
                          f'val ROC-AUC macro: {val_metric["auroc"]:.4f}, '
                          f'test ROC-AUC macro: {test_metric["auroc"]:.4f}, '
                          f'Val Average Precision Score: {val_metric["ap"]:.4f}, '
                          f'Test Average Precision Score: {test_metric["ap"]:.4f}. '
                          )
            print("###############################################################")
