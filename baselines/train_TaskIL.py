import torch
import numpy as np
from args import parse_args
import torch.optim as optim
from baseline_models import GCN, GAT
from earlystopping import EarlyStopping
from metric import f1_loss, BCE_loss, ap_score, _eval_rocauc
from CGL_DataLoader import load_mat_data, load_hyper_data, import_yelp
from utils import prepare_sub_graph
from LWF_CGLB_Github import NET
import seaborn as sns


def model_train(model, sub_g):
    # subgraph for a task
    model.train()
    optimizer.zero_grad()
    out = model(sub_g.x, sub_g.edge_index)

    # only evaluating on the target classes
    out = out[:, sub_g.target_classes]
    y = sub_g.y[:, sub_g.target_classes]

    loss = BCE_loss(out[sub_g.split["train"]], y[sub_g.split["train"]])
    micro_train, macro_train = f1_loss(y[sub_g.split["train"]], out[sub_g.split["train"]])
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
def model_test(model, sub_g):
    model.eval()
    out = model(sub_g.x, sub_g.edge_index)

    out = out[:, sub_g.target_classes]
    y = sub_g.y[:, sub_g.target_classes]

    loss_val = BCE_loss(out[sub_g.split["val"]], y[sub_g.split["val"]])
    micro_val, macro_val = f1_loss(y[sub_g.split["val"]], out[sub_g.split["val"]])
    roc_auc_val = _eval_rocauc(y[sub_g.split["val"]], out[sub_g.split["val"]])
    ap_val = ap_score(y[sub_g.split["val"]], out[sub_g.split["val"]])

    micro_test, macro_test = f1_loss(y[sub_g.split["test"]], out[sub_g.split["test"]])
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


if __name__ == "__main__":

    args = parse_args()
    if args.data_name == "blogcatalog":
        G = load_mat_data(data_name=args.data_name, path="../../code/data/")
    elif args.data_name == "Hyperspheres_10_10_0":
        G = load_hyper_data(data_name=args.data_name, path="../../code/data")
    elif args.data_name == "yelp":
        G = import_yelp(data_name=args.data_name, )
    else:
        print("datasets not found")

    backbone = GCN(in_channels=G.x.shape[1],
                    hidden_channels=256,
                    out_channels=G.y.shape[1],
                   )

    model = NET(model=backbone,
                task_manager=G.groups,
                args=args)

    optimizer = optim.Adam(model.parameters(), lr=args.lr, weight_decay=args.weight_decay)
    time_steps = len(G.groups.keys())

    metric = dict.fromkeys(np.arange(time_steps))

    # target classes in each task
    target_classes_groups = []
    train_losses = []
    # one group arrives at one time step
    for i, key in enumerate(G.groups.keys()):

        print(f'Training for Timestep: {i: 03d}')
        early_stopping = EarlyStopping(patience=args.patience, verbose=True)
        metric[i] = {}

        # prepare subgraph for the
        sub_g = prepare_sub_graph(G, key, args.Cross_Task_Message_Passing)
        print("classifying among classes: ", sub_g.target_classes)
        target_classes_groups.append(sub_g.target_classes)
        # print(list(flatten(key)))
        # # target classes of current task
        # target_classes = list(flatten(key))
        #
        # print(f'Training for Timestep: {i: 03d}')
        # print("classifying among classes: ", target_classes)
        #
        # # nodes in the group
        # node_ids = G.groups[key].int().long()
        #
        # # only keep the edges among the nodes that are in the group
        # #edge_index_complete, _ = subgraph(torch.tensor(node_ids).long(), G.edge_index, None)
        #
        # # also include the edges that connect the nodes in the subgraph with those that are not in the subgraph
        # node_mask = mask.index_to_mask(node_ids, size=G.num_nodes)
        #
        # # or operation of two boolean lists
        # edge_mask = node_mask[G.edge_index[0]] + node_mask[G.edge_index[1]]
        # edge_index_complete = G.edge_index[:, edge_mask]
        #
        # # all nodes in the subgraph: including target nodes and their neighbors
        # node_ids_subgraph = torch.unique(edge_index_complete.flatten()).long()
        #
        # # index target nodes in the subgraph nodes
        # target_ids_sub = [i for i, n in enumerate(node_ids_subgraph) if n in node_ids]
        #
        # # add the ids of the neighbors to the node_ids
        # # get the edge_index with the indices recalculate for the whole subgraph
        # edge_index = map_edge_index(node_ids_subgraph, edge_index_complete)
        #
        # features = G.x[node_ids_subgraph]
        # labels = G.y[node_ids_subgraph]
        # # map the ids to subgraph
        # split = map_split(node_ids_subgraph, G.splits[key])

        # train the model for this task

        for epoch in range(1, 2000):

            train_metric = model.train_task_IL(model, sub_g, t=i)
            train_losses.append(train_metric["loss"])
            # test on current task
            val_metric, test_metric = model_test(model, sub_g)
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
        model.load_state_dict(torch.load('checkpoint.pt'))

        print("###############################################################")
        # test all the tasks
        for j, key in enumerate(G.groups.keys()):
            sub_g = prepare_sub_graph(G, key, args.Cross_Task_Message_Passing)
            print("current Timestep:", i)
            print("test on group: ", j, ", target classes: ", sub_g.target_classes)
            val_metric, test_metric = model_test(model, sub_g)
            metric[i][j] = test_metric

            print(f'Val micro: {val_metric["micro"]:.4f}, Val macro: {val_metric["macro"]:.4f}, '
                  f'Test micro: {test_metric["micro"]:.4f}, Test macro: {test_metric["macro"]:.4f}, '
                  f'val ROC-AUC macro: {val_metric["auroc"]:.4f}, '
                  f'test ROC-AUC macro: {test_metric["auroc"]:.4f}, '
                  f'Val Average Precision Score: {val_metric["ap"]:.4f}, '
                  f'Test Average Precision Score: {test_metric["ap"]:.4f}. '
                )
        print("###############################################################")

    # visualize the losses
    train_losses = np.asarray(train_losses)
    x = np.arange(1, len(train_losses)+1)
    sns_plot = sns.lineplot(x=x, y=train_losses)
    fig = sns_plot.get_figure()
    fig.savefig("train_losses.png")









