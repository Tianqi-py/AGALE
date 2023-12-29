import sys
import torch.optim as optim
from args import parse_args
from earlystopping import EarlyStopping
from backbones.backbones import GCN
import os
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))
from data.JointTrain_Loader import *
from metric import BCE_loss, f1_Score, _eval_rocauc, ap_score

if __name__ == "__main__":

    args = parse_args()
    # load datasets, only use the feature label and graph structure
    try:
        if args.data_name == "blogcatalog":
            G = load_mat_data(args.shuffle_idx)
        # elif args.data_name == "Hyperspheres_10_10_0":
        #     G = load_hyper_data(args.shuffle_idx)
        elif args.data_name == "yelp":
            G = import_yelp(args.shuffle_idx)
        elif args.data_name == "dblp":
            G = import_dblp(args.shuffle_idx)
        elif args.data_name == "pcg_removed_isolated_nodes":
            G = import_pcg(args.shuffle_idx)
        # elif args.data_name == "humloc":
        #     G = import_humloc(args.shuffle_idx)
        # elif args.data_name == "eukloc":
        #     G = import_eukloc(args.shuffle_idx)
            
    except os.error:
        sys.exit("Can't find the data.")

    model = GCN(in_channels=G.x.shape[1],
                hidden_channels=args.hidden,
                out_channels=G.y.shape[1])

    early_stopping = EarlyStopping(model_name=args.model_name,
                                   setting=args.setting,
                                   data_name=args.data_name, split_name=args.shuffle_idx,
                                   patience=args.patience, verbose=True)
    optimizer = optim.Adam(model.parameters(), lr=args.lr, weight_decay=args.weight_decay)

    for epoch in range(1, args.epochs):
        model.train()
        optimizer.zero_grad()

        out = model.forward(G.x, G.edge_index)
        y = G.y

        # evaluation on current task
        loss_train = BCE_loss(out[G.split["train"]], y[G.split["train"]])
        micro_train, macro_train = f1_Score(y[G.split["train"]], out[G.split["train"]])
        roc_auc_train = _eval_rocauc(y[G.split["train"]], out[G.split["train"]])
        ap_train = ap_score(y[G.split["train"]], out[G.split["train"]])

        loss_train.backward()
        optimizer.step()

        loss_val = BCE_loss(out[G.split["val"]], y[G.split["val"]])
        micro_val, macro_val = f1_Score(y[G.split["val"]], out[G.split["val"]])
        roc_auc_val = _eval_rocauc(y[G.split["val"]], out[G.split["val"]])
        ap_val = ap_score(y[G.split["val"]], out[G.split["val"]])

        micro_test, macro_test = f1_Score(y[G.split["test"]], out[G.split["test"]])
        roc_auc_test = _eval_rocauc(y[G.split["test"]], out[G.split["test"]])
        ap_test = ap_score(y[G.split["test"]], out[G.split["test"]])

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
    
    out = model.forward(G.x, G.edge_index)
    y = G.y

    micro_test, macro_test = f1_Score(y[G.split["test"]], out[G.split["test"]])
    roc_auc_test = _eval_rocauc(y[G.split["test"]], out[G.split["test"]])
    ap_test = ap_score(y[G.split["test"]], out[G.split["test"]])

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
    
    
    
    