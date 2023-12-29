import argparse
def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--data_name", default='blogcatalog',
                        help='Name of the dataset'
                             'Hyperspheres_64_64_0'
                             'pcg_removed_isolated_nodes'
                             'Humloc'
                             'yelp'
                             'ogbn-proteins')
    parser.add_argument('--train_percent', type=float, default=0.6,
                        help='percentage of data used for training')
    parser.add_argument("--shuffle_idx", type=str, default='shuffle1',
                        help='Index of the shuffled class order')
    parser.add_argument("--model_name", default='GAT',
                        help='')
    parser.add_argument('--no-cuda', default=False,
                        help='Disables CUDA training.')
    parser.add_argument("--device_name", default='cuda',
                        help='Name of the device')
    parser.add_argument('--fastmode', action='store_true', default=False,
                        help='Validate during training pass.')
    parser.add_argument('--seed', type=int, default=42, help='Random seed.')
    parser.add_argument('--Cross_Task_Message_Passing', default=False,
                        help='Subgraph containing nodes from other tasks')
    parser.add_argument('--epochs', type=int, default=3,
                        help='Number of epochs to train.')
    parser.add_argument('--lr', type=float, default=0.01,
                        help='Initial learning rate.')
    parser.add_argument('--weight_decay', type=float, default=5e-4,
                        help='Weight decay (L2 loss on parameters).')
    parser.add_argument('--hidden', type=int, default=256,
                        help='Number of hidden units.')
    parser.add_argument('--dropout', type=float, default=0.2,
                        help='Dropout rate (1 - keep probability).')
    parser.add_argument('--layer', type=float, default=2,
                        help='number of layer in LFLF')
    parser.add_argument('--patience', type=float, default=100,
                        help='patience for early stopping.')
    parser.add_argument('--batch_size', type=int, default=256,
                        help='patience for early stopping.')
    return parser.parse_args()
