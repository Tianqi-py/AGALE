import argparse
def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--model_name", type=str, default='GCN',
                        help="name of the model"
                             "ERGNN")
    parser.add_argument("--data_name", default='blogcatalog',
                        help='Name of the dataset'
                             'Hyperspheres_10_10_0'
                             'pcg_removed_isolated_nodes'
                             'Humloc'
                             'yelp'
                             'ogbn-proteins'
                             'blogcatalog'
                             "cora"
                             "Citeseer"
                             "PubMed"
                             "CoraFull")
    parser.add_argument("--multi_class", default=False,
                        help='if the dataset is multi-class, default to false.')
    parser.add_argument("--setting", default='Task-IL',
                        help='name of the incremental setting'
                             'Task-IL'
                             'Class-IL'
                             'Domain-IL'
                             'Time-IL')
    parser.add_argument("--shuffle_idx", type=str, default='shuffle1',
                        help='Index of the shuffled class order')
    parser.add_argument('--Cross_Task_Message_Passing', default=False,
                        help='Subgraph containing nodes from other tasks'
                             'should be false in the task incremental setting')
    # parser.add_argument('--ergnn_args', type=str2dict, default={'budget': [100, 1000], 'd': [0.5], 'sampler': ['CM']},
    #                     help='sampler options: CM, CM_plus, MF, MF_plus')
    # memory strength of MAS
    parser.add_argument('--ewc_reg', type=int, default=10000)
    parser.add_argument('--mas_mem_str', type=int, default=10000)
    parser.add_argument('--budget_ergnn', type=list, default=[100, 1000],
                        help='buffer size in ERGNN: number of nodes sampled')
    parser.add_argument('--d_ergnn', type=float, default=0.5,
                        help='distance for CM sampler')
    parser.add_argument('--sampler_ergnn', type=str, default=["CM"],
                        help='sampler in ERGNN')

    parser.add_argument('--n_cls_per_t', type=int, default=2,
                        help='number of classes arrive per time step for Class-IL setting')
    parser.add_argument('--train_percent', type=float, default=0.6,
                        help='percentage of data used for training')
    # parser.add_argument("--split_name", default='split_1.pt',
    #                     help='Name of the split')
    parser.add_argument("--backbone", default='GCN',
                        help='backbone models'
                             "GCN"
                             "GAT")

    parser.add_argument('--no-cuda', default=False,
                        help='Disables CUDA training.')
    parser.add_argument("--device_name", default='cuda',
                        help='Name of the device')
    parser.add_argument('--fastmode', action='store_true', default=False,
                        help='Validate during training pass.')
    parser.add_argument('--seed', type=int, default=42, help='Random seed.')

    parser.add_argument('--epochs', type=int, default=1000,
                        help='Number of epochs to train.')
    parser.add_argument('--lr', type=float, default=0.01,
                        help='Initial learning rate.')
    parser.add_argument('--weight_decay', type=float, default=5e-4,
                        help='Weight decay (L2 loss on parameters).')
    parser.add_argument('--hidden', type=int, default=256,
                        help='Number of hidden units in each hidden layer.')
    parser.add_argument('--dropout', type=float, default=0.0,
                        help='Dropout rate (1 - keep probability).')
    parser.add_argument('--layer', type=float, default=2,
                        help='number of layer in LFLF')
    parser.add_argument('--patience', type=float, default=100,
                        help='patience for early stopping.')
    parser.add_argument('--batch_size', type=int, default=256,
                        help='patience for early stopping.')

    return parser.parse_args()