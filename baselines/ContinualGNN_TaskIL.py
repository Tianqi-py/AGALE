# task incremental ContinualGNN
from args import parse_args
from utils import prepare_sub_graph
from CGL_DataLoader import load_hyper_data, load_mat_data, import_yelp
from collections import defaultdict
import sys
import os
import logging
import time
import numpy as np
import warnings
warnings.filterwarnings("ignore")

import torch

from backbones.backbones import GraphSAGE
from baselines.models.ContinualGNN.ContinualGNN_model import EWC
from baselines.models.ContinualGNN.ContinualGNN_model import ModelHandler
from extensions import memory_handler


def train(data, model, args):
    # Model training
    times = []
    for epoch in range(args.num_epochs):
        losses = 0
        start_time = time.time()

        nodes = data.train_nodes
        np.random.shuffle(nodes)
        for batch in range(len(nodes) // args.batch_size):
            batch_nodes = nodes[batch * args.batch_size: (batch + 1) * args.batch_size]
            batch_labels = torch.LongTensor(data.labels[np.array(batch_nodes)]).to(args.device)

            model.optimizer.zero_grad()
            loss = model.loss(batch_nodes, batch_labels)
            loss.backward()
            model.optimizer.step()

            loss_val = loss.data.item()
            losses += loss_val * len(batch_nodes)
            if (np.isnan(loss_val)):
                logging.error('Loss Val is NaN !!!')
                sys.exit()

        if epoch % 10 == 0:
            logging.debug('--------- Epoch: ' + str(epoch) + ' ' + str(np.round(losses / data.train_size, 10)))
        end_time = time.time()
        times.append(end_time - start_time)

    avg_time = np.round(np.mean(times), 6)
    logging.info("Average epochs time: " + str(avg_time))
    return avg_time


if __name__ == "__main__":
    args = parse_args()
    # load datasets
    try:
        if args.data_name == "blogcatalog":
            G = load_mat_data()
        elif args.data_name == "Hyperspheres_10_10_0":
            G = load_hyper_data()
        elif args.data_name == "yelp":
            G = import_yelp()
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
            print("intialize ContinualGNN using GraphSAGE as backbone...")

            #######################################################################################
            # special data management for ContinualGNN

            # for ContinualGNN add adj_list in the form: {node1:set(neighbor1, neighbor2, ...),
            #                                             node2:set(neighbor1, neighbor2), ...}

            sub_g.adj_lists = defaultdict(set)
            edge_list = torch.transpose(sub_g.edge_index, 0, 1)
            # in edge index two directions are saved
            for e in edge_list:
                sub_g.adj_lists[e[0].item()].add(e[1].item())

            #train_nodes
            sub_g.train_nodes = sub_g[sub_g.split["train"]]
            sub_g.train_all_nodes_list = []
            sub_g.train_old_nodes = []
            sub_g.val_all_nodes = []
            sub_g.val_old_nodes = []
            #######################################################################################

            layers = [sub_g.shape[1]] + [args.hidden_unit] * args.layer + [sub_g.y.shape[1]]

            model = GraphSAGE(sub_g.adj_lists, sub_g.features, sub_g.adj_lists, args)

            memory_h = memory_handler.MemoryHandler(args)
            ewc_model = model.to(args.device)

            # Train
            ewc_model.optimizer = torch.optim.SGD(model.parameters(), lr=args.lr)
            avg_time = train(sub_g, ewc_model, args)

            # Model save
            model_handler_cur = ModelHandler(os.path.join(args.save_path, str(t)))
            model_handler_cur.save(model.state_dict(), 'graph_sage.pkl')

            # Memory save
            if args.memory_size > 0:
                train_output = model.forward(sub_g.train_nodes).data.cpu().numpy()
                memory_h.update(sub_g.train_nodes, x=train_output, y=sub_g.labels, adj_lists=sub_g.adj_lists)
                memory_handler.save(memory_h, 'M')

        # t > 0
        else:
            model_handler_pre = ModelHandler(os.path.join(args.save_path, str(t - 1)))
            if not model_handler_pre.not_exist():
                model.load_state_dict(model_handler_pre.load('graph_sage.pkl'))

            ewc_model = EWC(model, args.ewc_lambda, args.ewc_type).to(args.device)

            # whether use memory to store important nodes
            if args.memory_size > 0:
                memory_h = memory_handler.load('M', args)
                important_nodes_list = memory_h.memory
                sub_g.train_nodes = list(set(sub_g.train_nodes + important_nodes_list))
                logging.info(
                    'Important Data Size: ' + str(len(important_nodes_list)) + ' / ' + str(len(sub_g.train_nodes)))
            else:
                important_nodes_list = sub_g.train_old_nodes_list

            # calculate weight importance
            ewc_model.register_ewc_params(important_nodes_list,
                                          torch.LongTensor(sub_g.labels[important_nodes_list]).to(args.device))



