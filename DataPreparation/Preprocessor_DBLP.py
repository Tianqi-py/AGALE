import os
import torch
import random
import numpy as np
from sklearn.utils import shuffle
from sklearn.model_selection import train_test_split

def process_dblp(data_name="dblp", path="dblp/",  n_cls_per_t=2, train_ratio=0.6):

    # load data
    print('Loading dataset ' + data_name + '...')
    dblp_labels = torch.FloatTensor(np.genfromtxt(os.path.join(path, "labels.txt"), delimiter=","))

    # remove small class c if size(c) < 50:
    l = torch.transpose(dblp_labels, 0, 1)
    cls_size = [torch.nonzero(ll).shape[0] for ll in l]
    rmv_cls = [cls_size.index(s) for s in cls_size if s < 50]

    kp_cls = [i for i in range(dblp_labels.shape[1]) if i not in rmv_cls]
    dblp_lbl_clean = dblp_labels[:, kp_cls]
    torch.save(dblp_lbl_clean, path+'dblp_clean_lbl.pt')

    n_cls = dblp_lbl_clean.shape[1]

    # creat random shuffles
    cls_order = shuffle(range(dblp_lbl_clean.shape[1]))
    print(cls_order)
    torch.save(cls_order, path + 'dblp_shuffle1.pt')
    cls_order = shuffle(range(dblp_lbl_clean.shape[1]))
    print(cls_order)
    torch.save(cls_order, path + 'dblp_shuffle2.pt')
    cls_order = shuffle(range(dblp_lbl_clean.shape[1]))
    torch.save(cls_order, path + 'dblp_shuffle3.pt')
    print(cls_order)

    # creat tasks for TaskIL
    cls_order1 = torch.load(path + "dblp_shuffle1.pt")
    cls_order2 = torch.load(path + "dblp_shuffle2.pt")
    cls_order3 = torch.load(path + "dblp_shuffle3.pt")

    groups_idx1 = [tuple(cls_order1[i:i + n_cls_per_t]) for i in range(0, n_cls, n_cls_per_t)]
    groups_idx2 = [tuple(cls_order2[i:i + n_cls_per_t]) for i in range(0, n_cls, n_cls_per_t)]
    groups_idx3 = [tuple(cls_order3[i:i + n_cls_per_t]) for i in range(0, n_cls, n_cls_per_t)]
    # check if the last group of classes only have one label
    if len(groups_idx1[-1]) == 1:
        # if so move it to the second last tuple
        groups_idx1[-2] = groups_idx1[-2] + groups_idx1[-1]
        groups_idx1.pop(-1)

        groups_idx2[-2] = groups_idx2[-2] + groups_idx2[-1]
        groups_idx2.pop(-1)

        groups_idx3[-2] = groups_idx3[-2] + groups_idx3[-1]
        groups_idx3.pop(-1)

    groups_idxs = []
    groups_idxs.append(groups_idx1)
    groups_idxs.append(groups_idx2)
    groups_idxs.append(groups_idx3)

    # cls_asgn is a dictionary {class1:torch.tensor(node1, node5,...)}, values are the nodes which have class1
    # note: for multi-label datasets, the node ids may appear in multiple keys' values
    cls = torch.transpose(dblp_lbl_clean, 0, 1)
    print(cls.shape)
    cls_asgn = {}
    for i, label in enumerate(cls):
        cls_asgn[i] = torch.nonzero(label).flatten()
    cls_asgn.keys()

    # TaskIL split
    # requirment: no duplicates within each group of classes

    # split: 60%train, 10%val, 30%test

    # three orders to present the classes
    for k, groups_idx in enumerate(groups_idxs):

        # nested dictionary splits={group1:{train:[], val:[], test[]}, group2...}
        splits = {}

        # each group of classes in one order
        for group in groups_idx:

            print("##########################################")
            print("classes in this group", group)
            train_ids = []
            val_ids = []
            test_ids = []
            split = {}

            # check for duplicates: nodes ids already seen, no duplicates in n_g
            n_g = []

            # split the small class first to avoid complete node overlap
            # sort the classes base on their sizes
            size = []
            for i, c in enumerate(group):
                size.append(len(cls_asgn[c]))

            sort_index = np.argsort(size)
            group = tuple([group[j] for j in sort_index])

            # split the group
            for i, c in enumerate(group):

                # the first class: 60%, 10%, 30%
                if i == 0:
                    n_c = cls_asgn[c].tolist()
                    n_g.extend(n_c)
                    ids_train, ids_val_test = train_test_split(n_c, test_size=0.4, random_state=40)
                    ids_val, ids_test = train_test_split(ids_val_test, test_size=0.75, random_state=39)
                    train_ids.extend(ids_train)
                    val_ids.extend(ids_val)
                    test_ids.extend(ids_test)
                    print("number of nodes in total in c1", len(n_c))
                    print("total train for c1", len(ids_train))
                    print("total val for c1", len(ids_val))
                    print("total test for c1", len(ids_test))

                # other group of classes
                else:
                    # nodes ids for the duplicates
                    dup = []

                    print(c)
                    # check for duplicate nodes in the previous classes and the current class
                    n_c = cls_asgn[c].tolist()
                    n_c_all = n_c.copy()
                    print("number of nodes in total in class", len(n_c))
                    for m in n_c_all:
                        # if there is, remove these nodes from random split
                        if m in n_g:
                            n_c.remove(m)
                            dup.append(m)
                            print("node ", m, "already split from its other labels")
                    print(len(n_c))
                    print("check if there is duplicated nodes in the remaining nodes:", len(set(n_c)), len(n_c))
                    n_g.extend(n_c)

                    # split current class
                    # num_n in train, val, test in total to sample for this class
                    n_c_train = int(len(n_c_all) * train_ratio)
                    n_c_val = int((len(n_c_all) - n_c_train) * 0.25)
                    n_c_test = len(n_c_all) - n_c_train - n_c_val
                    print("total train for class", n_c_train)
                    print("total val for class", n_c_val)
                    print("total test for class", n_c_test)

                    # number to sample from the new apeared nodes
                    for d in dup:
                        if d in train_ids:
                            n_c_train = n_c_train - 1
                        if d in val_ids:
                            n_c_val = n_c_val - 1
                        if d in test_ids:
                            n_c_test = n_c_test - 1
                    # sample training nodes
                    print("number of nodes after removal", len(n_c))
                    print("remain in train set to sample", n_c_train)
                    print("remain in val set to sample", n_c_val)
                    print("remain in test set to sample", n_c_test)

                    train_to_sample = random.sample(n_c, k=n_c_train)
                    train_ids.extend(train_to_sample)
                    print(len(train_to_sample))
                    remain_nodes = list(set(n_c) - set(train_to_sample))
                    print("remain_nodes", len(remain_nodes))
                    # sample val nodes
                    val_to_sample = random.sample(remain_nodes, k=n_c_val)
                    print(len(val_to_sample))
                    val_ids.extend(val_to_sample)
                    # sample test nodes
                    remain_nodes = list(set(remain_nodes) - set(val_to_sample))
                    print("remain_nodes", len(remain_nodes))
                    print(len(remain_nodes))
                    # remains go to test set
                    test_ids.extend(remain_nodes)

            # check for duplicates
            print(len(n_g))
            print("all nodes at t", len(list(set(n_g))))
            print("final number of training nodes", len(train_ids))
            print("final number of val nodes", len(val_ids))
            print("final number of test nodes", len(test_ids))
            for i in train_ids:
                if i in val_ids:
                    print(i)
                    raise Exception("Sorry, train found in val")
                if i in test_ids:
                    print(i)
                    raise Exception("Sorry, train found in test")
            for i in val_ids:
                if i in test_ids:
                    print(i)
                    raise Exception("Sorry, val found in test")

            # save in split
            split["train"] = train_ids
            split["val"] = val_ids
            split["test"] = test_ids
            splits[group] = split
            torch.save(splits, path+"/split_" + str(k+1) + ".pt")

    for i in range(3):
        splits = torch.load(path + "split_" + str(i + 1) + ".pt")
        groups = {}
        for sub_g in splits.keys():
            ids = []
            print(sub_g)
            for split in splits[sub_g].keys():
                print(split)
                dct = splits[sub_g]
                ids.extend(dct[split])
            groups[sub_g] = ids

        torch.save(groups, data_name + "/groups" + str(i + 1) + ".pt")


process_dblp()
print("preprocessing finished")

