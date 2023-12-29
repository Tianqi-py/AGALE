import torch
import numpy as np
from sklearn.metrics import f1_score
from sklearn.preprocessing import MultiLabelBinarizer
from sklearn.metrics import average_precision_score
from sklearn.metrics import roc_auc_score
from torch.autograd import Variable

def f1_Score(y, predictions):
    y = y.data.cpu().numpy()
    predictions = predictions.data.cpu().numpy()
    number_of_labels = y.shape[1]
    # find the indices (labels) with the highest probabilities (ascending order)
    pred_sorted = np.argsort(predictions, axis=1)

    # the true number of labels for each node
    num_labels = np.sum(y, axis=1)
    # we take the best k label predictions for all nodes, where k is the true number of labels
    pred_reshaped = []
    for pr, num in zip(pred_sorted, num_labels):
        pred_reshaped.append(pr[-int(num):].tolist())

    # convert back to binary vectors
    pred_transformed = MultiLabelBinarizer(classes=range(number_of_labels)).fit_transform(pred_reshaped)
    f1_micro = f1_score(y, pred_transformed, average='micro')
    f1_macro = f1_score(y, pred_transformed, average='macro')
    return f1_micro, f1_macro


def BCE_loss(outputs: torch.Tensor, labels: torch.Tensor):
    loss = torch.nn.BCELoss()
    bce = loss(outputs, labels)
    return bce


def ap_score(y_true, y_pred):
    ap_score = average_precision_score(y_true.cpu().detach().numpy(), y_pred.cpu().detach().numpy())

    return ap_score


def _eval_rocauc(y_true, y_pred):
    y_true = y_true.detach().cpu().numpy()
    y_pred = y_pred.detach().cpu().numpy()
    rocauc_list = []

    for i in range(y_true.shape[1]):

        # AUC is only defined when there is at least one positive data.
        if np.sum(y_true[:, i] == 1) > 0 and np.sum(y_true[:, i] == 0) > 0:
            is_labeled = y_true[:, i] == y_true[:, i]
            rocauc_list.append(roc_auc_score(y_true[is_labeled, i], y_pred[is_labeled, i]))

    if len(rocauc_list) == 0:
        raise RuntimeError('No positively labeled data available. Cannot compute ROC-AUC.')

    # return {'rocauc': sum(rocauc_list)/len(rocauc_list)}
    return sum(rocauc_list) / len(rocauc_list)


def CE_loss(logits, labels):
    # the multi-class prediction use logits
    loss = torch.nn.functional.cross_entropy(logits, labels)
    return loss


def MultiClassCrossEntropy(logits, labels, T):
    labels = Variable(labels.data, requires_grad=False).cuda()
    outputs = torch.log_softmax(logits/T, dim=1)   # compute the log of softmax values
    labels = torch.softmax(labels/T, dim=1)
    outputs = torch.sum(outputs * labels, dim=1, keepdim=False)
    outputs = -torch.mean(outputs, dim=0, keepdim=False)
    return outputs


def accuracy(logits, labels, cls_balance=False, ids_per_cls=None):
    if cls_balance:
        logi = logits.cpu().detach().numpy()
        _, indices = torch.max(logits, dim=1)
        ids = _.cpu().detach().numpy()
        acc_per_cls = [torch.sum((indices == labels)[ids])/len(ids) for ids in ids_per_cls]
        return sum(acc_per_cls).item()/len(acc_per_cls)
    else:
        _, indices = torch.max(logits, dim=1)
        # convert the one-hot encoded labels into intergers
        labels = np.argmax(labels, axis=1)
        correct = torch.sum(indices == labels)
        return correct.item() * 1.0 / len(labels)

