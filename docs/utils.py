# import features form new version python
from __future__ import division  # '/' for division
from __future__ import print_function  # print with '()'

import numpy as np
import scipy.sparse as sp
import torch
import sys

# from train.py

import time
import argparse

import torch
import torch.nn.functional as F
import torch.optim as optim

# from pygcn.utils import load_data, accuracy
from pygcn.models import GCN

from sklearn.metrics import accuracy_score,f1_score,roc_auc_score,recall_score,precision_score

from tensorflow.python.keras.engine import training

import utils
import eli5
from eli5.sklearn import PermutationImportance

def encode_onehot(labels):
    classes = set(labels)
    classes_dict = {c: np.identity(len(classes))[i, :] for i, c in
                    enumerate(classes)}
    labels_onehot = np.array(list(map(classes_dict.get, labels)),
                             dtype=np.int32)
    return labels_onehot


def load_data(path="../data/coraTest0326/", dataset="coraSh"):
    """Load citation network dataset (cora only for now)"""

    print('Phishing Loading {} dataset...'.format(dataset))

    # content file: <paper_id> <word_attributes>+ <class_label>
    # 分别对应 0, 1:-1, -1
    idx_features_labels = np.genfromtxt("{}{}.content".format(path, dataset),
                                        dtype=np.dtype(str))
    features = sp.csr_matrix(idx_features_labels[:, 1:-1], dtype=np.float32)    # features, 储存为csr型稀疏矩阵
    labels = encode_onehot(idx_features_labels[:, -1])                          # labels, onthot格式，如第一类代表[1,0,0,0,0,0,0]

    # build graph
    # cites file: <cited paper ID>  <citing paper ID>
    # 根据前面的contents与这里的cites创建图，算出edges矩阵与adj矩阵
    # idx = np.array(idx_features_labels[:, 0], dtype=np.int32)
    idx = np.array(idx_features_labels[:, 0], dtype=np.float32)
    idx = idx.astype(int)
    # 由于文件中节点并非是按顺序排列的，因此建立一个编号为0-(node_size-1)的哈希表idx_map，
    # 哈希表中每一项为id: number，即节点id对应的编号为number
    idx_map = {j: i for i, j in enumerate(idx)}
    # edges_unordered为直接从边表文件中直接读取的结果，是一个(edge_num, 2)的数组，每一行表示一条边两个端点的idx
    # 边的edges_unordered中存储的是端点id，要将每一项的id换成编号。
    # 在idx_map中以idx作为键查找得到对应节点的编号，reshape成与edges_unordered形状一样的数组
    edges_unordered = np.genfromtxt("{}{}.cites".format(path, dataset),
                                    dtype=np.int32)
    edges = np.array(list(map(idx_map.get, edges_unordered.flatten())), # flatten：降维，返回一维数组
                     dtype=np.int32).reshape(edges_unordered.shape)
    # 根据coo矩阵性质，这一段的作用就是，网络有多少条边，邻接矩阵就有多少个1，
    # 所以先创建一个长度为edge_num的全1数组，每个1的填充位置就是一条边中两个端点的编号，
    # 即edges[:, 0], edges[:, 1]，矩阵的形状为(node_size, node_size)。
    adj = sp.coo_matrix((np.ones(edges.shape[0]), (edges[:, 0], edges[:, 1])),  # coo型稀疏矩阵
                        shape=(labels.shape[0], labels.shape[0]),
                        dtype=np.float32)

    # build symmetric adjacency matrix, A^=(D~)^0.5 A~ (D~)^0.5
    # 对于无向图，邻接矩阵是对称的。上一步得到的adj是按有向图构建的，转换成无向图的邻接矩阵需要扩充成对称矩阵
    adj = adj + adj.T.multiply(adj.T > adj) - adj.multiply(adj.T > adj)

    features = normalize(features)
    # eye创建单位矩阵，第一个参数为行数，第二个为列数, A~=A+IN
    adj = normalize(adj + sp.eye(adj.shape[0]))

    # 3 data sets: train, verification, test
    sample_ratio = 0.05
    idx_test = list(range(0, int(1700+1660*sample_ratio))) + list(range(3360, int(3360+92978*sample_ratio)))          # 140 - 8678 (8678)
    idx_val = list(range(int(1700+1660*sample_ratio), int(1700+1660*sample_ratio*2))) + list(range(int(3360+92978*sample_ratio), int(3360+92978*sample_ratio*2)))       # 200, 500 - 8679, 27273 (18595)
    idx_train = list(range(int(1700+1660*sample_ratio*2),3360)) + list(range(int(3360+92978*sample_ratio*2), 92977))     # 500, 1500 - 27274, 92977 (65703)
    features = torch.FloatTensor(np.array(features.todense()))  # tensor为pytorch常用的数据结构
    labels = torch.LongTensor(np.where(labels)[1])  # 将onthot label转回index
    adj = sparse_mx_to_torch_sparse_tensor(adj)     # 邻接矩阵转为tensor处理

    idx_train = torch.LongTensor(idx_train)
    idx_val = torch.LongTensor(idx_val)
    idx_test = torch.LongTensor(idx_test)

    return adj, features, labels, idx_train, idx_val, idx_test


def normalize(mx):
    """Row-normalize sparse matrix"""
    rowsum = np.array(mx.sum(1))
    r_inv = np.power(rowsum, -1).flatten()
    r_inv[np.isinf(r_inv)] = 0.
    r_mat_inv = sp.diags(r_inv)
    mx = r_mat_inv.dot(mx)
    return mx


def accuracy(output, labels, printint):
    preds = output.max(1)[1].type_as(labels)
    correct = preds.eq(labels).double()
    correct = correct.sum()

    # acc_score = accuracy_score(labels, preds)
    # roc_auc_score1 = roc_auc_score(labels, preds)
    raca_score = recall_score(labels, preds)
    prec_score = precision_score(labels,preds)
    f1 = f1_score(labels, preds)

    if printint == 1:
        print("recall:")
        print(raca_score)
        print("precision:")
        print(prec_score)
        print("f1:")
        print(f1)
    return correct / len(labels)


def sparse_mx_to_torch_sparse_tensor(sparse_mx):
    """Convert a scipy sparse matrix to a torch sparse tensor."""
    sparse_mx = sparse_mx.tocoo().astype(np.float32)
    indices = torch.from_numpy(
        np.vstack((sparse_mx.row, sparse_mx.col)).astype(np.int64))
    values = torch.from_numpy(sparse_mx.data)
    shape = torch.Size(sparse_mx.shape)
    return torch.sparse.FloatTensor(indices, values, shape)


# from train.py
# Training settings
np.set_printoptions(threshold=np.inf)

parser = argparse.ArgumentParser()
parser.add_argument('--no-cuda', action='store_true', default=False,
                    help='Disables CUDA training.')
parser.add_argument('--fastmode', action='store_true', default=False,
                    help='Validate during training pass.')
parser.add_argument('--seed', type=int, default=42, help='Random seed.')
parser.add_argument('--epochs', type=int, default=200,
                    help='Number of epochs to train.')
parser.add_argument('--lr', type=float, default=0.01,
                    help='Initial learning rate.')
parser.add_argument('--weight_decay', type=float, default=5e-4,
                    help='Weight decay (L2 loss on parameters).')
parser.add_argument('--hidden', type=int, default=16,
                    help='Number of hidden units.')
parser.add_argument('--dropout', type=float, default=0.5,
                    help='Dropout rate (1 - keep probability).')

args = parser.parse_args()
args.cuda = not args.no_cuda and torch.cuda.is_available()

np.random.seed(args.seed)
torch.manual_seed(args.seed)
if args.cuda:
    torch.cuda.manual_seed(args.seed)
"""
其中train.py line 25～line 66主要进行数据读取，预处理等工作，
截止到train.py line 66，利用cora的数据集构成的图还是一个network的图，
train.py line 67将network图转换为DGL的图，这里体现了DGL的一个特性，支持对network图进行类型转换。
train.py Line 79~94开始实例化模型，定义loss及优化器。"""
# Load data
adj, features, labels, idx_train, idx_val, idx_test = load_data()

# Model and optimizer
model = GCN(nfeat=features.shape[1],
            nhid=args.hidden,
            nclass=labels.max().item() + 1,
            dropout=args.dropout)
optimizer = optim.Adam(model.parameters(),
                       lr=args.lr, weight_decay=args.weight_decay)

if args.cuda:
    model.cuda()
    features = features.cuda()
    adj = adj.cuda()
    labels = labels.cuda()
    idx_train = idx_train.cuda()
    idx_val = idx_val.cuda()
    idx_test = idx_test.cuda()


def train(epoch):
    t = time.time()
    model.train()
    optimizer.zero_grad()
    output = model(features, adj)
    loss_train = F.nll_loss(output[idx_train], labels[idx_train])
    acc_train = accuracy(output[idx_train], labels[idx_train], 0)
    loss_train.backward()
    optimizer.step()

    if not args.fastmode:
        # Evaluate validation set performance separately,
        # deactivates dropout during validation run.
        model.eval()
        output = model(features, adj)

    loss_val = F.nll_loss(output[idx_val], labels[idx_val])
    acc_val = accuracy(output[idx_val], labels[idx_val], 0)
    print('Epoch: {:04d}'.format(epoch+1),
          'loss_train: {:.4f}'.format(loss_train.item()),
          'acc_train: {:.4f}'.format(acc_train.item()),
          'loss_val: {:.4f}'.format(loss_val.item()),
          'acc_val: {:.4f}'.format(acc_val.item()),
          'time: {:.4f}s'.format(time.time() - t))


def test():
    model.eval()
    # for par in model.parameters():
    #     print("line:")
    #     print(par)
    output = model(features, adj)
    loss_test = F.nll_loss(output[idx_test], labels[idx_test])
    acc_test = accuracy(output[idx_test], labels[idx_test], 1)

    print("Test set results:",
          "loss= {:.4f}".format(loss_test.item()),
          "accuracy= {:.4f}".format(acc_test.item()))
          # "precision= {:.4f}".format(precision_test.item()),
          # "recall= {:.4f}".format(recall_test.item()),
          # "F1_test= {:.4f}".format(F1_test.item()),
          # "accuracy2= {:.4f}".format(acc_test2.item()))

    perm = PermutationImportance(model, random_state=1).fit(features, labels[idx_test])
    eli5.show_weights(perm, feature_names=features.columns.tolist())


if __name__ == "__main__":
    # Train model
    t_total = time.time()
    for epoch in range(args.epochs):
        train(epoch)
    print("Optimization Finished!")
    print("Total time elapsed: {:.4f}s".format(time.time() - t_total))

    # Testing
    test()
