import torch
import numpy as np
from torch_scatter import scatter_add
import torch.nn.functional as F
import dgl
from dgl import ops
import sklearn.feature_selection as skfs

def normalize_tensor(mx, symmetric=0):
    """Row-normalize sparse matrix"""
    rowsum = torch.sum(mx, 1)
    if symmetric == 0:
        r_inv = torch.pow(rowsum, -1).flatten()
        r_inv[torch.isinf(r_inv)] = 0.
        r_mat_inv = torch.diag(r_inv)
        mx = torch.mm(r_mat_inv, mx)
        return mx

    else:
        r_inv = torch.pow(rowsum, -0.5).flatten()
        r_inv[torch.isinf(r_inv)] = 0.
        r_mat_inv = torch.diag(r_inv)
        mx = torch.mm(torch.mm(r_mat_inv, mx), r_mat_inv)
        return mx

def remove_self_loops(edge_index, edge_attr=None):
    r"""Removes every self-loop in the graph given by :attr:`edge_index`, so
    that :math:`(i,i) \not\in \mathcal{E}` for every :math:`i \in \mathcal{V}`.

    Args:
        edge_index (LongTensor): The edge indices.
        edge_attr (Tensor, optional): Edge weights or multi-dimensional
            edge features. (default: :obj:`None`)

    :rtype: (:class:`LongTensor`, :class:`Tensor`)
    """
    row, col = edge_index[0], edge_index[1]
    mask = row != col
    edge_attr = edge_attr if edge_attr is None else edge_attr[mask]
    edge_index = edge_index[:, mask]

    return edge_index, edge_attr

def node_homophily(A, labels):
    """ average of homophily for each node
    """
    src_node = A.coalesce().indices()[0, :]
    targ_node = A.coalesce().indices()[1, :]
    edge_idx = torch.tensor(np.vstack((src_node, targ_node)), dtype=torch.long).contiguous()
    labels = torch.tensor(labels)
    num_nodes = A.shape[0]
    return node_homophily_edge_idx(edge_idx, labels, num_nodes)


def node_homophily_edge_idx(edge_idx, labels, num_nodes):
    """ edge_idx is 2 x(number edges) """
    edge_index = remove_self_loops(edge_idx)[0]
    hs = torch.zeros(num_nodes)
    degs = torch.bincount(edge_index[0, :]).float()
    if degs.size(0) < hs.size(0):
        degs = F.pad(degs, (0, hs.size(0) - degs.size(0)))
    matches = (labels[edge_index[0, :]] == labels[edge_index[1, :]]).float()
    hs = hs.scatter_add(0, edge_index[0, :], matches) / degs
    return hs[degs != 0].mean()

def edge_homophily(A, labels, ignore_negative=False):
    """ gives edge homophily, i.e. proportion of edges that are intra-class
    compute homophily of classes in labels vector
    See Zhu et al. 2020 "Beyond Homophily ..."
    if ignore_negative = True, then only compute for edges where nodes both have
        nonnegative class labels (negative class labels are treated as missing
    """
    src_node, targ_node = A.coalesce().indices()[0, :], A.coalesce().indices()[1, :]  # A.nonzero()
    matching = labels[src_node] == labels[targ_node]
    labeled_mask = (labels[src_node] >= 0) * (labels[targ_node] >= 0)
    if ignore_negative:
        edge_hom = np.mean(matching[labeled_mask])
    else:
        edge_hom = torch.mean(matching.float())
    return edge_hom

def compact_matrix_edge_idx(edge_idx, labels):
    """
     c x c compatibility matrix, where c is number of classes
     H[i,j] is proportion of endpoints that are class j 
     of edges incident to class i nodes 
     "Generalizing GNNs Beyond Homophily"
     treats negative labels as unlabeled
     """
    edge_index = remove_self_loops(edge_idx)[0]
    src_node, targ_node = edge_index[0, :], edge_index[1, :]
    labeled_nodes = (labels[src_node] >= 0) * (labels[targ_node] >= 0)
    label = labels.squeeze()
    c = label.max() + 1
    H = torch.zeros((c, c)).to(edge_index)
    src_label = label[src_node[labeled_nodes]]
    targ_label = label[targ_node[labeled_nodes]]
    for k in range(c):
        sum_idx = torch.where(src_label == k)[0]
        add_idx = targ_label[sum_idx]
        scatter_add(torch.ones_like(add_idx).to(H.dtype), add_idx, out=H[k, :], dim=-1)
        # H[k, :].scatter(dim=-1, index=add_idx, src=torch.ones_like(add_idx).to(H.dtype), reduce="add")
    H = H / torch.sum(H, dim=1, keepdim=True)
    return H

def class_homophily(A, label):
    """ 
    our measure \hat{h}
    treats negative labels as unlabeled 
    """
    src_node = A.coalesce().indices()[0, :]
    targ_node = A.coalesce().indices()[1, :]
    edge_index = torch.tensor(torch.stack((src_node, targ_node)), dtype=torch.long).contiguous()
    label = label.squeeze()
    c = label.max() + 1
    H = compact_matrix_edge_idx(edge_index, label)
    nonzero_label = label[label >= 0]
    counts = nonzero_label.unique(return_counts=True)[1]
    proportions = counts.float() / nonzero_label.shape[0]
    val = 0
    for k in range(c):
        class_add = torch.clamp(H[k, k] - proportions[k], min=0)
        if not torch.isnan(class_add):
            # only add if not nan
            val += class_add
    val /= c - 1
    return val


def class_distribution(A, labels):
    edge_index = A.coalesce().indices()
    src_node, targ_node = edge_index[0, :], edge_index[1, :]
    hs = torch.zeros(A.shape[0])
    degs = torch.bincount(src_node).float()
    if degs.size(0) < hs.size(0):
        degs = F.pad(degs, (0, hs.size(0) - degs.size(0)))

    # remove self-loop
    deg = degs - 1
    edge_index = remove_self_loops(A.coalesce().indices())[0]
    src_node, targ_node = edge_index[0, :], edge_index[1, :]

    labels = labels.squeeze()
    p = labels.unique(return_counts=True)[1] / labels.shape[0]
    num_class = int(labels.max().cpu() + 1)
    p_bar = torch.zeros(num_class)
    pc = torch.zeros((num_class, num_class))
    for i in range(num_class):
        p_bar[i] = torch.sum(deg[torch.where(labels == i)])

        for j in range(num_class):
            pc[i, j] = torch.sum(labels[targ_node[torch.where(labels[src_node] == i)]] == j)
    p_bar, pc = p_bar / torch.sum(deg), pc / torch.sum(deg)
    p_bar[torch.where(p_bar == 0)], pc[torch.where(pc == 0)] = 1e-8, 1e-8
    return p, p_bar, pc

def adjusted_homo(A, label):
    p, p_bar, pc = class_distribution(A, label)
    edge_homo = edge_homophily(A, label)
    adj_homo = (edge_homo - torch.sum(p_bar ** 2)) / (1 - torch.sum(p_bar ** 2))

    return adj_homo

def mi_agg(graph, features, ori_labels, train_idx=None):
    graph = dgl.remove_self_loop(graph)

    degrees = graph.out_degrees().float()
    degree_edge_products = ops.u_mul_v(graph, degrees, degrees)
    epsilon = 1e-7
    coefs = 1 / (degree_edge_products ** 0.5 + epsilon)
    feat_agg = ops.u_mul_e_sum(graph, features, coefs)
    # mi_nei_lst = []
    # for fi in range(features.shape[1]):
    #     # mi_nei = mutual_info_score(labels.cpu(), feat_agg[:,fi].cpu())
    #     mi_nei = mutual_info_classif(feat_agg[:,[fi]].cpu(), ori_labels.cpu())[0]
    #     mi_nei_lst.append(mi_nei)
    
    if train_idx is not None:
        feat_agg = feat_agg[train_idx].cpu()
        ori_labels = ori_labels[train_idx].cpu()
    mi_nei_lst = skfs.mutual_info_classif(feat_agg, ori_labels)
    hom_res = torch.tensor(mi_nei_lst)
    return hom_res