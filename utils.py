import os
import yaml
import numpy as np
import torch
import dgl
from dgl import ops
import sklearn.feature_selection as skfs
from tqdm import tqdm
from sklearn.metrics.pairwise import cosine_similarity

class Logger:
    def __init__(self, args, metric, num_data_splits):
        self.save_dir = self.get_save_dir(base_dir=args.save_dir, dataset=args.dataset, name=args.name)
        self.verbose = args.verbose
        self.metric = metric
        self.val_metrics = []
        self.test_metrics = []
        self.best_steps = []
        self.num_runs = args.num_runs
        self.num_data_splits = num_data_splits
        self.cur_run = None
        self.cur_data_split = None

        print(f'Results will be saved to {self.save_dir}.')
        with open(os.path.join(self.save_dir, 'args.yaml'), 'w') as file:
            yaml.safe_dump(vars(args), file, sort_keys=False)

    def start_run(self, run, data_split):
        self.cur_run = run
        self.cur_data_split = data_split
        self.val_metrics.append(0)
        self.test_metrics.append(0)
        self.best_steps.append(None)

        if self.num_data_splits == 1:
            print(f'Starting run {run}/{self.num_runs}...')
        else:
            print(f'Starting run {run}/{self.num_runs} (using data split {data_split}/{self.num_data_splits})...')

    def update_metrics(self, metrics, step):
        if metrics[f'val {self.metric}'] > self.val_metrics[-1]:
            self.val_metrics[-1] = metrics[f'val {self.metric}']
            self.test_metrics[-1] = metrics[f'test {self.metric}']
            self.best_steps[-1] = step

        if self.verbose:
            print(f'run: {self.cur_run:02d}, step: {step:03d}, '
                  f'train {self.metric}: {metrics[f"train {self.metric}"]:.4f}, '
                  f'val {self.metric}: {metrics[f"val {self.metric}"]:.4f}, '
                  f'test {self.metric}: {metrics[f"test {self.metric}"]:.4f}')

    def finish_run(self):
        self.save_metrics()
        print(f'Finished run {self.cur_run}. '
              f'Best val {self.metric}: {self.val_metrics[-1]:.4f}, '
              f'corresponding test {self.metric}: {self.test_metrics[-1]:.4f} '
              f'(step {self.best_steps[-1]}).\n')

    def save_metrics(self):
        num_runs = len(self.val_metrics)
        val_metric_mean = np.mean(self.val_metrics).item()
        val_metric_std = np.std(self.val_metrics, ddof=1).item() if len(self.val_metrics) > 1 else np.nan
        test_metric_mean = np.mean(self.test_metrics).item()
        test_metric_std = np.std(self.test_metrics, ddof=1).item() if len(self.test_metrics) > 1 else np.nan

        metrics = {
            'num runs': num_runs,
            f'val {self.metric} mean': val_metric_mean,
            f'val {self.metric} std': val_metric_std,
            f'test {self.metric} mean': test_metric_mean,
            f'test {self.metric} std': test_metric_std,
            f'val {self.metric} values': self.val_metrics,
            f'test {self.metric} values': self.test_metrics,
            'best steps': self.best_steps
        }

        with open(os.path.join(self.save_dir, 'metrics.yaml'), 'w') as file:
            yaml.safe_dump(metrics, file, sort_keys=False)

    def print_metrics_summary(self):
        with open(os.path.join(self.save_dir, 'metrics.yaml'), 'r') as file:
            metrics = yaml.safe_load(file)

        print(f'Finished {metrics["num runs"]} runs.')
        print(f'Val {self.metric} mean: {metrics[f"val {self.metric} mean"]:.4f}')
        print(f'Val {self.metric} std: {metrics[f"val {self.metric} std"]:.4f}')
        print(f'Test {self.metric} mean: {metrics[f"test {self.metric} mean"]:.4f}')
        print(f'Test {self.metric} std: {metrics[f"test {self.metric} std"]:.4f}')

    @staticmethod
    def get_save_dir(base_dir, dataset, name):
        idx = 1
        save_dir = os.path.join(base_dir, dataset, f'{name}_{idx:02d}')
        while os.path.exists(save_dir):
            idx += 1
            save_dir = os.path.join(base_dir, dataset, f'{name}_{idx:02d}')

        os.makedirs(save_dir)

        return save_dir


def get_parameter_groups(model):
    no_weight_decay_names = ['bias', 'normalization', 'label_embeddings']

    parameter_groups = [
        {
            'params': [param for name, param in model.named_parameters()
                       if not any(no_weight_decay_name in name for no_weight_decay_name in no_weight_decay_names)]
        },
        {
            'params': [param for name, param in model.named_parameters()
                       if any(no_weight_decay_name in name for no_weight_decay_name in no_weight_decay_names)],
            'weight_decay': 0
        },
    ]

    return parameter_groups


def get_lr_scheduler_with_warmup(optimizer, num_warmup_steps=None, num_steps=None, warmup_proportion=None,
                                 last_step=-1):

    if num_warmup_steps is None and (num_steps is None or warmup_proportion is None):
        raise ValueError('Either num_warmup_steps or num_steps and warmup_proportion should be provided.')

    if num_warmup_steps is None:
        num_warmup_steps = int(num_steps * warmup_proportion)

    def get_lr_multiplier(step):
        if step < num_warmup_steps:
            return (step + 1) / (num_warmup_steps + 1)
        else:
            return 1

    lr_scheduler = torch.optim.lr_scheduler.LambdaLR(optimizer, lr_lambda=get_lr_multiplier, last_epoch=last_step)

    return lr_scheduler

# node feature norm
def z_norm(tensor):
    mean = torch.mean(tensor)
    std = torch.std(tensor)
    return (tensor - mean) / std

def mi_agg(graph, features, ori_labels, train_idx=None):
    graph = dgl.remove_self_loop(graph)
    degrees = graph.out_degrees().float()
    degree_edge_products = ops.u_mul_v(graph, degrees, degrees)
    epsilon = 1e-7
    coefs = 1 / (degree_edge_products ** 0.5 + epsilon)
    feat_agg = ops.u_mul_e_sum(graph, features, coefs)
    if train_idx is not None:
        feat_agg = feat_agg[train_idx].cpu()
        ori_labels = ori_labels[train_idx].cpu()
    mi_nei_lst = skfs.mutual_info_classif(feat_agg, ori_labels)
    hom_res = torch.tensor(mi_nei_lst)
    return hom_res

def generalized_edge_homophily(graph, features, ori_labels, train_idx=None):
    graph = dgl.remove_self_loop(graph)
    graph = dgl.add_self_loop(graph)

    nedges = graph.num_edges()
    device = features.device
    adj = graph.adjacency_matrix().to_dense()
    adj = adj - torch.diag(torch.diag(adj))
    adj = (adj > 0).float()
    g_edge_homo = torch.zeros(features.shape[1]).to(device)
    print(f"calculate generalized_edge_homophily...")
    # calculate similarity and g_edge_homo for each dimension of feature
    for i in tqdm(range(features.shape[1])):
        # calculate the similarity metrix for i th dimension of feature 
        sim = torch.tensor(cosine_similarity(features[:, i].cpu().unsqueeze(1))).to(device)
        sim[torch.isnan(sim)] = 0
        g_edge_homo[i] = torch.sum(sim * adj) / torch.sum(adj)

    return g_edge_homo

def attribute_homophily(graph, features, ori_labels, train_idx=None):
    assert(torch.logical_not(ori_labels.unique()>=0).sum()==0) # All the labels must start from 0
    assert(len(ori_labels.unique())==ori_labels.max()+1)
    graph = dgl.remove_self_loop(graph)
    graph = dgl.add_self_loop(graph)

    num_node, num_feat = features.shape[0], features.shape[1]
    src_node, _ = graph.edges()
    out = ops.u_mul_v(graph, features, features)
    out = out.transpose(1,0)
    device = features.device
    feat_agg = torch.zeros(num_feat, num_node).to(device)
    feat_agg = feat_agg.scatter_add_(1,src_node.repeat(num_feat,1).long(),out)
    feat_agg = feat_agg.transpose(1,0)
    degs = torch.bincount(src_node).float()
    # replaced by dgl.add_self_loop(graph)
    # degs[degs==0] = 1 # To prevent zero divide
    feat_agg = feat_agg/degs.unsqueeze(1).repeat(1,num_feat)
    feat_sum = features.sum(dim=0)
    feat_sum[feat_sum==0]=1
    feat_agg = feat_agg.sum(dim=0)
    hom = feat_agg/feat_sum
    return hom

def localsim_cos_homophily(graph, features, ori_labels, train_idx=None):
    graph = dgl.remove_self_loop(graph)
    device = features.device
    src_node, _ = graph.edges()
    num_node, num_feat = features.shape[0], features.shape[1]
    sim_nodes = torch.zeros(num_feat).to(device)

    out = ops.u_mul_v(graph, features, features)
    out_norm = ops.u_mul_v(graph, features.norm(dim=1), features.norm(dim=1)).unsqueeze(1)
    sim = out/out_norm
    sim[sim.isnan()] = 0
    print(f"calculate localsim_cos_homophily...")
    for i in tqdm(range(num_feat)):
        # degs = torch.bincount(src_node).float()
        degs = graph.in_degrees().float()
        degs[degs==0] = 1 # To prevent zero divide
        sim_node = torch.zeros(num_node).to(device)
        sim_node = sim_node.scatter_add_(0,src_node.long(), sim[:, i].squeeze(0))
        sim_node = sim_node/degs
        sim_nodes[i] = sim_node.mean()
    return sim_nodes

def localsim_euc_homophily(graph, features, ori_labels, train_idx=None):
    graph = dgl.remove_self_loop(graph)
    device = features.device
    src_node, _ = graph.edges()
    num_node, num_feat = features.shape[0], features.shape[1]
    sim_nodes = torch.zeros(num_feat).to(device)

    out = ops.u_sub_v(graph, features, features)
    sim = -out
    sim[sim.isnan()] = 0
    print(f"calculate localsim_euc_homophily...")
    for i in tqdm(range(num_feat)):
        # degs = torch.bincount(src_node).float()
        degs = graph.in_degrees().float()
        degs[degs==0] = 1 # To prevent zero divide
        sim_node = torch.zeros(num_node).to(device)
        sim_node = sim_node.scatter_add_(0,src_node.long(), sim[:, i].squeeze(0))
        sim_node = sim_node/degs
        sim_nodes[i] = sim_node.mean()
    return sim_nodes

def class_controlled_feature_homophily(graph, features, ori_labels, train_idx=None):
    assert(torch.logical_not(ori_labels.unique() >= 0).sum() == 0) # All the labels must start from 0
    assert(len(ori_labels.unique()) == ori_labels.max() + 1)
    graph = dgl.remove_self_loop(graph)
    # graph = dgl.add_self_loop(graph)
    if train_idx is not None:
        labels = ori_labels[train_idx]
        # use class info, so feature and graph be modified
        features = features[train_idx]
        graph = graph.subgraph(train_idx.int())

    num_class = labels.unique().shape[0]
    num_node, num_feat = features.shape[0], features.shape[1]
    device = features.device
    feature_cls = torch.zeros(num_class, num_feat).to(device)
    labels = labels.long().to(device)
    for c in range(num_class):
        feature_cls[c] = features[labels==c].mean(dim=0)
    cls_ctrl_feat = features - feature_cls[labels]

    base_hom = (cls_ctrl_feat - cls_ctrl_feat.mean(dim=0).repeat(num_node,1))
    base_hom = 2*base_hom*base_hom # -> [node_num, feat_dim]
    # base_hom = base_hom.sum(dim=1) # -> [node_num, 1]
    src_node, targ_node = graph.edges()
    node_pair_distance = (cls_ctrl_feat[src_node]-cls_ctrl_feat[targ_node])*(cls_ctrl_feat[src_node]-cls_ctrl_feat[targ_node])
    # node_pair_distance = node_pair_distance.sum(dim=1) # [edge_num, feat_dim] -> [edge_num, 1]
    node_pair_CFH = base_hom[src_node] - node_pair_distance[src_node] # -> [edge_num, feat_dim]
    node_level_CFH = torch.zeros(num_node, num_feat).to(device)
    node_level_CFH = node_level_CFH.scatter_add_(0, src_node.long().unsqueeze(1).expand(-1, num_feat), node_pair_CFH)
    return node_level_CFH.mean(dim=0)