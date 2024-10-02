import os
import numpy as np
import torch
from torch.nn import functional as F
import dgl
from sklearn.metrics import roc_auc_score
from utils import z_norm, mi_agg

SELECTION_FUNCTION = {
    'mi_agg': mi_agg,
}

NORMALIZATION = {
    'z-norm': z_norm,
    'torch-norm': torch.norm
}
class Dataset:
    def __init__(self, name, add_self_loops=False, device='cpu', node_selection_with_train_idx=True,
                 node_feature_norm='None', node_selection_fn_name='None', ratio=1.0, save_dir="TFI/"):
        print('Preparing data...')
        data = np.load(os.path.join('data', f'{name.replace("-", "_")}.npz'))
        node_features = torch.tensor(data['node_features'])
        labels = torch.tensor(data['node_labels'])
        edges = torch.tensor(data['edges'])

        graph = dgl.graph((edges[:, 0], edges[:, 1]), num_nodes=len(node_features), idtype=torch.int)
        print("graph constructed.")

        if 'directed' not in name:
            graph = dgl.to_bidirected(graph)

        # feature norm, z-norm, torch.norm (default z-norm)
        if node_feature_norm != 'None':
            node_features = NORMALIZATION[node_feature_norm](node_features)
        
        train_masks = torch.tensor(data['train_masks'])
        val_masks = torch.tensor(data['val_masks'])
        test_masks = torch.tensor(data['test_masks'])

        train_idx_list = [torch.where(train_mask)[0] for train_mask in train_masks]
        val_idx_list = [torch.where(val_mask)[0] for val_mask in val_masks]
        test_idx_list = [torch.where(test_mask)[0] for test_mask in test_masks]

        print("train, val, test split loaded.")
        
        self.ori_node_features = node_features
        # save homophily feature selection degree
        self.save_dir = f"data/{save_dir}"
        if not os.path.exists(self.save_dir):
            os.makedirs(self.save_dir, exist_ok=True)
        # feature selection and feature split
        if node_selection_fn_name != 'None' and node_selection_fn_name in SELECTION_FUNCTION and not node_selection_with_train_idx:
            degree_path = os.path.join(self.save_dir, f'{name.replace("-", "_")}_{node_selection_fn_name}_{node_feature_norm}.npy')
            if os.path.exists(degree_path):
                degree = np.load(degree_path)
            else:
                degree = Dataset.get_tfi_of_node_features(node_selection_fn_name, graph, self.ori_node_features, labels)
                np.save(degree_path, degree)
            node_features = Dataset.sort_node_features_by_tfi(node_features, degree)
            print("node feature selection done.")
        elif node_selection_fn_name != 'None' and node_selection_fn_name in SELECTION_FUNCTION:
            # feature split with mask, begin with self.cur_data_split = 0 -> mask_0
            degree_path = os.path.join(self.save_dir, f'{name.replace("-", "_")}_{node_selection_fn_name}_{node_feature_norm}_mask_0.npy')
            if os.path.exists(degree_path):
                degree = np.load(degree_path)
            else:
                degree = Dataset.get_tfi_of_node_features(node_selection_fn_name, graph, self.ori_node_features, labels, train_idx_list[0])
                np.save(degree_path, degree)
            node_features = Dataset.sort_node_features_by_tfi(node_features, degree)
            print("node feature selection with train idx done.")

        self.node_selection_with_train_idx = node_selection_with_train_idx
        self.node_selection_fn_name = node_selection_fn_name
        self.node_feature_norm = node_feature_norm

        if add_self_loops:
            graph = dgl.add_self_loop(graph)

        num_classes = len(labels.unique())
        num_targets = 1 if num_classes == 2 else num_classes
        if num_targets == 1:
            labels = labels.float()

        node_features, node_features_1 = Dataset.split_node_features_by_ratio(node_features, ratio)
        print("node feature split done.")

        self.name = name
        self.device = device
        self.ratio = ratio
        self.graph = graph.to(device)
        self.node_features = node_features.to(device) if node_features.shape[1] != 0 else None
        self.node_features_1 = node_features_1.to(device) if node_features_1.shape[1] != 0 else None
        self.labels = labels.to(device)

        self.train_idx_list = [train_idx.to(device) for train_idx in train_idx_list]
        self.val_idx_list = [val_idx.to(device) for val_idx in val_idx_list]
        self.test_idx_list = [test_idx.to(device) for test_idx in test_idx_list]
        self.num_data_splits = len(train_idx_list)
        self.cur_data_split = 0

        self.num_node_features = node_features.shape[1] if node_features is not None else 0
        self.num_node_features_1 = node_features_1.shape[1] if node_features_1 is not None else 0
        self.num_targets = num_targets

        self.loss_fn = F.binary_cross_entropy_with_logits if num_targets == 1 else F.cross_entropy
        self.metric = 'ROC AUC' if num_targets == 1 else 'accuracy'

        print('Dataset initialized.')

    @property
    def train_idx(self):
        return self.train_idx_list[self.cur_data_split]

    @property
    def val_idx(self):
        return self.val_idx_list[self.cur_data_split]

    @property
    def test_idx(self):
        return self.test_idx_list[self.cur_data_split]

    def next_data_split(self):
        self.cur_data_split = (self.cur_data_split + 1) % self.num_data_splits
        if self.node_selection_fn_name != 'None' and self.node_selection_fn_name in SELECTION_FUNCTION and self.node_selection_with_train_idx:
            degree_path = os.path.join(self.save_dir, f'{self.name.replace("-", "_")}_{self.node_selection_fn_name}_{self.node_feature_norm}_mask_{self.cur_data_split}.npy')
            if os.path.exists(degree_path):
                degree = np.load(degree_path)
            else:
                degree = Dataset.get_tfi_of_node_features(self.node_selection_fn_name, self.graph.to("cpu"), self.ori_node_features, self.labels, train_idx=self.train_idx_list[self.cur_data_split])
                np.save(degree_path, degree)
            node_features = Dataset.sort_node_features_by_tfi(self.ori_node_features, degree)
            print("node feature selection with train idx done.")
        node_features, node_features_1 = Dataset.split_node_features_by_ratio(node_features, self.ratio)
        self.node_features = node_features.to(self.device) if node_features.shape[1] != 0 else None
        self.node_features_1 = node_features_1.to(self.device) if node_features_1.shape[1] != 0 else None

    def compute_metrics(self, logits):
        if self.num_targets == 1:
            train_metric = roc_auc_score(y_true=self.labels[self.train_idx].cpu().numpy(),
                                         y_score=logits[self.train_idx].cpu().numpy()).item()

            val_metric = roc_auc_score(y_true=self.labels[self.val_idx].cpu().numpy(),
                                       y_score=logits[self.val_idx].cpu().numpy()).item()

            test_metric = roc_auc_score(y_true=self.labels[self.test_idx].cpu().numpy(),
                                        y_score=logits[self.test_idx].cpu().numpy()).item()

        else:
            preds = logits.argmax(axis=1)
            train_metric = (preds[self.train_idx] == self.labels[self.train_idx]).float().mean().item()
            val_metric = (preds[self.val_idx] == self.labels[self.val_idx]).float().mean().item()
            test_metric = (preds[self.test_idx] == self.labels[self.test_idx]).float().mean().item()

        metrics = {
            f'train {self.metric}': train_metric,
            f'val {self.metric}': val_metric,
            f'test {self.metric}': test_metric
        }

        return metrics

    @staticmethod
    def get_tfi_of_node_features(node_selection_fn_name, graph, node_features, ori_labels, train_idx=None):
        node_selection_fn = SELECTION_FUNCTION[node_selection_fn_name]
        if train_idx is not None:
            train_idx = train_idx.cpu()
        degree = node_selection_fn(graph, node_features, ori_labels, train_idx)
        return degree

    @staticmethod
    def sort_node_features_by_tfi(node_features, degree):
        degree = np.array(degree)
        sorted_indices = np.argsort(degree)[::-1]
        # At least one stride in the given numpy array is negative, 
        # and tensors with negative strides are not currently supported. 
        # (You can probably work around this by making a copy of your array with array.copy().
        sorted_node_features = node_features[:, sorted_indices.copy()]
        return sorted_node_features

    @staticmethod
    def split_node_features_by_ratio(node_features, ratio=1.0):
        assert 0 <= ratio <= 1., 'Ratio should be in [0, 1].'
        n_feature = node_features.shape[1]
        split = int(n_feature * ratio)
        return torch.split(node_features, [split, n_feature - split], dim=1)
