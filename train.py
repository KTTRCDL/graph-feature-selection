import argparse
from tqdm import tqdm

import torch
# from torch.cuda.amp import autocast, GradScaler
# future pytorch version
from torch.amp import autocast, GradScaler

from model import Model
from datasets import Dataset
from utils import Logger, get_parameter_groups, get_lr_scheduler_with_warmup
import copy


def get_args():
    parser = argparse.ArgumentParser()

    parser.add_argument('--name', type=str, default=None, help='Experiment name. If None, model name is used.')
    parser.add_argument('--save_dir', type=str, default='experiments', help='Base directory for saving information.')
    parser.add_argument('--dataset', type=str, default='Children',
                        choices=['roman-empire', 'amazon-ratings', 'minesweeper', 'tolokers', 'questions',
                                 'squirrel-filtered', 'chameleon-filtered', 
                                 'actor', 'texas', 'texas-4-classes', 'cornell', 'wisconsin',
                                 'Children', 'Computers', 'Fitness', 'History', 'Photo',
                                 "cora", "pubmed", "citeseer"])

    # feature selection
    parser.add_argument('--feature_norm_fn', type=str, default='z-norm', choices=['None', 'z-norm', 'torch-norm'], help='Feature normalization function')
    parser.add_argument('--node_selection_fn_name', type=str, default='mi_agg', help='Feature selection function name',
                        choices=['None', 'mi_agg', 'h_GE', 'h_attr', 'h_sim-cos', 'h_sim-euc', 'h_CTF'])
    parser.add_argument('--ratio', type=float, default=0.5, 
                        help='split ratio for feature selection, ratio feature will deliver to GNN, if raio=1.0, use GNN model')
    parser.add_argument('--node_selection_with_train_idx', action='store_true', default=True, help='use train_idx of split to select node features')

    # model architecture
    parser.add_argument('--model', type=str, default='GCN-MLP',
                        choices=['GCN-MLP', 'GAT-MLP', 'SAGE-MLP', 'GT-MLP', 'SGC-MLP', 'APPNP-MLP', 'ACMGCN-MLP', 'FAGCN-MLP'])
    parser.add_argument('--num_layers', type=int, default=2)
    parser.add_argument('--num_layers_1', type=int, default=3)
    parser.add_argument('--hidden_dim', type=int, default=512)
    parser.add_argument('--hidden_dim_1', type=int, default=512)
    parser.add_argument('--hidden_dim_multiplier', type=float, default=1)
    parser.add_argument('--num_heads', type=int, default=8)
    parser.add_argument('--normalization', type=str, default='LayerNorm', choices=['None', 'LayerNorm', 'BatchNorm'])
    parser.add_argument('--graph_self_loop', default=True, action='store_true')

    # regularization
    parser.add_argument('--dropout', type=float, default=0.2)
    parser.add_argument('--weight_decay', type=float, default=0)

    # training parameters
    parser.add_argument('--lr', type=float, default=3e-5)
    parser.add_argument('--num_steps', type=int, default=1000)
    parser.add_argument('--num_warmup_steps', type=int, default=None,
                        help='If None, warmup_proportion is used instead.')
    parser.add_argument('--warmup_proportion', type=float, default=0, help='Only used if num_warmup_steps is None.')

    parser.add_argument('--num_runs', type=int, default=10)
    parser.add_argument('--device', type=str, default='cuda:0')
    parser.add_argument('--amp', default=False, action='store_true')
    parser.add_argument('--verbose', default=False, action='store_true')

    args = parser.parse_args()

    if args.name is None:
        args.name = args.model

    return args


def train_step(model, dataset: Dataset, optimizer, scheduler, scaler, amp=False):
    model.train()

    # with autocast(enabled=amp):
    # future pytorch version
    with autocast('cuda', enabled=amp):
        logits = model(graph=dataset.graph, x=dataset.node_features, x_1=dataset.node_features_1)
        loss = dataset.loss_fn(input=logits[dataset.train_idx], target=dataset.labels[dataset.train_idx])

    scaler.scale(loss).backward()
    scaler.step(optimizer)
    scaler.update()
    optimizer.zero_grad()
    scheduler.step()


@torch.no_grad()
def evaluate(model, dataset, amp=False):
    model.eval()

    # with autocast(enabled=amp):
    # future pytorch version
    with autocast('cuda',enabled=amp):
        if not hasattr(dataset, 'node_features_1'):
            logits = model(graph=dataset.graph, x=dataset.node_features)
        else:
            logits = model(graph=dataset.graph, x=dataset.node_features, x_1=dataset.node_features_1)

    metrics = dataset.compute_metrics(logits)

    return metrics


def main():
    args = get_args()

    torch.manual_seed(0)

    if args.node_selection_fn_name == 'None':
        dataset = Dataset(name=args.dataset,
                    add_self_loops=(args.graph_self_loop),
                    device=args.device,
                    node_selection_fn_name='None',
                    node_feature_norm=args.feature_norm_fn,
                    node_selection_with_train_idx=args.node_selection_with_train_idx,
                    ratio=1.0, save_dir=f"TFI/")
    else:
        dataset = Dataset(name=args.dataset,
                        add_self_loops=(args.graph_self_loop),
                        device=args.device,
                        node_selection_fn_name=args.node_selection_fn_name,
                        node_feature_norm=args.feature_norm_fn,
                        node_selection_with_train_idx=args.node_selection_with_train_idx,
                        ratio=args.ratio, save_dir=f"TFI/")

    args_log = argparse.Namespace(**vars(args))
    if args.node_selection_fn_name != 'None' and args.ratio != 1.0 and args.ratio != 0.0:
        args_log.model = args.model
    elif (args.node_selection_fn_name == 'None' and args.ratio != 0.0) or args.ratio == 1.0:
        args_log.model = f"{args.model.split('-')[0]}"
    else:
        args_log.model = f"{args.model.split('-')[1]}"
    logger = Logger(args_log, metric=dataset.metric, num_data_splits=dataset.num_data_splits)

    for run in range(1, args.num_runs + 1):
        model = Model(model_name=args.model,
                    num_layers=args.num_layers,
                    num_layers_1=args.num_layers_1,
                    input_dim=dataset.num_node_features,
                    input_dim_1=dataset.num_node_features_1,
                    hidden_dim=args.hidden_dim,
                    hidden_dim_1=args.hidden_dim_1,
                    output_dim=dataset.num_targets,
                    hidden_dim_multiplier=args.hidden_dim_multiplier,
                    num_heads=args.num_heads,
                    normalization=args.normalization,
                    dropout=args.dropout)
        model.to(args.device)

        parameter_groups = get_parameter_groups(model)
        optimizer = torch.optim.AdamW(parameter_groups, lr=args.lr, weight_decay=args.weight_decay)
        scaler = GradScaler(enabled=args.amp)
        scheduler = get_lr_scheduler_with_warmup(optimizer=optimizer, num_warmup_steps=args.num_warmup_steps,
                                                 num_steps=args.num_steps, warmup_proportion=args.warmup_proportion)

        logger.start_run(run=run, data_split=dataset.cur_data_split + 1)
        with tqdm(total=args.num_steps, desc=f'Run {run}', disable=args.verbose) as progress_bar:
            for step in range(1, args.num_steps + 1):
                train_step(model=model, dataset=dataset, optimizer=optimizer, scheduler=scheduler,
                           scaler=scaler, amp=args.amp)
                metrics = evaluate(model=model, dataset=dataset, amp=args.amp)
                logger.update_metrics(metrics=metrics, step=step)

                progress_bar.update()
                progress_bar.set_postfix({metric: f'{value:.2f}' for metric, value in metrics.items()})

        logger.finish_run()
        model.cpu()
        dataset.next_data_split()

    logger.print_metrics_summary()


if __name__ == '__main__':
    main()
