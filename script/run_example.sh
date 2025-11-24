CUDA_VISIBLE_DEVICES=0
ratio=0.5

# To run a MLP model on dataset Children
python train.py --name MLP --dataset Children --ratio 0.0 --device cuda:$CUDA_VISIBLE_DEVICES

# To run a GCN+GFS model on dataset Children with ratio 0.5
python train.py --name GCN-MLP_$ratio --dataset Children --ratio $ratio --device cuda:$CUDA_VISIBLE_DEVICES

# To run a GCN model on dataset Children
python train.py --name GCN --dataset Children --ratio 1.0 --device cuda:$CUDA_VISIBLE_DEVICES