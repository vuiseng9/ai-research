#!/usr/bin/env bash

export OMP_NUM_THREADS=64

#python -m torch.distributed.launch --nproc_per_node=$NUM_GPU main.py --network_name mobilenet_v1 --lr_start 0.000005 --gamma 64 --lr_coefficient 0.005 --batch_size 256 --n_epochs 30 --target_compression 8.0 --fp16

python -m torch.distributed.launch --nproc_per_node 4 main.py --network_name mobilenet_v1 --lr_start 0.000005 --gamma 64 --lr_coefficient 0.005 --batch_size 256 --n_epochs 30 --target_compression 8.0 --fp16

