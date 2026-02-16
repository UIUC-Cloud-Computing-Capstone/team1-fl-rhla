##############
## CIFAR100 ##
##############

# rank8
NCCL_DEBUG=INFO TORCH_DISTRIBUTED_DETAIL=DEBUG accelerate launch --main_process_port 29505 main.py --config_name 'experiments/ablation/iid-cifar/alternating-training-warm20-double-rank-int-rank8.yaml'

# rank16
NCCL_DEBUG=INFO TORCH_DISTRIBUTED_DETAIL=DEBUG accelerate launch --main_process_port 29505 main.py --config_name 'experiments/ablation/iid-cifar/alternating-training-warm20-double-rank-int-rank16.yaml'

# rank24
NCCL_DEBUG=INFO TORCH_DISTRIBUTED_DETAIL=DEBUG accelerate launch --main_process_port 29505 main.py --config_name 'experiments/ablation/iid-cifar/alternating-training-warm20-double-rank-int-rank24.yaml'

# rank32
NCCL_DEBUG=INFO TORCH_DISTRIBUTED_DETAIL=DEBUG accelerate launch --main_process_port 29505 main.py --config_name 'experiments/ablation/iid-cifar/alternating-training-warm20-double-rank-int-rank32.yaml'

# nosvd
NCCL_DEBUG=INFO TORCH_DISTRIBUTED_DETAIL=DEBUG accelerate launch --main_process_port 29505 main.py --config_name 'experiments/ablation/iid-cifar/alternating-training-warm20-double-rank-int-rank24-nosvd.yaml'

# no alternating
NCCL_DEBUG=INFO TORCH_DISTRIBUTED_DETAIL=DEBUG accelerate launch --main_process_port 29505 main.py --config_name 'experiments/ablation/iid-cifar/alternating-training-warm20-double-rank-int-rank24-noalternating.yaml'

# no rank selection
NCCL_DEBUG=INFO TORCH_DISTRIBUTED_DETAIL=DEBUG accelerate launch --main_process_port 29505 main.py --config_name 'experiments/ablation/iid-cifar/alternating-training-warm20-double-rank-int-rank24-norankselection.yaml'

