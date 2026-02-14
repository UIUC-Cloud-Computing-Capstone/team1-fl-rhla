##############
## CIFAR100 ##
##############

# 0.0
NCCL_DEBUG=INFO TORCH_DISTRIBUTED_DETAIL=DEBUG accelerate launch --main_process_port 29505 main.py --config_name 'experiments/ablation/diff-alpha-cifar100-non-iid-20/alternating-training-warm20-double-rank-int-no-rank-vary-rank24-0dot0.yaml'

# 0.2
NCCL_DEBUG=INFO TORCH_DISTRIBUTED_DETAIL=DEBUG accelerate launch --main_process_port 29505 main.py --config_name 'experiments/ablation/diff-alpha-cifar100-non-iid-20/alternating-training-warm20-double-rank-int-no-rank-vary-rank24-0dot2.yaml'

# 0.4
NCCL_DEBUG=INFO TORCH_DISTRIBUTED_DETAIL=DEBUG accelerate launch --main_process_port 29505 main.py --config_name 'experiments/ablation/diff-alpha-cifar100-non-iid-20/alternating-training-warm20-double-rank-int-no-rank-vary-rank24-0dot4.yaml'

# 0.6
NCCL_DEBUG=INFO TORCH_DISTRIBUTED_DETAIL=DEBUG accelerate launch --main_process_port 29505 main.py --config_name 'experiments/ablation/diff-alpha-cifar100-non-iid-20/alternating-training-warm20-double-rank-int-no-rank-vary-rank24-0dot6.yaml'

# 0.8
NCCL_DEBUG=INFO TORCH_DISTRIBUTED_DETAIL=DEBUG accelerate launch --main_process_port 29505 main.py --config_name 'experiments/ablation/diff-alpha-cifar100-non-iid-20/alternating-training-warm20-double-rank-int-no-rank-vary-rank24-0dot8.yaml'

# 1.0
NCCL_DEBUG=INFO TORCH_DISTRIBUTED_DETAIL=DEBUG accelerate launch --main_process_port 29505 main.py --config_name 'experiments/ablation/diff-alpha-cifar100-non-iid-20/alternating-training-warm20-double-rank-int-no-rank-vary-rank24-1dot0.yaml'