##############
## ledgar ##
##############

#cifar100 iid
# rank16
NCCL_DEBUG=INFO TORCH_DISTRIBUTED_DETAIL=DEBUG accelerate launch --main_process_port 29505 main.py --config_name 'experiments/ablation/iid/alternating-training-warm20-double-rank-int-no-rank-vary-rank8.yaml'

# rank24
NCCL_DEBUG=INFO TORCH_DISTRIBUTED_DETAIL=DEBUG accelerate launch --main_process_port 29505 main.py --config_name 'experiments/ablation/iid/alternating-training-warm20-double-rank-int-no-rank-vary-rank16.yaml'

# rank32
NCCL_DEBUG=INFO TORCH_DISTRIBUTED_DETAIL=DEBUG accelerate launch --main_process_port 29505 main.py --config_name 'experiments/ablation/iid/alternating-training-warm20-double-rank-int-no-rank-vary-rank32.yaml'

#cifar100 non-iid-10
# rank16
NCCL_DEBUG=INFO TORCH_DISTRIBUTED_DETAIL=DEBUG accelerate launch --main_process_port 29505 main.py --config_name 'experiments/ablation/noniid-10-cifar100/alternating-training-warm20-double-rank-int-no-rank-vary-rank8.yaml'

# rank24
NCCL_DEBUG=INFO TORCH_DISTRIBUTED_DETAIL=DEBUG accelerate launch --main_process_port 29505 main.py --config_name 'experiments/ablation/noniid-10-cifar100/alternating-training-warm20-double-rank-int-no-rank-vary-rank16.yaml'

# rank32
NCCL_DEBUG=INFO TORCH_DISTRIBUTED_DETAIL=DEBUG accelerate launch --main_process_port 29505 main.py --config_name 'experiments/ablation/noniid-10-cifar100/alternating-training-warm20-double-rank-int-no-rank-vary-rank32.yaml'

#cifar100 non-iid-20
# rank16
NCCL_DEBUG=INFO TORCH_DISTRIBUTED_DETAIL=DEBUG accelerate launch --main_process_port 29505 main.py --config_name 'experiments/ablation/noniid-20-cifar100/alternating-training-warm20-double-rank-int-no-rank-vary-rank8.yaml'

# rank24
NCCL_DEBUG=INFO TORCH_DISTRIBUTED_DETAIL=DEBUG accelerate launch --main_process_port 29505 main.py --config_name 'experiments/ablation/noniid-20-cifar100/alternating-training-warm20-double-rank-int-no-rank-vary-rank16.yaml'

# rank32
NCCL_DEBUG=INFO TORCH_DISTRIBUTED_DETAIL=DEBUG accelerate launch --main_process_port 29505 main.py --config_name 'experiments/ablation/noniid-20-cifar100/alternating-training-warm20-double-rank-int-no-rank-vary-rank32.yaml'

#ledgar non-iid-10
# rank16
NCCL_DEBUG=INFO TORCH_DISTRIBUTED_DETAIL=DEBUG accelerate launch --main_process_port 29505 main.py --config_name 'experiments/ablation/noniid-10-ledgar/alternating-training-warm20-double-rank-int-no-rank-vary-rank8.yaml'

# rank24
NCCL_DEBUG=INFO TORCH_DISTRIBUTED_DETAIL=DEBUG accelerate launch --main_process_port 29505 main.py --config_name 'experiments/ablation/noniid-10-ledgar/alternating-training-warm20-double-rank-int-no-rank-vary-rank16.yaml'

# rank32
NCCL_DEBUG=INFO TORCH_DISTRIBUTED_DETAIL=DEBUG accelerate launch --main_process_port 29505 main.py --config_name 'experiments/ablation/noniid-10-ledgar/alternating-training-warm20-double-rank-int-no-rank-vary-rank32.yaml'

#ledgar non-iid-20
# rank16
NCCL_DEBUG=INFO TORCH_DISTRIBUTED_DETAIL=DEBUG accelerate launch --main_process_port 29505 main.py --config_name 'experiments/ablation/noniid-20-ledgar/alternating-training-warm20-double-rank-int-no-rank-vary-rank8.yaml'

# rank24
NCCL_DEBUG=INFO TORCH_DISTRIBUTED_DETAIL=DEBUG accelerate launch --main_process_port 29505 main.py --config_name 'experiments/ablation/noniid-20-ledgar/alternating-training-warm20-double-rank-int-no-rank-vary-rank16.yaml'

# rank32
NCCL_DEBUG=INFO TORCH_DISTRIBUTED_DETAIL=DEBUG accelerate launch --main_process_port 29505 main.py --config_name 'experiments/ablation/noniid-20-ledgar/alternating-training-warm20-double-rank-int-no-rank-vary-rank32.yaml'



