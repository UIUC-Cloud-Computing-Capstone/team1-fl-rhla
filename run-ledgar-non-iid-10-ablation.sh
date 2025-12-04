##############
## ledgar ##
##############

# rank16
NCCL_DEBUG=INFO TORCH_DISTRIBUTED_DETAIL=DEBUG accelerate launch --main_process_port 29505 main.py --config_name 'experiments/ablation/noniid-10-ledgar/alternating-training-warm20-double-rank-int-no-rank-vary-rank16.yaml'

# rank24
NCCL_DEBUG=INFO TORCH_DISTRIBUTED_DETAIL=DEBUG accelerate launch --main_process_port 29505 main.py --config_name 'experiments/ablation/noniid-10-ledgar/alternating-training-warm20-double-rank-int-no-rank-vary-rank24.yaml'

# nosvd
NCCL_DEBUG=INFO TORCH_DISTRIBUTED_DETAIL=DEBUG accelerate launch --main_process_port 29505 main.py --config_name 'experiments/ablation/noniid-10-ledgar/alternating-training-warm20-double-rank-int-no-rank-vary-rank24-nosvd.yaml'

# no alternating
NCCL_DEBUG=INFO TORCH_DISTRIBUTED_DETAIL=DEBUG accelerate launch --main_process_port 29505 main.py --config_name 'experiments/ablation/noniid-10-ledgar/alternating-training-warm20-double-rank-int-no-rank-vary-rank24-noalternating.yaml'

# no rank selection
NCCL_DEBUG=INFO TORCH_DISTRIBUTED_DETAIL=DEBUG accelerate launch --main_process_port 29505 main.py --config_name 'experiments/ablation/noniid-10-ledgar/alternating-training-warm20-double-rank-int-no-rank-vary-rank24-norankselection.yaml'

# rank8
NCCL_DEBUG=INFO TORCH_DISTRIBUTED_DETAIL=DEBUG accelerate launch --main_process_port 29505 main.py --config_name 'experiments/ablation/noniid-10-ledgar/alternating-training-warm20-double-rank-int-no-rank-vary-rank8.yaml'

# rank32
NCCL_DEBUG=INFO TORCH_DISTRIBUTED_DETAIL=DEBUG accelerate launch --main_process_port 29505 main.py --config_name 'experiments/ablation/noniid-10-ledgar/alternating-training-warm20-double-rank-int-no-rank-vary-rank32.yaml'
