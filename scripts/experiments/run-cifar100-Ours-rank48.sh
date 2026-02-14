##############
## CIFAR100 ##
##############

NCCL_DEBUG=INFO TORCH_DISTRIBUTED_DETAIL=DEBUG accelerate launch --main_process_port 29505 main.py --config_name 'experiments/cifar100_vit_lora/Ours/alternating-training-warm20-double-rank-int-rank48.yaml'
NCCL_DEBUG=INFO TORCH_DISTRIBUTED_DETAIL=DEBUG accelerate launch --main_process_port 29505 main.py --config_name 'experiments/cifar100_vit_lora/Ours/alternating-training-warm20-double-rank-int-no-rank-vary-noniid-10-rank48.yaml' 
NCCL_DEBUG=INFO TORCH_DISTRIBUTED_DETAIL=DEBUG accelerate launch --main_process_port 29505 main.py --config_name 'experiments/cifar100_vit_lora/Ours/alternating-training-warm20-double-rank-int-no-rank-vary-noniid-20-rank48.yaml'
