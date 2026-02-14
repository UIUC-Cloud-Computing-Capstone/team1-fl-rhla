##############
## CIFAR100 ##
##############

NCCL_DEBUG=INFO TORCH_DISTRIBUTED_DETAIL=DEBUG accelerate launch --main_process_port 29505 main.py --config_name 'experiments/cifar100_vit_lora/Exclusive/Exclusive-rank24-iid.yaml'
NCCL_DEBUG=INFO TORCH_DISTRIBUTED_DETAIL=DEBUG accelerate launch --main_process_port 29505 main.py --config_name 'experiments/cifar100_vit_lora/Exclusive/Exclusive-rank24-noniid-pat_10_dir.yaml' 
NCCL_DEBUG=INFO TORCH_DISTRIBUTED_DETAIL=DEBUG accelerate launch --main_process_port 29505 main.py --config_name 'experiments/cifar100_vit_lora/Exclusive/Exclusive-rank24-noniid-pat_20_dir.yaml'
