##############
## CIFAR100 ##
##############

NCCL_DEBUG=INFO TORCH_DISTRIBUTED_DETAIL=DEBUG accelerate launch --main_process_port 29505 main.py --config_name 'experiments/cifar100_vit_lora/FedHello/FedHello_rank-24iid-noprior-s50-e50.yaml'
NCCL_DEBUG=INFO TORCH_DISTRIBUTED_DETAIL=DEBUG accelerate launch --main_process_port 29505 main.py --config_name 'experiments/cifar100_vit_lora/FedHello/FedHello_noniid-pat_10_dir-noprior-s50-e50.yaml' 
NCCL_DEBUG=INFO TORCH_DISTRIBUTED_DETAIL=DEBUG accelerate launch --main_process_port 29505 main.py --config_name 'experiments/cifar100_vit_lora/FedHello/FedHello_noniid-pat_20_dir-noprior-s50-e50.yaml'
