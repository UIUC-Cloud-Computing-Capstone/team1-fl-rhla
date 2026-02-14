##############
## CIFAR100 ##
##############

# exclusive
NCCL_DEBUG=INFO TORCH_DISTRIBUTED_DETAIL=DEBUG accelerate launch --main_process_port 29505 main.py --config_name 'experiments/cifar100-100client-10select/Exclusive/Exclusive-rank24-iid.yaml' 

# fedhello
NCCL_DEBUG=INFO TORCH_DISTRIBUTED_DETAIL=DEBUG accelerate launch --main_process_port 29505 main.py --config_name 'experiments/cifar100-100client-10select/FedHello/FedHello_rank-24iid-noprior-s50-e50.yaml' 

#fedit
NCCL_DEBUG=INFO TORCH_DISTRIBUTED_DETAIL=DEBUG accelerate launch --main_process_port 29505 main.py --config_name 'experiments/cifar100-100client-10select/FedIT/FedIT-iid.yaml' 

#ffa-lora
NCCL_DEBUG=INFO TORCH_DISTRIBUTED_DETAIL=DEBUG accelerate launch --main_process_port 29505 main.py --config_name 'experiments/cifar100-100client-10select/FFA-LoRA/FFA-LORA-image-b-only-iid.yaml' 

#LEGEND
NCCL_DEBUG=INFO TORCH_DISTRIBUTED_DETAIL=DEBUG accelerate launch --main_process_port 29505 main.py --config_name 'experiments/cifar100-100client-10select/LEGEND/LEGEND-iid.yaml' 

#lokr
NCCL_DEBUG=INFO TORCH_DISTRIBUTED_DETAIL=DEBUG accelerate launch --main_process_port 29505 main.py --config_name 'experiments/cifar100-100client-10select/LOKR/LOKR-iid.yaml' 

#ours
NCCL_DEBUG=INFO TORCH_DISTRIBUTED_DETAIL=DEBUG accelerate launch --main_process_port 29505 main.py --config_name 'experiments/cifar100-100client-10select/Ours/alternating-training-warm20-double-rank-int.yaml' 

#straggler
NCCL_DEBUG=INFO TORCH_DISTRIBUTED_DETAIL=DEBUG accelerate launch --main_process_port 29505 main.py --config_name 'experiments/cifar100-100client-10select/Straggler/straggler-rank24_iid.yaml' 

