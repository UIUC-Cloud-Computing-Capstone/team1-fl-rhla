##############
## CIFAR100 ##
##############

# exclusive
NCCL_DEBUG=INFO TORCH_DISTRIBUTED_DETAIL=DEBUG accelerate launch --main_process_port 29505 main.py --config_name 'experiments/cifar100-100client-10select/Exclusive/Exclusive-rank24-noniid-pat_10_dir.yaml' 

# fedhello
NCCL_DEBUG=INFO TORCH_DISTRIBUTED_DETAIL=DEBUG accelerate launch --main_process_port 29505 main.py --config_name 'experiments/cifar100-100client-10select/FedHello/FedHello_noniid-pat_10_dir-noprior-s50-e50.yaml' 

#fedit
NCCL_DEBUG=INFO TORCH_DISTRIBUTED_DETAIL=DEBUG accelerate launch --main_process_port 29505 main.py --config_name 'experiments/cifar100-100client-10select/FedIT/FedIT-noniid-pat_10_dir.yaml' 

#ffa-lora
NCCL_DEBUG=INFO TORCH_DISTRIBUTED_DETAIL=DEBUG accelerate launch --main_process_port 29505 main.py --config_name 'experiments/cifar100-100client-10select/FFA-LoRA/FFA-LORA-image-b-only-noniid-10.yaml' 

#LEGEND
NCCL_DEBUG=INFO TORCH_DISTRIBUTED_DETAIL=DEBUG accelerate launch --main_process_port 29505 main.py --config_name 'experiments/cifar100-100client-10select/LEGEND/LEGEND-noniid-10.yaml' 

#lokr
NCCL_DEBUG=INFO TORCH_DISTRIBUTED_DETAIL=DEBUG accelerate launch --main_process_port 29505 main.py --config_name 'experiments/cifar100-100client-10select/LOKR/LOKR-noniid-10.yaml' 

#ours
NCCL_DEBUG=INFO TORCH_DISTRIBUTED_DETAIL=DEBUG accelerate launch --main_process_port 29505 main.py --config_name 'experiments/cifar100-100client-10select/Ours/alternating-training-warm20-double-rank-int-noniid-10.yaml' 

#straggler
NCCL_DEBUG=INFO TORCH_DISTRIBUTED_DETAIL=DEBUG accelerate launch --main_process_port 29505 main.py --config_name 'experiments/cifar100-100client-10select/Straggler/straggler-rank24-noniid-pat_10_dir.yaml' 

#FlexLoRA
NCCL_DEBUG=INFO TORCH_DISTRIBUTED_DETAIL=DEBUG accelerate launch --main_process_port 29505 main.py --config_name 'experiments/cifar100-100client-10select/FlexLoRA/FlexLoRA-noniid-pat_10_dir.yaml' 

#HetLoRA
NCCL_DEBUG=INFO TORCH_DISTRIBUTED_DETAIL=DEBUG accelerate launch --main_process_port 29505 main.py --config_name 'experiments/cifar100-100client-10select/HetLoRA/HetLoRA-noniid-pat_10_dir.yaml' 
