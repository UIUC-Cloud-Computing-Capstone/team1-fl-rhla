##############
## CIFAR100 ##
##############

# exclusive
NCCL_DEBUG=INFO TORCH_DISTRIBUTED_DETAIL=DEBUG accelerate launch --main_process_port 29505 main.py --config_name 'experiments/cifar100_vit_lora/Exclusive/Exclusive-rank24-noniid-pat_20_dir.yaml'

# fedhello
NCCL_DEBUG=INFO TORCH_DISTRIBUTED_DETAIL=DEBUG accelerate launch --main_process_port 29505 main.py --config_name 'experiments/cifar100_vit_lora/FedHello/FedHello_noniid-pat_20_dir-noprior-s50-e50.yaml' 

#fedit
NCCL_DEBUG=INFO TORCH_DISTRIBUTED_DETAIL=DEBUG accelerate launch --main_process_port 29505 main.py --config_name 'experiments/cifar100_vit_lora/FedIT/FedIT-noniid-pat_20_dir.yaml' 

#ffa-lora
NCCL_DEBUG=INFO TORCH_DISTRIBUTED_DETAIL=DEBUG accelerate launch --main_process_port 29505 main.py --config_name 'experiments/cifar100_vit_lora/FFA-LoRA/FFA-LORA-image-b-only-noniid-20.yaml' 

#LEGEND
NCCL_DEBUG=INFO TORCH_DISTRIBUTED_DETAIL=DEBUG accelerate launch --main_process_port 29505 main.py --config_name 'experiments/cifar100_vit_lora/LEGEND/LEGEND-noniid-20.yaml' 

#lokr
NCCL_DEBUG=INFO TORCH_DISTRIBUTED_DETAIL=DEBUG accelerate launch --main_process_port 29505 main.py --config_name 'experiments/cifar100_vit_lora/LOKR/LOKR-noniid-20.yaml' 

#ours
NCCL_DEBUG=INFO TORCH_DISTRIBUTED_DETAIL=DEBUG accelerate launch --main_process_port 29505 main.py --config_name 'experiments/cifar100_vit_lora/Ours/alternating-training-warm20-double-rank-int-noniid-20.yaml' 

#straggler
NCCL_DEBUG=INFO TORCH_DISTRIBUTED_DETAIL=DEBUG accelerate launch --main_process_port 29505 main.py --config_name 'experiments/cifar100_vit_lora/Straggler/straggler-rank24-noniid-pat_20_dir.yaml' 

#FlexLoRA
NCCL_DEBUG=INFO TORCH_DISTRIBUTED_DETAIL=DEBUG accelerate launch --main_process_port 29505 main.py --config_name 'experiments/cifar100_vit_lora/FlexLoRA/FlexLoRA-noniid-pat_20_dir.yaml'

#HetLoRA
NCCL_DEBUG=INFO TORCH_DISTRIBUTED_DETAIL=DEBUG accelerate launch --main_process_port 29505 main.py --config_name 'experiments/cifar100_vit_lora/HetLoRA/HetLoRA-noniid-pat_20_dir.yaml'
