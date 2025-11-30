##############
## CIFAR100 ##
##############

NCCL_DEBUG=INFO TORCH_DISTRIBUTED_DETAIL=DEBUG accelerate launch --main_process_port 29505 main.py --config_name 'experiments/cifar100_vit_lora/FFA-LoRA/FFA-LORA-image-b-only-iid.yaml'
NCCL_DEBUG=INFO TORCH_DISTRIBUTED_DETAIL=DEBUG accelerate launch --main_process_port 29505 main.py --config_name 'experiments/cifar100_vit_lora/FFA-LoRA/FFA-LORA-image-b-only-noniid-10.yaml' 
NCCL_DEBUG=INFO TORCH_DISTRIBUTED_DETAIL=DEBUG accelerate launch --main_process_port 29505 main.py --config_name 'experiments/cifar100_vit_lora/FFA-LoRA/FFA-LORA-image-b-only-noniid-20.yaml'
