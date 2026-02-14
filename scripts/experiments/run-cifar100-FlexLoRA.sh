##############
## CIFAR100 ##
##############

NCCL_DEBUG=INFO TORCH_DISTRIBUTED_DETAIL=DEBUG accelerate launch --main_process_port 29505 main.py --config_name 'experiments/cifar100_vit_lora/FlexLoRA/FlexLoRA-iid.yaml'
NCCL_DEBUG=INFO TORCH_DISTRIBUTED_DETAIL=DEBUG accelerate launch --main_process_port 29505 main.py --config_name 'experiments/cifar100_vit_lora/FlexLoRA/FlexLoRA-noniid-pat_10_dir.yaml' 
NCCL_DEBUG=INFO TORCH_DISTRIBUTED_DETAIL=DEBUG accelerate launch --main_process_port 29505 main.py --config_name 'experiments/cifar100_vit_lora/FlexLoRA/FlexLoRA-noniid-pat_20_dir.yaml'

NCCL_DEBUG=INFO TORCH_DISTRIBUTED_DETAIL=DEBUG accelerate launch --main_process_port 29505 main.py --config_name 'experiments/cifar100-100client-10select/FlexLoRA/FlexLoRA-iid.yaml'
NCCL_DEBUG=INFO TORCH_DISTRIBUTED_DETAIL=DEBUG accelerate launch --main_process_port 29505 main.py --config_name 'experiments/cifar100-100client-10select/FlexLoRA/FlexLoRA-noniid-pat_10_dir.yaml' 
NCCL_DEBUG=INFO TORCH_DISTRIBUTED_DETAIL=DEBUG accelerate launch --main_process_port 29505 main.py --config_name 'experiments/cifar100-100client-10select/FlexLoRA/FlexLoRA-noniid-pat_20_dir.yaml'

NCCL_DEBUG=INFO TORCH_DISTRIBUTED_DETAIL=DEBUG accelerate launch --main_process_port 29505 main.py --config_name 'experiments/ledgar_bert_lora/FlexLoRA/FlexLoRA-iid.yaml'
NCCL_DEBUG=INFO TORCH_DISTRIBUTED_DETAIL=DEBUG accelerate launch --main_process_port 29505 main.py --config_name 'experiments/ledgar_bert_lora/FlexLoRA/FlexLoRA-noniid-pat_10_dir.yaml' 
NCCL_DEBUG=INFO TORCH_DISTRIBUTED_DETAIL=DEBUG accelerate launch --main_process_port 29505 main.py --config_name 'experiments/ledgar_bert_lora/FlexLoRA/FlexLoRA-noniid-pat_20_dir.yaml'
