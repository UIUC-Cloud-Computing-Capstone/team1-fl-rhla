##############
## CIFAR100 ##
##############

#NCCL_DEBUG=INFO TORCH_DISTRIBUTED_DETAIL=DEBUG accelerate launch --main_process_port 29505 main.py --config_name 'experiments/cifar100_vit_lora/Ours/image-rank-var-b-noniid-10.yaml'
# Similar for other datasets ....


# Fed-HeLLo-Bone
#NCCL_DEBUG=INFO TORCH_DISTRIBUTED_DETAIL=DEBUG accelerate launch --main_process_port 29505 main.py --config_name 'experiments/cifar100_vit_lora/depthffm_fim/image_cifar100_vit_fedavg_depthffm_fim-6_9_12-bone_noniid-pat_10_dir-noprior-s50-e50.yaml' # Fed-HELLo Bone
#NCCL_DEBUG=INFO TORCH_DISTRIBUTED_DETAIL=DEBUG accelerate launch --main_process_port 29505 main.py --config_name 'experiments/cifar100_vit_lora/depthffm_fim/image_cifar100_vit_fedavg_depthffm_fim-6_9_12-bone_iid-noprior-s50-e50.yaml' # Fed-HELLo Bone
#NCCL_DEBUG=INFO TORCH_DISTRIBUTED_DETAIL=DEBUG accelerate launch --main_process_port 29505 main.py --config_name 'experiments/cifar100_vit_lora/depthffm_fim/image-rank-variation-only.yaml' # Fed-HELLo Bone
#NCCL_DEBUG=INFO TORCH_DISTRIBUTED_DETAIL=DEBUG accelerate launch --main_process_port 29505 main.py --config_name 'experiments/cifar100_vit_lora/depthffm_fim/image_cifar100_vit_fedavg_depthffm_fim-6_9_12-bone_noniid-pat_20_dir-noprior-s50-e50.yaml' # Fed-HELLo Bone
#NCCL_DEBUG=INFO TORCH_DISTRIBUTED_DETAIL=DEBUG accelerate launch --main_process_port 29505 main.py --config_name 'experiments/cifar100_vit_lora/depthffm_fim/image_cifar100_vit_fedavg_depthffm_fim-6_9_12-bone_noniid-pat_10_dir-noprior-s50-e50.yaml' # Fed-HELLo Bone


# Straggler Learning
#NCCL_DEBUG=INFO TORCH_DISTRIBUTED_DETAIL=DEBUG accelerate launch --main_process_port 29505 main.py --config_name 'experiments/cifar100_vit_lora/0_0_12/image_cifar100_vit_fedavg_depthfl-0_0_12_noniid-pat_10_dir.yaml'
#NCCL_DEBUG=INFO TORCH_DISTRIBUTED_DETAIL=DEBUG accelerate launch --main_process_port 29505 main.py --config_name 'experiments/cifar100_vit_lora/6_6_6/image_cifar100_vit_fedavg_depthfl-6_6_6_noniid-pat_10_dir.yaml'
#NCCL_DEBUG=INFO TORCH_DISTRIBUTED_DETAIL=DEBUG accelerate launch --main_process_port 29505 main.py --config_name 'experiments/cifar100_vit_lora/6_6_6/image_cifar100_vit_fedavg_depthfl-6_6_6_noniid-pat_20_dir.yaml'

# Exclusive Learning
#NCCL_DEBUG=INFO TORCH_DISTRIBUTED_DETAIL=DEBUG accelerate launch --main_process_port 29505 main.py --config_name 'experiments/cifar100_vit_lora/Ours/image-rank-var-b-iid.yaml'
#NCCL_DEBUG=INFO TORCH_DISTRIBUTED_DETAIL=DEBUG accelerate launch --main_process_port 29505 main.py --config_name 'experiments/cifar100_vit_lora/Ours/image-rank-var-b-all-enableiid.yaml'
#NCCL_DEBUG=INFO TORCH_DISTRIBUTED_DETAIL=DEBUG accelerate launch --main_process_port 29505 main.py --config_name 'experiments/cifar100_vit_lora/depthffm_fim/FedHello_rank-24iid-noprior-s50-e50.yaml'
#NCCL_DEBUG=INFO TORCH_DISTRIBUTED_DETAIL=DEBUG accelerate launch --main_process_port 29505 main.py --config_name 'experiments/cifar100_vit_lora/Ours/image-rank-var-b-no-svd-iid.yaml'
NCCL_DEBUG=INFO TORCH_DISTRIBUTED_DETAIL=DEBUG accelerate launch --main_process_port 29505 main.py --config_name 'experiments/cifar100_vit_lora/Ours/alternating-training-warm20-double-rank-int.yaml'
#NCCL_DEBUG=INFO TORCH_DISTRIBUTED_DETAIL=DEBUG accelerate launch --main_process_port 29505 main.py --config_name 'experiments/cifar100_vit_lora/Ours/alternating-training-warm20.yaml'
#NCCL_DEBUG=INFO TORCH_DISTRIBUTED_DETAIL=DEBUG accelerate launch --main_process_port 29505 main.py --config_name 'experiments/cifar100_vit_lora/Ours/b-only-warm20.yaml'