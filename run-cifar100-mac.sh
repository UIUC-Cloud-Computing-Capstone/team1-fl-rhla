##############
## CIFAR100 ##
##############


# Fed-HeLLo-Bone
#accelerate launch --config_file accelerate_macos_config.yaml main.py --config_name 'experiments/cifar100_vit_lora/depthffm_fim/image_cifar100_vit_fedavg_depthffm_fim-6_9_12-bone_iid-noprior-s50-e50.yaml' # Fed-HELLo Bone
accelerate launch --config_file accelerate_macos_config.yaml main.py --config_name 'experiments/cifar100_vit_lora/depthffm_fim/image_cifar100_vit_fedavg_depthffm_fim-6_9_12-bone_noniid-pat_20_dir-noprior-s50-e50.yaml' # Fed-HELLo Bone
#accelerate launch --config_file accelerate_macos_config.yaml main.py --config_name 'experiments/cifar100_vit_lora/depthffm_fim/image_cifar100_vit_fedavg_depthffm_fim-6_9_12-bone_noniid-pat_10_dir-noprior-s50-e50.yaml' # Fed-HELLo Bone

# Similar for other datasets ....