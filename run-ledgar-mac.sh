##############
## LEDGAR ###
##############

# Straggler Learning (6_6_6)
#NCCL_DEBUG=INFO TORCH_DISTRIBUTED_DETAIL=DEBUG accelerate launch --main_process_port 29505 main.py --config_name 'experiments/ledgar_bert_lora/6_6_6/text_ledgar_bert_fedavg_depthfl-6_6_6_iid.yaml'
#NCCL_DEBUG=INFO TORCH_DISTRIBUTED_DETAIL=DEBUG accelerate launch --main_process_port 29505 main.py --config_name 'experiments/ledgar_bert_lora/6_6_6/text_ledgar_bert_fedavg_depthfl-6_6_6_noniid-pat_10_dir.yaml'
#NCCL_DEBUG=INFO TORCH_DISTRIBUTED_DETAIL=DEBUG accelerate launch --main_process_port 29505 main.py --config_name 'experiments/ledgar_bert_lora/6_6_6/text_ledgar_bert_fedavg_depthfl-6_6_6_noniid-pat_20_dir.yaml'

# Exclusive Learning (0_0_12)
#NCCL_DEBUG=INFO TORCH_DISTRIBUTED_DETAIL=DEBUG accelerate launch --main_process_port 29505 main.py --config_name 'experiments/ledgar_bert_lora/0_0_12/text_ledgar_bert_fedavg_depthfl-0_0_12_iid.yaml'
#NCCL_DEBUG=INFO TORCH_DISTRIBUTED_DETAIL=DEBUG accelerate launch --main_process_port 29505 main.py --config_name 'experiments/ledgar_bert_lora/0_0_12/text_ledgar_bert_fedavg_depthfl-0_0_12_noniid-pat_10_dir.yaml'
#NCCL_DEBUG=INFO TORCH_DISTRIBUTED_DETAIL=DEBUG accelerate launch --main_process_port 29505 main.py --config_name 'experiments/ledgar_bert_lora/0_0_12/text_ledgar_bert_fedavg_depthfl-0_0_12_noniid-pat_20_dir.yaml'

# Fed-HeLLo-Bone (depthffm_fim)
accelerate launch --config_file accelerate_macos_config.yaml main.py --config_name 'experiments/ledgar_bert_lora/depthffm_fim/text_ledgar_bert_fedavg_depthffm_fim-6_9_12-bone_iid-noprior-s50-e50.yaml' # Fed-HELLo Bone
#NCCL_DEBUG=INFO TORCH_DISTRIBUTED_DETAIL=DEBUG accelerate launch --main_process_port 29505 main.py --config_name 'experiments/ledgar_bert_lora/depthffm_fim/text_ledgar_bert_fedavg_depthffm_fim-6_9_12-bone_noniid-pat_20_dir-noprior-s50-e50.yaml' # Fed-HELLo Bone
#NCCL_DEBUG=INFO TORCH_DISTRIBUTED_DETAIL=DEBUG accelerate launch --main_process_port 29505 main.py --config_name 'experiments/ledgar_bert_lora/depthffm_fim/text_ledgar_bert_fedavg_depthffm_fim-6_9_12-bone_noniid-pat_10_dir-noprior-s50-e50.yaml' # Fed-HELLo Bone

# Similar for other datasets ....
