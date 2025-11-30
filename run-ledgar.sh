
# Proposed approach
# Run using the accelerate framework with provided config yaml
NCCL_DEBUG=INFO TORCH_DISTRIBUTED_DETAIL=DEBUG accelerate launch --main_process_port 29505 main.py --config_name 'experiments/ledgar_bert_lora/depthffm_fim/proposed-approach.yaml' 




























#NCCL_DEBUG=INFO TORCH_DISTRIBUTED_DETAIL=DEBUG accelerate launch --main_process_port 29505 main.py --config_name 'experiments/ledgar_bert_lora/depthffm_fim/text_ledgar_bert_fedavg_depthffm_fim-6_6_6-bone_iid-noprior-s50-e50.yaml' # Fed-HELLo Bone
#NCCL_DEBUG=INFO TORCH_DISTRIBUTED_DETAIL=DEBUG accelerate launch --main_process_port 29505 main.py --config_name 'experiments/ledgar_bert_lora/depthffm_fim/text_ledgar_bert_fedavg_depthffm_fim-6_9_12-bone_iid-noprior-s50-e50.yaml' # Fed-HELLo Bone