##############
## ledgar   ##
##############

# exclusive
NCCL_DEBUG=INFO TORCH_DISTRIBUTED_DETAIL=DEBUG accelerate launch --main_process_port 29505 main.py --config_name 'experiments/ledgar_bert_lora/Exclusive/Exclusive-rank24-iid.yaml'

# fedhello
NCCL_DEBUG=INFO TORCH_DISTRIBUTED_DETAIL=DEBUG accelerate launch --main_process_port 29505 main.py --config_name 'experiments/ledgar_bert_lora/FedHello/FedHello_rank-24iid-noprior-s50-e50.yaml'

#ours
NCCL_DEBUG=INFO TORCH_DISTRIBUTED_DETAIL=DEBUG accelerate launch --main_process_port 29505 main.py --config_name 'experiments/ledgar_bert_lora/Ours/alternating-training-warm20-double-rank-int-no-rank-vary.yaml'

#straggler
NCCL_DEBUG=INFO TORCH_DISTRIBUTED_DETAIL=DEBUG accelerate launch --main_process_port 29505 main.py --config_name 'experiments/ledgar_bert_lora/Straggler/straggler-rank24_iid.yaml'

#fedit
NCCL_DEBUG=INFO TORCH_DISTRIBUTED_DETAIL=DEBUG accelerate launch --main_process_port 29505 main.py --config_name 'experiments/ledgar_bert_lora/FedIT/FedIT-iid.yaml'

#ffa-lora
NCCL_DEBUG=INFO TORCH_DISTRIBUTED_DETAIL=DEBUG accelerate launch --main_process_port 29505 main.py --config_name 'experiments/ledgar_bert_lora/FFA-LoRA/FFA-LORA-image-b-only-iid.yaml'

#LEGEND
NCCL_DEBUG=INFO TORCH_DISTRIBUTED_DETAIL=DEBUG accelerate launch --main_process_port 29505 main.py --config_name 'experiments/ledgar_bert_lora/LEGEND/LEGEND-iid.yaml'

#lokr
NCCL_DEBUG=INFO TORCH_DISTRIBUTED_DETAIL=DEBUG accelerate launch --main_process_port 29505 main.py --config_name 'experiments/ledgar_bert_lora/LOKR/LOKR-iid.yaml'

#HetLoRA
NCCL_DEBUG=INFO TORCH_DISTRIBUTED_DETAIL=DEBUG accelerate launch --main_process_port 29505 main.py --config_name 'experiments/ledgar_bert_lora/HetLoRA/HetLoRA-iid.yaml'
