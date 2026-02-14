##############
## ledgar   ##
##############

# exclusive
NCCL_DEBUG=INFO TORCH_DISTRIBUTED_DETAIL=DEBUG accelerate launch --main_process_port 29505 main.py --config_name 'experiments/ledgar_bert_lora/Exclusive/Exclusive-rank24-noniid-pat_10_dir.yaml'

# fedhello
NCCL_DEBUG=INFO TORCH_DISTRIBUTED_DETAIL=DEBUG accelerate launch --main_process_port 29505 main.py --config_name 'experiments/ledgar_bert_lora/FedHello/FedHello_noniid-pat_10_dir-noprior-s50-e50.yaml'

#ours
NCCL_DEBUG=INFO TORCH_DISTRIBUTED_DETAIL=DEBUG accelerate launch --main_process_port 29505 main.py --config_name 'experiments/ledgar_bert_lora/Ours/alternating-training-warm20-double-rank-int-no-rank-vary-noniid-10.yaml'

#straggler
NCCL_DEBUG=INFO TORCH_DISTRIBUTED_DETAIL=DEBUG accelerate launch --main_process_port 29505 main.py --config_name 'experiments/ledgar_bert_lora/Straggler/straggler-rank24-noniid-pat_10_dir.yaml'

#fedit
NCCL_DEBUG=INFO TORCH_DISTRIBUTED_DETAIL=DEBUG accelerate launch --main_process_port 29505 main.py --config_name 'experiments/ledgar_bert_lora/FedIT/FedIT-noniid-pat_10_dir.yaml'

#ffa-lora
NCCL_DEBUG=INFO TORCH_DISTRIBUTED_DETAIL=DEBUG accelerate launch --main_process_port 29505 main.py --config_name 'experiments/ledgar_bert_lora/FFA-LoRA/FFA-LORA-image-b-only-noniid-10.yaml'

#LEGEND
NCCL_DEBUG=INFO TORCH_DISTRIBUTED_DETAIL=DEBUG accelerate launch --main_process_port 29505 main.py --config_name 'experiments/ledgar_bert_lora/LEGEND/LEGEND-noniid-10.yaml'

#lokr
NCCL_DEBUG=INFO TORCH_DISTRIBUTED_DETAIL=DEBUG accelerate launch --main_process_port 29505 main.py --config_name 'experiments/ledgar_bert_lora/LOKR/LOKR-noniid-10.yaml'



