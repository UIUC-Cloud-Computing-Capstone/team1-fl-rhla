##############
## ledgar   ##
##############

# rank-16
NCCL_DEBUG=INFO TORCH_DISTRIBUTED_DETAIL=DEBUG accelerate launch --main_process_port 29505 main.py --config_name 'experiments/ledgar_bert_lora/Ours/alternating-training-warm20-double-rank-int-rank16.yaml'

# FedHello rank-16
NCCL_DEBUG=INFO TORCH_DISTRIBUTED_DETAIL=DEBUG accelerate launch --main_process_port 29505 main.py --config_name 'experiments/ledgar_bert_lora/FedHello/FedHello_rank-16iid-noprior-s50-e50.yaml'


# rank-12
NCCL_DEBUG=INFO TORCH_DISTRIBUTED_DETAIL=DEBUG accelerate launch --main_process_port 29505 main.py --config_name 'experiments/ledgar_bert_lora/Ours/alternating-training-warm20-double-rank-int-rank12.yaml'


# FedHello rank-12
NCCL_DEBUG=INFO TORCH_DISTRIBUTED_DETAIL=DEBUG accelerate launch --main_process_port 29505 main.py --config_name 'experiments/ledgar_bert_lora/FedHello/FedHello_rank-12iid-noprior-s50-e50.yaml'


# rank-8
NCCL_DEBUG=INFO TORCH_DISTRIBUTED_DETAIL=DEBUG accelerate launch --main_process_port 29505 main.py --config_name 'experiments/ledgar_bert_lora/Ours/alternating-training-warm20-double-rank-int-rank8.yaml'


# FedHello rank-8
NCCL_DEBUG=INFO TORCH_DISTRIBUTED_DETAIL=DEBUG accelerate launch --main_process_port 29505 main.py --config_name 'experiments/ledgar_bert_lora/FedHello/FedHello_rank-8iid-noprior-s50-e50.yaml'



