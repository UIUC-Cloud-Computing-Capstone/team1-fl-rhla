from transformers import AutoModelForImageClassification, AutoModelForSequenceClassification, AutoModelForMultipleChoice
import numpy as np
import torch
from peft import LoraConfig, get_peft_model
model = AutoModelForImageClassification.from_pretrained(
    'facebook/deit-small-patch16-224', # https://huggingface.co/google/bert_uncased_L-4_H-256_A-4
)
config = LoraConfig(
    r=48,
    lora_alpha=48,
    target_modules=["query", "value"],
    lora_dropout=0.1,
    bias="none",
    modules_to_save=["classifier"],
)
net_glob = get_peft_model(model, config)

for name, _ in net_glob.named_parameters():
    print(name)


print('########### trainable param ###########')
trainable_names = [n for n, p in net_glob.named_parameters() if p.requires_grad]
for n in trainable_names:
    print(n)
    # only lora and trainable 

global_model = net_glob.state_dict()

model = net_glob

no_weight_lora = [1,2,3]

optimizer_grouped_parameters = [
        {
            "params": [p for n, p in model.named_parameters() if not any(str(nd) in n for nd in no_weight_lora)]
        },
        {
            "params": [p for n, p in model.named_parameters() if any(str(nd) in n for nd in no_weight_lora)],
            'lr': 0.0
        }
]

print('######## normal learning rate param ###########')
param_name = [n for n, p in model.named_parameters() if (('lora_A' in n) or any(('layer.' + str(nd) + '.') in n for nd in no_weight_lora))];
for  n in param_name:
    print(n)
#B*A

#s = 'base_model.model.vit.encoder.layer.11.attention.attention.query.lora_A.default.weight'

#import re
#match = re.search(r'\.layer\.(\d+)\.', s)
#if match:
#    layer_num = int(match.group(1))
#    print(layer_num)  # 2


#print(model.parameters())

# A has value, B is all zero
#print(global_model['base_model.model.vit.encoder.layer.11.attention.attention.value.lora_A.default.weight'].shape) # (48,384)
#print(global_model['base_model.model.vit.encoder.layer.11.attention.attention.value.lora_B.default.weight']) # (384,16)

# A (48,384) -> (24,384)
# B (384,48) -> (384,24)


#W + B@A #(384,384)





#print(global_model['base_model.model.vit.encoder.layer.11.attention.attention.value.lora_A.default.weight'])

#for k in global_model.keys():
#    if 'lora_A' in k:
#        global_model[k][24:,:] = 0
#    elif 'lora_B' in k:
#        global_model[k][:,24:] = 0

#print(global_model['base_model.model.vit.encoder.layer.11.attention.attention.value.lora_A.default.weight'])

#print(global_model['base_model.model.vit.encoder.layer.11.attention.attention.value.lora_B.default.weight'])


#k = 'base_model.model.vit.encoder.layer.11.attention.attention.value.lora_B.default.weight'
#B = global_model[k]
#new_name = k.replace('lora_B', 'lora_A')
#A = global_model[new_name]
#U, S, VT = torch.linalg.svd(B@A, full_matrices=False) 

#print(f'U shape {U.shape}, S shape {S.shape}, VT shape {VT.shape} ' )
#print(torch.sum(S))
#global_model[k] = (U@torch.diag(S))[:,0:48]
#global_model[new_name] = VT[0:48,:]

#print(global_model['base_model.model.vit.encoder.layer.11.attention.attention.value.lora_A.default.weight'])

#print(global_model['base_model.model.vit.encoder.layer.11.attention.attention.value.lora_B.default.weight'])

#print(global_model[k]@global_model[new_name])