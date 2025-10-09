import torch
from torch import nn
import copy

import numpy as np
from tqdm import tqdm
from accelerate import Accelerator
from accelerate import DistributedDataParallelKwargs

class LocalUpdate(object):
    def __init__(self, args):
        self.args = args
        if args.data_type == 'image':
            self.loss_func = nn.CrossEntropyLoss()
        elif args.data_type == 'text':
            self.loss_func = nn.CrossEntropyLoss()
        elif args.data_type == 'sentiment':
            self.loss_func = nn.NLLLoss()

    def lora_tuning(self, model, ldr_train, args, client_index, client_real_id, round, hete_group_id):
        ddp_kwargs = DistributedDataParallelKwargs(find_unused_parameters=True)
        accelerator = Accelerator(kwargs_handlers=[ddp_kwargs])

        if isinstance(getattr(args, 'heterogeneous_group'+str(hete_group_id)+'_lora'), list):
            no_weight_lora = list(set(range(args.lora_layer)) - set(getattr(args, 'heterogeneous_group'+str(hete_group_id)+'_lora')))
        elif isinstance(getattr(args, 'heterogeneous_group'+str(hete_group_id)+'_lora'), int):
            no_weight_lora = list(set(range(args.lora_layer)) - set(args.block_ids_list[client_real_id]))

        # early stop for exclusive training
        if len(no_weight_lora) == args.lora_layer:
            return model.state_dict(), None, no_weight_lora

        optimizer_grouped_parameters = [
                {
                    "params": [p for n, p in model.named_parameters() if not any(str(nd) in n for nd in no_weight_lora)]
                },
                {
                    "params": [p for n, p in model.named_parameters() if any(str(nd) in n for nd in no_weight_lora)],
                    'lr': 0.0
                }
        ]
        optimizer = torch.optim.AdamW(optimizer_grouped_parameters, lr=args.local_lr)
        # # Prepare everything with our `accelerator`.
        model, optimizer, train_dataloader = accelerator.prepare(model, optimizer, ldr_train)
        total_loss = []
        for t_au in range(self.args.tau):
            with accelerator.accumulate(model):
                for step, batch in tqdm(enumerate(train_dataloader), desc='Local Training Client '+str(client_index)+' Tau: '+str(t_au), total=len(train_dataloader), disable=(not accelerator.is_local_main_process)):
                    optimizer.zero_grad()
                    outputs = model(**batch)
                    loss = outputs.loss
                    accelerator.backward(loss)
                    optimizer.step()
                    # lr_scheduler.step()
                    if accelerator.is_local_main_process:
                        total_loss.append(loss.detach().float().cpu())
                    accelerator.wait_for_everyone()
                    
        args.logger.info(f'Total local training loss is: {np.mean(total_loss)}', main_process_only=True)

        # optimizer.zero_grad()
        return accelerator.unwrap_model(model).state_dict(), np.mean(total_loss), no_weight_lora

