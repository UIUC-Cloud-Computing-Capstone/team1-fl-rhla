import torch
from torch import nn
import copy
import re

import numpy as np
from tqdm import tqdm
import gc
from accelerate import Accelerator
from accelerate import DistributedDataParallelKwargs
import time

def get_parameter_names(model, forbidden_layer_types):
    """
    Returns the names of the model parameters that are not inside a forbidden layer.
    """
    result = []
    for name, child in model.named_children():
        result += [
            f"{name}.{n}"
            for n in get_parameter_names(child, forbidden_layer_types)
            if not isinstance(child, tuple(forbidden_layer_types))
        ]
    # Add model specific parameters (defined with nn.Parameter) since they are not in any child.
    result += list(model._parameters.keys())
    return result

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
        """
        Run local LoRA fine-tuning for one client with heterogeneous layer and rank assignment.

        Only LoRA (or LoKR) parameters in the client's assigned layers (args.block_ids_list[client_real_id])
        are trained; optionally only lora_A or lora_B depending on args.train_a / args.train_b.
        Supports per-layer rank via gradient hooks (proposed_method, LEGEND, HetLoRA, FlexLoRA) and
        FlexLoRA rank reduction via SVD before training. Uses Accelerator for distributed training.
        If the client has no trainable layers, returns immediately with no_weight_lora covering all layers.

        Args:
            model: The global model (will be copied and trained locally).
            ldr_train: DataLoader for this client's local training data.
            args: Config with block_ids_list, rank_list, train_a, train_b, local_lr, tau, LOKR,
                  FlexLoRA, proposed_method, LEGEND, HetLoRA, weight_decay, logger, etc.
            client_index (int): Index of this client in the current round's selected set.
            client_real_id (int): Global client id (index into block_ids_list, rank_list).
            round (int): Current communication round (for logging).
            hete_group_id (int): Heterogeneous group id; used to resolve group config
                                 (e.g. heterogeneous_group{id}_lora) and no_weight_lora.

        Returns:
            tuple: (state_dict, mean_loss, no_weight_lora)
                - state_dict: Updated model state after local training.
                - mean_loss: Mean training loss (float), or None if client had nothing to train.
                - no_weight_lora: List of LoRA layer indices not trained by this client
                  (used by the server to avoid aggregating those parameters from this client).
        """
        ddp_kwargs = DistributedDataParallelKwargs(find_unused_parameters=True)
        accelerator = Accelerator(kwargs_handlers=[ddp_kwargs])

        if isinstance(getattr(args, 'heterogeneous_group'+str(hete_group_id)+'_lora'), list):
            no_weight_lora = list(set(range(args.lora_layer)) - set(getattr(args, 'heterogeneous_group'+str(hete_group_id)+'_lora')))
        elif isinstance(getattr(args, 'heterogeneous_group'+str(hete_group_id)+'_lora'), int):
            no_weight_lora = list(set(range(args.lora_layer)) - set(args.block_ids_list[client_real_id]))

        # early stop for exclusive training
        if len(no_weight_lora) == args.lora_layer:
            print(f'client {client_real_id} has not weight to train, return')
            return model.state_dict(), None, no_weight_lora

        # set everything to no-trainable
        for name, param in model.named_parameters():
            param.requires_grad = False

        # only train the enabled lora module.
        lora_str = 'lora'
        if args.LOKR:
            lora_str = 'lokr_w'
        for name, param in model.named_parameters():
            if (lora_str in name and any(('layer.' + str(nd) + '.') in name for nd in args.block_ids_list[client_real_id])) or 'classifier' in name:
                if args.train_b and 'lora_B' in name:
                    param.requires_grad = True

                if args.train_a and 'lora_A' in name:
                    param.requires_grad = True

                # set all lokr param to trainable.
                if args.LOKR:
                    param.requires_grad = True

        if args.FlexLoRA:
            params = dict(model.named_parameters())
            print('Rank reduction for flexlora')
            with torch.no_grad():
                for name, param in model.named_parameters():
                    if ("lora_B" not in name) or (not param.requires_grad):
                        continue

                    layer_id = int(re.findall(r"\d+", name)[0])
                    layer_index = args.block_ids_list[client_real_id].index(layer_id)
                    rank = args.rank_list[client_real_id][layer_index]

                    lora_A_name = name.replace("lora_B", "lora_A")

                    # pull weights
                    B = params[name].detach().cpu()
                    A = params[lora_A_name].detach().cpu()
                    # print(f'{name} = {B}')
                    U, S, VT = torch.linalg.svd(B @ A, full_matrices=False)

                    # keep top-'rank' singular values (optionally also threshold)
                    tol = 1e-6
                    S_trunc = S.clone()
                    S_trunc[S_trunc < tol] = 0
                    S_trunc[rank:] = 0

                    sqrtS = torch.sqrt(S_trunc)
                    # print(sqrtS)
                    # k = args.lora_max_rank  # fixed LoRA rank in module
                    # If you want *effective* rank=rank, use k = rank instead.

                    new_B = (U @ torch.diag(S))[:, :args.lora_max_rank]
                    new_A = VT[:args.lora_max_rank, :]
                    # if 'layer.11.' in name:
                    #     print(f'{lora_A_name} = {new_A}')
                    #     print(f'{name} = {new_B}')
                    # copy back to original device/dtype
                    new_B = new_B.to(device=params[name].device, dtype=params[name].dtype)
                    new_A = new_A.to(device=params[lora_A_name].device, dtype=params[lora_A_name].dtype)

                    params[name].copy_(new_B)
                    params[lora_A_name].copy_(new_A)


        if args.proposed_method or args.LEGEND or args.HetLoRA or args.FlexLoRA:
            # add register to truncate the rank if rank variation is enable
            def lora_A_hook(cut_rank: int):
                def hook(grad):
                    grad = grad.clone()  # ensure writable
                    grad[cut_rank:, :] = 0
                    return grad       # zero out rows from idx onward
                return hook

            def lora_B_hook(cut_rank: int):
                def hook(grad):
                    # grad is a tensor
                    grad = grad.clone()
                    grad[:,cut_rank:] = 0       # zero out cols from idx onward
                    return grad
                return hook

            print(f'client {client_real_id} block_ids_list = {args.block_ids_list[client_real_id]}, rank_list = {args.rank_list[client_real_id]}, rank_budget = {sum(args.rank_list[client_real_id])}')
            for name, param in model.named_parameters():
                if 'lora' in name and param.requires_grad:
                    layer_id = int(re.findall(r"\d+", name)[0])
                    layer_index = args.block_ids_list[client_real_id].index(layer_id)

                    rank = args.rank_list[client_real_id][layer_index]
                    #print(f'layer id {layer_id}, rank = {rank}')
                    if 'lora_A' in name:
                        #print(f'lora_A name {name}')
                        param.register_hook(lora_A_hook(rank) )

                        if args.HetLoRA:
                            # truncate the param for HetLoRA
                            param.data[rank:, :].zero_()


                    elif 'lora_B' in name:
                        #print(f'lora_B name {name}')
                        param.register_hook(lora_B_hook(rank) )

                        if args.HetLoRA:
                            # truncate the param for HetLoRA
                            param.data[:,rank:].zero_()
        #print('############## trainable param ############')
        #print(f'args.block_ids_list[client_real_id] = {args.block_ids_list[client_real_id]}')
        #for name, param in model.named_parameters():
        #    if param.requires_grad:
        #        print(name) 

        # Note: Have to set the weight_decay to zero otherwise 0 gradient part will still be updated.
        # weight declay is set to zero only for rank variation
        weight_decay = args.weight_decay

        optimizer = torch.optim.AdamW(filter(lambda p: p.requires_grad, model.parameters()), lr=args.local_lr,weight_decay=weight_decay)
        # # Prepare everything with our `accelerator`.
        model, optimizer, train_dataloader = accelerator.prepare(model, optimizer, ldr_train)
        total_loss = []

        #unwrapped_model = accelerator.unwrap_model(model)
        for t_au in range(self.args.tau):
            with accelerator.accumulate(model):
                for step, batch in tqdm(enumerate(train_dataloader), desc='Local Training Client '+str(client_index)+' Tau: '+str(t_au), total=len(train_dataloader), disable=(not accelerator.is_local_main_process)):
                    optimizer.zero_grad()
                    outputs = model(**batch)
                    loss = outputs.loss
                    accelerator.backward(loss)

                    # ---- check gradients here ----

                    #for name, param in unwrapped_model.named_parameters():
                    #    if param.requires_grad and 'lora_A' in name:
                    #        grad_norm = param.grad.detach().data
                    #        accelerator.print(f"[step {step}] {name} grad = {grad_norm}")
                    #        #accelerator.print(f"param -")
                            #break  # uncomment if you only want the first layer’s grad
                    # --------------------------------

                    optimizer.step()
                    # lr_scheduler.step()
                    if accelerator.is_local_main_process:
                        total_loss.append(loss.detach().float().cpu())
                    accelerator.wait_for_everyone()
                    
        args.logger.info(f'Total local training loss is: {np.mean(total_loss)}', main_process_only=True)

        # optimizer.zero_grad()
        return accelerator.unwrap_model(model).state_dict(), np.mean(total_loss), no_weight_lora