import torch
import numpy as np

def average_lora_depthfl(args, global_model, loc_updates):
    '''
    hetero average
    '''
    model_update_avg_dict = {}

    print('############## global aggregation ####################')
    lora_str = 'lora'
    if args.LOKR:
        lora_str = 'lokr'

    for k in global_model.keys():
        if lora_str in k or 'classifier' in k:
            for loc_update in loc_updates:
                if k in loc_update:
                    if k in model_update_avg_dict:
                        model_update_avg_dict[k].append(loc_update[k])
                    else:
                        model_update_avg_dict[k] = []
                        model_update_avg_dict[k].append(loc_update[k])

    for k in global_model.keys():
        if k in model_update_avg_dict:
            global_model[k] = global_model[k].detach().cpu() +  sum(model_update_avg_dict[k]) / len(model_update_avg_dict[k])
    return global_model

def weighted_average_lora_depthfl(args, global_model, loc_updates, num_samples):
    '''
    weighted hetero average
    '''
    model_update_avg_dict = {}
    model_weights_cnt = {}
    model_weights_list = {}

    lora_str = 'lora'
    if args.LOKR:
        lora_str = 'lokr'
    for k in global_model.keys():
        if lora_str in k or 'classifier' in k: # classifier is not included
            for client_i, loc_update in enumerate(loc_updates):
                if k in loc_update:
                    if k in model_update_avg_dict:
                        model_update_avg_dict[k].append(loc_update[k])
                        model_weights_cnt[k] += num_samples[client_i]
                        model_weights_list[k].append(num_samples[client_i])
                    else:
                        model_update_avg_dict[k] = []
                        model_update_avg_dict[k].append(loc_update[k])
                        model_weights_cnt[k] = 0.0
                        model_weights_cnt[k] += num_samples[client_i]
                        model_weights_list[k] = []
                        model_weights_list[k].append(num_samples[client_i])

    for k in global_model.keys():
        if k in model_update_avg_dict:
            weight_based = model_weights_cnt[k]
            model_weights_list[k] = [item/weight_based for item in model_weights_list[k]]
            global_model[k] = global_model[k].detach().cpu() + sum([model*weight for model, weight in zip(model_update_avg_dict[k], model_weights_list[k])])



    return global_model


def svd_average(args, global_model, loc_updates, num_samples):
    '''
    hetero average
    '''
    update_weights = []
    for updates in loc_updates:
        svd_weights = []
        keys = list(updates.keys())
        for i in range(0, len(keys), 2):
            keyA = keys[i]
            keyB = keys[i+1]
            tensorA = updates[keyA]
            tensorB = updates[keyB]
            tensorSVD = torch.matmul(tensorB, tensorA)
            svd_weights.append(tensorSVD)

        svd_weights = torch.cat(svd_weights, dim=0)
        print(f'$$$$$$$$$$$$$$ after stack size of svd_weights = {svd_weights.shape}')
        frobenius_norm = np.linalg.norm(svd_weights, 'fro')
        update_weights.append(frobenius_norm)
    update_weights = update_weights / np.sum(update_weights)
    args.logger.info(f'the weights of different clients {update_weights}', main_process_only=True)
    for index, (u, w) in enumerate(zip(loc_updates, update_weights)):
        loc_updates[index] = {key: value * w for key, value in u.items()}

    model_update_avg_dict = {}
    for k in global_model.keys():
        if 'lora' in k or 'classifier' in k:
            for loc_update in loc_updates:
                if k in loc_update:
                    if k in model_update_avg_dict:
                        model_update_avg_dict[k].append(loc_update[k])
                    else:
                        model_update_avg_dict[k] = []
                        model_update_avg_dict[k].append(loc_update[k])
    for k in global_model.keys():
        if k in model_update_avg_dict:
            global_model[k] = global_model[k].detach().cpu() +  sum(model_update_avg_dict[k])
    return global_model


def product_average(args, global_model, loc_updates, num_samples):
    '''
    hetero average
    '''
    print('#### aggregate')
    model_update_avg_dict = {}

    print('############## global aggregation ####################')
    lora_str = 'lora'
    if args.LOKR:
        lora_str = 'lokr'

    for k in global_model.keys():
        if lora_str in k or 'classifier' in k:
            for loc_update in loc_updates:
                if k in loc_update:
                    if k in model_update_avg_dict:
                        model_update_avg_dict[k].append(global_model[k].detach().cpu()+ loc_update[k])
                    else:
                        model_update_avg_dict[k] = []
                        model_update_avg_dict[k].append(global_model[k].detach().cpu() + loc_update[k])

    for k in global_model.keys():
        if k in model_update_avg_dict:
            value = model_update_avg_dict[k]
            global_model[k] = torch.mean(torch.stack(value), dim=0)
    return global_model