import re
def get_rank(args, client_real_id, module_name):
    layer_id = int(re.findall(r"\d+", module_name)[0])    
    layer_index = args.block_ids_list[client_real_id].index(layer_id)
    rank = args.rank_list[client_real_id][layer_index]
    return rank