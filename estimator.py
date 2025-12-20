from transformers import AutoModelForImageClassification, AutoConfig
from utils.memory_tracker import MemoryTracker

FEDHELLO = 'FedHello'
OURS = 'Ours'
MEM_ONLY = 'mem_only'

class RankEstimator:

    def __init__(self):
        self._tracker = MemoryTracker()

    def get_rank_for_all_client_groups(self, args, config, base_model, memory_summary_dict):

        #config = AutoConfig.from_pretrained(args.model)
        rank_for_all_client_groups = []
        for i in range(len(args.heterogeneous_group)):
            self._helper(args, config, base_model, memory_summary_dict, rank_for_all_client_groups, i)
        
        print(f'rank budget per module for all client groups respectively: {str(rank_for_all_client_groups)}')
        return rank_for_all_client_groups
    
    def get_rank_for_one_client_group(self, args, config, base_model, memory_summary_dict):

        #config = AutoConfig.from_pretrained(args.model)
        rank_for_all_client_groups = []
        for i in range(1):
            self._helper(args, config, base_model, memory_summary_dict, rank_for_all_client_groups, i)

        print(f'rank budget per module for all client groups respectively: {str(rank_for_all_client_groups)}')
        return rank_for_all_client_groups

    def _helper(self, args, config, base_model, memory_summary_dict, rank_for_all_client_groups, i):
            print(f"client group {i}")
            total_gpu_memory_size_in_GB_for_one_client_group = args.gpu_memory_size_for_each_group_in_GB[i]
            upload_network_speed_in_Mbps_for_one_client_group = args.avg_upload_network_speed_for_each_group_in_Mbps[i]
            download_network_speed_in_Mbps_for_one_client_group = args.avg_download_network_speed_for_each_group_in_Mbps[i]
            desired_uploading_time_in_seconds_for_one_client_group = args.desired_uploading_time_for_each_group_in_seconds[i]
            desired_downloading_time_in_seconds_for_one_client_group = args.desired_downloading_time_for_each_group_in_seconds[i]
            
            rank_for_one_client_group = self._get_rank_for_one_client_group(args, config, base_model, total_gpu_memory_size_in_GB_for_one_client_group, upload_network_speed_in_Mbps_for_one_client_group, download_network_speed_in_Mbps_for_one_client_group, desired_uploading_time_in_seconds_for_one_client_group, desired_downloading_time_in_seconds_for_one_client_group, memory_summary_dict)
            rank_for_all_client_groups.append(rank_for_one_client_group)
            
            memory_summary_dict['total_para_bytes'] = memory_summary_dict['base_model_para_bytes'] + memory_summary_dict['lora_param_bytes']
            memory_summary_dict['total_fwd_bytes'] = memory_summary_dict['base_model_fwd_bytes'] + memory_summary_dict['lora_fwd_bytes']
            memory_summary_dict['total_optimizer_states_bytes'] = memory_summary_dict['lora_optimizer_states_bytes']
            memory_summary_dict['total_grads_bytes'] = memory_summary_dict['lora_grads_bytes']
            memory_summary_dict['total_memory_bytes'] = round(memory_summary_dict['total_para_bytes'] + memory_summary_dict['total_fwd_bytes'] + memory_summary_dict['total_optimizer_states_bytes'] + memory_summary_dict['total_grads_bytes'], 2)
            
            print('------------------------------------------------------------------------------------------------')
            print('estimated: ')
            for k, v in memory_summary_dict.items():
                print(k, self._bytes_to_mb(v))

    def _get_rank_for_one_client_group(self, args, config, base_model, total_gpu_memory_size_in_GB, upload_network_speed_in_Mbps, download_network_speed_in_Mbps, desired_uploading_time_in_seconds, desired_downloading_time_in_seconds, memory_summary_dict):
        if args.rank_estimator_method == FEDHELLO or args.rank_estimator_method == MEM_ONLY:
            return self._get_rank_based_on_gpu_memory(args, config, base_model, total_gpu_memory_size_in_GB, memory_summary_dict)
        elif args.rank_estimator_method == OURS:
            return self._get_rank_based_on_all(args, config, base_model, total_gpu_memory_size_in_GB, upload_network_speed_in_Mbps, download_network_speed_in_Mbps, desired_uploading_time_in_seconds, desired_downloading_time_in_seconds, memory_summary_dict)
        else:
            raise ValueError(f'Invalid rank estimator method: {args.rank_estimator_method}')

    def _get_rank_based_on_all(self, args, config, base_model, total_gpu_memory_size_in_GB, upload_network_speed_in_Mbps, download_network_speed_in_Mbps, desired_uploading_time_in_seconds, desired_downloading_time_in_seconds, memory_summary_dict):
        rank_based_on_gpu_memory = self._get_rank_based_on_gpu_memory(args, config, base_model,  total_gpu_memory_size_in_GB, memory_summary_dict)
        rank_based_on_upload_network_speed = self._get_rank_based_on_network_speed(args, config, base_model, upload_network_speed_in_Mbps, desired_uploading_time_in_seconds)
        rank_based_on_download_network_speed = self._get_rank_based_on_network_speed(args, config, base_model, download_network_speed_in_Mbps, desired_downloading_time_in_seconds)
        return self._get_final_rank(args, config, rank_based_on_gpu_memory, rank_based_on_upload_network_speed, rank_based_on_download_network_speed)

    def _get_final_rank(self, args, config, rank_based_on_gpu_memory, rank_based_on_upload_network_speed, rank_based_on_download_network_speed):
        return min(rank_based_on_gpu_memory, rank_based_on_upload_network_speed, rank_based_on_download_network_speed) 
        # TODO
        #* config.num_hidden_layers * len(args.lora_target_modules)


    
    def _get_rank_based_on_gpu_memory(self, args, config, base_model, total_gpu_memory_size_in_GB, memory_summary_dict):

        total_gpu_memory_size_in_bytes = self._get_total_gpu_memory_size_in_bytes(args, total_gpu_memory_size_in_GB)
        base_model_portion = self._get_base_model_portion(args, config, base_model, memory_summary_dict)
        lora_portion = total_gpu_memory_size_in_bytes - base_model_portion

        return self._get_rank_based_on_lora_portion(args, config, base_model, lora_portion, memory_summary_dict)

    def _get_base_model_portion(self, args, config, base_model, memory_summary_dict):
        base_model_para_bytes = self._get_base_model_para_in_bytes(args, base_model)
        base_model_fwd_bytes, overhead_bytes = self._tracker.get_base_model_fwd_in_bytes_for_estimator(args, config, base_model)
        base_model_portion_bytes = base_model_para_bytes + base_model_fwd_bytes
        
        if memory_summary_dict is not None:
            memory_summary_dict['base_model_para_bytes'] = base_model_para_bytes
            memory_summary_dict['base_model_fwd_bytes'] = base_model_fwd_bytes
            memory_summary_dict['base_model_portion_bytes'] = base_model_portion_bytes
            memory_summary_dict['overhead_bytes'] = overhead_bytes
        return base_model_portion_bytes

    def _bytes_to_mb(self, bytes_value):
        return round(bytes_value / 1024 / 1024, 2)

    def _get_num_of_adapted_matrices(self, args):
        return 2 # A and B matrices

    def _get_num_of_modules_per_layer(self, args):
        return len(args.lora_target_modules)

    def _get_rank_based_on_lora_portion(self, args, config, base_model, lora_portion, memory_summary_dict):
        if lora_portion <= 0:
            print(f'Warning: GPU memory is too small to train the model')
            return 0

        B = args.batch_size
        H = config.hidden_size
        mlp_ratio = config.intermediate_size / H   
        sequence_length = self._get_sequence_length(args, config)
        bytes_per_parameter = self._get_byte_per_parameter(args.precision)

    
        def is_normal_mod(module_name):
            return 'query' in module_name or 'key' in module_name or 'value' in module_name or 'attention.output.dense' in module_name or 'intermediate.dense' in module_name

        C = 2 # two matrices A and B
        
        def get_param_mem(r, module_name, bytes_per_parameter):
            if is_normal_mod(module_name):
                return H * r * C * bytes_per_parameter
            elif 'output.dense' in module_name:
                return mlp_ratio * H * r * C * bytes_per_parameter
            
            raise ValueError('invalid module name: ' + module_name)

        def get_optimizer_state_count(optim_type):
            if optim_type == 'adam' or optim_type == 'adamw':
                return 2
            elif optim_type == 'sgd':
                return 1
            raise NotImplementedError('unsupported optimizer type')

        def get_optimizer_state_mem(r, module_name, bytes_per_parameter, optim_type):
            return get_optimizer_state_count(optim_type) * get_param_mem(r, module_name, bytes_per_parameter)
            
        def get_gradient_mem(r, module_name, bytes_per_parameter):
            return get_param_mem(r, module_name, bytes_per_parameter)


        # (beta1 * B * sequence_length * H + beta2 * B * sequence_length * r) * bytes_per_parameter * layers = lora_portion

        layers = config.num_hidden_layers
        lora_portion_per_layer = lora_portion / layers
        D = H * bytes_per_parameter * C
        
        total_dim = 0
        sum_of_b1BSHbytes = 0
        sum_of_ratio_D = 0
        sum_of_b2BSbytes = 0

        (beta1, beta2) = self._tracker.get_lora_betas_v2(args, config, base_model, args.lora_target_modules, B, sequence_length, H, bytes_per_parameter, memory_summary_dict)
        for lora_target_module in args.lora_target_modules:
            print(lora_target_module)
            ratio = 1 if is_normal_mod(lora_target_module) else mlp_ratio
            sum_of_ratio_D += ratio * D
            b2BSbytes = beta2 * B * sequence_length * bytes_per_parameter * C
            total_dim += ratio * D * (2 + get_optimizer_state_count(args.optimizer)) * layers
            sum_of_b1BSHbytes += beta1 * B * sequence_length * H * bytes_per_parameter
        
        lora_portion -= beta1 * B * sequence_length * H * bytes_per_parameter
        total_dim += b2BSbytes
        rank = int(lora_portion_per_layer / total_dim)
        rank = min(rank, H)
        print(rank)
        rank = max(rank, 0)
        print('est rank by memory:', rank)

        print('sum_of_ratio_D * layers', sum_of_ratio_D * layers)
        memory_summary_dict['lora_param_bytes'] = sum_of_ratio_D * layers * rank
        print(memory_summary_dict['lora_param_bytes'], self._bytes_to_mb(memory_summary_dict['lora_param_bytes']))
        memory_summary_dict['lora_optimizer_states_bytes'] = memory_summary_dict['lora_param_bytes'] * get_optimizer_state_count(args.optimizer)
        memory_summary_dict['lora_grads_bytes'] = memory_summary_dict['lora_param_bytes']
        memory_summary_dict['lora_fwd_bytes'] = (beta1 * B * sequence_length * H * bytes_per_parameter + b2BSbytes * rank)
        memory_summary_dict['lora_total_bytes'] = memory_summary_dict['lora_param_bytes'] + memory_summary_dict['lora_fwd_bytes'] + memory_summary_dict['lora_optimizer_states_bytes'] + memory_summary_dict['lora_grads_bytes']

        return rank
    
    def _get_total_gpu_memory_size_in_bytes(self, args, total_gpu_memory_size_in_GB):
        return total_gpu_memory_size_in_GB * 1024 * 1024 * 1024

    def _get_base_model_para_in_bytes(self, args, base_model):
        '''
        base_model = AutoModelForImageClassification.from_pretrained('facebook/deit-small-patch16-224')
        '''
        
        parameter_size = sum(p.numel() for p in base_model.parameters())
        
        byte_per_parameter = self._get_byte_per_parameter(args.precision)

        return parameter_size * byte_per_parameter

    def _get_byte_per_parameter(self, precision):
        if precision == 'fp32':
            return 4
        elif precision == 'fp16':
            return 2
        else:
            raise ValueError(f'Invalid precision: {precision}')

    def _get_sequence_length(self, args, config):
        config = AutoConfig.from_pretrained(args.model)
        CLS_TOKEN = args.CLS_TOKEN
        
        return config.image_size / config.patch_size * config.image_size / config.patch_size + CLS_TOKEN
        
    def _get_base_model_optimizer_states_memory_size_in_bytes(self, args, base_model_memory_size_in_bytes):
        '''
        Optimizer states include the part for base model and the part for LoRA.
        This function only calculates the memory size for the base model portion.
        '''

        if not args.train_classifier:
            return 0
        else:
            raise NotImplementedError('Not implemented yet.')

    def _get_rank_based_on_network_speed(self, args, config, model,network_speed_in_Mbps, desired_communication_time_in_seconds):
        bytes_per_second = network_speed_in_Mbps * 1_000_000 / 8
        parameter_size_in_bytes = desired_communication_time_in_seconds * bytes_per_second
        num_modules_per_layer = self._get_num_of_modules_per_layer(args)
        H = config.hidden_size
        C = self._get_num_of_adapted_matrices(args)
        num_layers = args.num_of_layers_to_allocate_LoRA
        bytes_per_parameter = self._get_byte_per_parameter(args.precision)
        total_dimension_size = C * num_modules_per_layer * H * num_layers * bytes_per_parameter
        rank = int(parameter_size_in_bytes / total_dimension_size)
        return rank



