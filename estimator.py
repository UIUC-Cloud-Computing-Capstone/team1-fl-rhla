from transformers import AutoModelForImageClassification, AutoConfig

FEDHELLO = 'FedHello'
OURS = 'Ours'

class RankEstimator:

    def get_rank_for_all_client_groups(self, args, base_model):

        config = AutoConfig.from_pretrained(args.model)
        rank_for_all_client_groups = []
        for i in range(len(args.heterogeneous_group)):
            print(f"client group {i}")
            total_gpu_memory_size_in_GB_for_one_client_group = args.gpu_memory_size_for_each_group_in_GB[i]
            upload_network_speed_in_Mbps_for_one_client_group = args.avg_upload_network_speed_for_each_group_in_Mbps[i]
            download_network_speed_in_Mbps_for_one_client_group = args.avg_download_network_speed_for_each_group_in_Mbps[i]
            desired_uploading_time_in_seconds_for_one_client_group = args.desired_uploading_time_for_each_group_in_seconds[i]
            desired_downloading_time_in_seconds_for_one_client_group = args.desired_downloading_time_for_each_group_in_seconds[i]
            memory_summary_dict = {}
            rank_for_one_client_group = self._get_rank_for_one_client_group(args, config, base_model, total_gpu_memory_size_in_GB_for_one_client_group, upload_network_speed_in_Mbps_for_one_client_group, download_network_speed_in_Mbps_for_one_client_group, desired_uploading_time_in_seconds_for_one_client_group, desired_downloading_time_in_seconds_for_one_client_group, memory_summary_dict)
            rank_for_all_client_groups.append(rank_for_one_client_group)
            
            memory_summary_dict['total_parameters_in_MB'] = memory_summary_dict['base_model_para_in_MB'] + memory_summary_dict.get('lora_portion_parameter_size_in_MB', 0)
            memory_summary_dict['total_fwd_in_MB'] = memory_summary_dict['base_model_fwd_in_MB'] + memory_summary_dict.get('lora_portion_activations_gradients_and_workspace_margin_in_MB', 0)
            memory_summary_dict['total_optimizer_states_in_MB'] = memory_summary_dict.get('base_model_optimizer_states_memory_size_in_MB', 0) + memory_summary_dict.get('lora_portion_optimizer_states_size_in_MB', 0)
            memory_summary_dict['total_grads_in_MB'] = memory_summary_dict.get('base_model_grads_memory_size_in_MB', 0) + memory_summary_dict.get('lora_portion_grads_size_in_MB', 0)
            memory_summary_dict['total_memory_in_MB'] = round(memory_summary_dict['total_parameters_in_MB'] + memory_summary_dict['total_fwd_in_MB'] + memory_summary_dict['total_optimizer_states_in_MB'], 2)
            
            self._print_memory_summary(memory_summary_dict)

            print('------------------------------------------------------------------------------------------------')
        
        
        print(f'rank budget per module for all client groups respectively: {str(rank_for_all_client_groups)}')
        
        
        
        return rank_for_all_client_groups

    def _print_memory_summary(self, memory_summary_dict):
        total_parameters_in_MB = memory_summary_dict['total_parameters_in_MB']
        total_fwd_in_MB = memory_summary_dict['total_fwd_in_MB']
        total_optimizer_states_in_MB = memory_summary_dict['total_optimizer_states_in_MB']
        total_grads_in_MB = memory_summary_dict['total_grads_in_MB']
        total_memory_in_MB = memory_summary_dict['total_memory_in_MB']
        
        print(f"Parameters: {total_parameters_in_MB} MB ({total_parameters_in_MB / total_memory_in_MB * 100:.2f}%)")
        print(f"Optimizer States: {total_optimizer_states_in_MB} MB ({total_optimizer_states_in_MB / total_memory_in_MB * 100:.2f}%)")
        print(f"Fwd: {total_fwd_in_MB} MB ({total_fwd_in_MB / total_memory_in_MB * 100:.2f}%)")
        print(f"Grads: {total_fwd_in_MB} MB ({total_grads_in_MB / total_memory_in_MB * 100:.2f}%)")
        print(f"Total Memory: {total_memory_in_MB} MB ({total_memory_in_MB / total_memory_in_MB * 100:.2f}%)")

    def _get_rank_for_one_client_group(self, args, config, base_model, total_gpu_memory_size_in_GB, upload_network_speed_in_Mbps, download_network_speed_in_Mbps, desired_uploading_time_in_seconds, desired_downloading_time_in_seconds, memory_summary_dict):
        if args.rank_estimator_method == FEDHELLO:
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
        return min(rank_based_on_gpu_memory, rank_based_on_upload_network_speed, rank_based_on_download_network_speed) * config.num_hidden_layers * args.lora_target_modules_per_layer
    
    def _get_rank_based_on_gpu_memory(self, args, config, base_model, total_gpu_memory_size_in_GB, memory_summary_dict):

        total_gpu_memory_size_in_bytes = self._get_total_gpu_memory_size_in_bytes(args, total_gpu_memory_size_in_GB)
        base_model_portion = self._get_base_model_portion(args, config, base_model, memory_summary_dict)
        lora_portion = total_gpu_memory_size_in_bytes - base_model_portion

        return self._get_rank_based_on_lora_portion(args, config, base_model, lora_portion, memory_summary_dict)

    def _get_base_model_portion(self, args, config, model, memory_summary_dict):
        base_model_para_in_MB = self._get_base_model_para_in_MB(args, model)
        base_model_fwd_in_bytes = self._get_base_model_fwd_in_bytes(args, config)
        result = base_model_para_in_MB + base_model_fwd_in_bytes
        
        if memory_summary_dict is not None:
            memory_summary_dict['base_model_para_in_MB'] = self._bytes_to_mb(base_model_para_in_MB)
            memory_summary_dict['base_model_fwd_in_MB'] = self._bytes_to_mb(base_model_fwd_in_bytes)
            memory_summary_dict['base_model_portion_in_MB'] = self._bytes_to_mb(result)
        return result

    def _bytes_to_mb(self, bytes_value):
        return round(bytes_value / 1024 / 1024, 2)

    def _get_num_of_adapted_matrices(self, args):
        return 2 # A and B matrices

    def _get_num_of_modules_per_layer(self, args):
        return len(args.lora_target_modules)

    # TODO Liam
    def _get_rank_based_on_lora_portion(self, args, config, model, lora_portion, memory_summary_dict):
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
            elif optim_type == 'SGD':
                return 1
            raise NotImplementedError('unsupported optimizer type')

        def get_optimizer_state_mem(r, module_name, bytes_per_parameter, optim_type):
            return get_optimizer_state_count(optim_type) * get_param_mem(r, module_name, bytes_per_parameter)
            

        def get_gradient_mem(r, module_name, bytes_per_parameter):
            return get_param_mem(r, module_name, bytes_per_parameter)
        
        def get_fwd_betas(module_name):
            if is_normal_mod(module_name):
                return 1, 1.25 
            elif 'output.dense' in module_name:
                return 5, 1

        def get_forward_mem(r, module_name, bytes_per_parameter):
            
            beta1, beta2 = get_fwd_betas(module_name)
            return (beta1 * B * sequence_length * H + beta2 * B * sequence_length * r) * bytes_per_parameter

        
        
        D = H * C * bytes_per_parameter
       
        total_dim = 0
        total_layers = config.num_hidden_layers
        # currently it does not support regex TODO
        # need to support layer.0.query, query

        b1BSHb_sum = 0
        module_count = 0
        for name, module in model.named_modules():
            # TODO test
            matched_lora_target_module = None
            for lora_target_module in args.lora_target_modules:
                if lora_target_module in name:
                    matched_lora_target_module = lora_target_module
                    break
            
            if matched_lora_target_module is None:
                continue

            module_count += 1

            beta1, beta2 = get_fwd_betas(matched_lora_target_module)
            b2BSb =  beta2 * B * sequence_length * bytes_per_parameter
            if is_normal_mod(matched_lora_target_module):
                total_dim += (get_optimizer_state_count(args.optimizer) + 2) * D + b2BSb
            else:
                total_dim += (get_optimizer_state_count(args.optimizer) + 2) * D * mlp_ratio + b2BSb
            b1BSHb_sum += beta1 * B * sequence_length * H * bytes_per_parameter

        
        
        b1BSHb = beta1 * B * sequence_length * H * bytes_per_parameter
        return (lora_portion - b1BSHb * module_count) / (total_dim)
    
    def _get_total_gpu_memory_size_in_bytes(self, args, total_gpu_memory_size_in_GB):
        return total_gpu_memory_size_in_GB * 1024 * 1024 * 1024

    def _get_base_model_para_in_MB(self, args, base_model):
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
    
    # TODO Liam
    def _get_base_model_fwd_in_bytes(self, args, config):

        """
        B: batch size
        S: sequence length. number of tokens per image (patches + special tokens)
        D: hidden dimension
        dtype_bytes: bytes per parameter
        num_blocks: number of blocks
        num_heads: number of heads
        workspace_margin: workspace margin, 20% by default
        
        bytes_per_block = B * S * D * dtype_bytes
        total_forward = bytes_per_block * num_blocks
        attn_scores = B * num_heads * S * S * dtype_bytes
        peak_activations ≈ (total_forward + attn_scores) * (1 + workspace_margin)
        """




        batch_size = args.batch_size
        sequence_length = self._get_sequence_length(args, config)
        hidden_dimension = config.hidden_size
        num_layers = config.num_hidden_layers
        num_heads = config.num_attention_heads
        intermediate_size = config.intermediate_size
        dtype_bytes = self._get_byte_per_parameter(args.precision)
        workspace_margin = args.overhead_and_safety_margin_factor

        base_beta1, base_beta2 = 8, 49
        return (base_beta1 * batch_size * sequence_length * hidden_dimension + base_beta2 * batch_size * sequence_length * sequence_length * num_heads) * dtype_bytes

    def _get_sequence_length(self, args, config):
        config = AutoConfig.from_pretrained(args.model)
        CLS_TOKEN = args.CLS_TOKEN
        print(config.image_size)
        
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



