FEDHELLO = 'FedHello'
OURS = 'Ours'

class RankEstimator:

    def get_rank_for_all_client_groups(self, args, base_model):

        # TODO Liam: maybe remove this
        if args.model != 'facebook/deit-small-patch16-224':
            raise NotImplementedError(f'Invalid model: {args.model}. Only facebook/deit-small-patch16-224 is supported.')

        rank_for_all_client_groups = []
        for i in range(len(args.heterogeneous_group)):
            print(f"client group {i}")
            total_gpu_memory_size_in_GB_for_one_client_group = args.gpu_memory_size_for_each_group_in_GB[i]
            upload_network_speed_in_Mbps_for_one_client_group = args.avg_upload_network_speed_for_each_group_in_Mbps[i]
            download_network_speed_in_Mbps_for_one_client_group = args.avg_download_network_speed_for_each_group_in_Mbps[i]
            desired_uploading_time_in_seconds_for_one_client_group = args.desired_uploading_time_for_each_group_in_seconds[i]
            desired_downloading_time_in_seconds_for_one_client_group = args.desired_downloading_time_for_each_group_in_seconds[i]
            memory_summary_dict = {}
            rank_for_one_client_group = self._get_rank_for_one_client_group(args, base_model, total_gpu_memory_size_in_GB_for_one_client_group, upload_network_speed_in_Mbps_for_one_client_group, download_network_speed_in_Mbps_for_one_client_group, desired_uploading_time_in_seconds_for_one_client_group, desired_downloading_time_in_seconds_for_one_client_group, memory_summary_dict)
            rank_for_all_client_groups.append(rank_for_one_client_group)
            
            
            # TODO Liam: change
            memory_summary_dict['total_parameters_in_MB'] = memory_summary_dict['base_model_parameter_memory_size_in_MB'] + memory_summary_dict.get('lora_portion_parameter_size_in_MB', 0)
            memory_summary_dict['total_activations_gradients_and_with_safety_margin_in_MB'] = memory_summary_dict['base_model_activations_gradients_and_safety_margin_memory_size_in_MB'] + memory_summary_dict.get('lora_portion_activations_gradients_and_workspace_margin_in_MB', 0)
            memory_summary_dict['total_optimizer_states_in_MB'] = memory_summary_dict.get('base_model_optimizer_states_memory_size_in_MB', 0) + memory_summary_dict.get('lora_portion_optimizer_states_size_in_MB', 0)
            memory_summary_dict['total_memory_in_MB'] = round(memory_summary_dict['total_parameters_in_MB'] + memory_summary_dict['total_activations_gradients_and_with_safety_margin_in_MB'] + memory_summary_dict['total_optimizer_states_in_MB'], 2)
            
            self._print_memory_summary(memory_summary_dict)

            print('------------------------------------------------------------------------------------------------')
        
        
        print(f'rank budget per module for all client groups respectively: {str(rank_for_all_client_groups)}')
        
        
        
        return rank_for_all_client_groups

    def _print_memory_summary(self, memory_summary_dict):
        total_parameters_in_MB = memory_summary_dict['total_parameters_in_MB']
        total_activations_gradients_and_with_safety_margin_in_MB = memory_summary_dict['total_activations_gradients_and_with_safety_margin_in_MB']
        total_optimizer_states_in_MB = memory_summary_dict['total_optimizer_states_in_MB']
        total_memory_in_MB = memory_summary_dict['total_memory_in_MB']
        # TODO Liam
        print(f"Parameters: {total_parameters_in_MB} MB ({total_parameters_in_MB / total_memory_in_MB * 100:.2f}%)")
        print(f"Optimizer States: {total_optimizer_states_in_MB} MB ({total_optimizer_states_in_MB / total_memory_in_MB * 100:.2f}%)")
        print(f"Activations, Gradients and Safety Margin: {total_activations_gradients_and_with_safety_margin_in_MB} MB ({total_activations_gradients_and_with_safety_margin_in_MB / total_memory_in_MB * 100:.2f}%)")
        print(f"Total Memory: {total_memory_in_MB} MB ({total_memory_in_MB / total_memory_in_MB * 100:.2f}%)")

    def _get_rank_for_one_client_group(self, args, base_model, total_gpu_memory_size_in_GB, upload_network_speed_in_Mbps, download_network_speed_in_Mbps, desired_uploading_time_in_seconds, desired_downloading_time_in_seconds, memory_summary_dict):
        if args.rank_estimator_method == FEDHELLO:
            return self._get_rank_based_on_gpu_memory(args, base_model, total_gpu_memory_size_in_GB, memory_summary_dict)
        elif args.rank_estimator_method == OURS:
            return self._get_rank_based_on_all(args, base_model, total_gpu_memory_size_in_GB, upload_network_speed_in_Mbps, download_network_speed_in_Mbps, desired_uploading_time_in_seconds, desired_downloading_time_in_seconds, memory_summary_dict)
        else:
            raise ValueError(f'Invalid rank estimator method: {args.rank_estimator_method}')

    def _get_rank_based_on_all(self, args, base_model, total_gpu_memory_size_in_GB, upload_network_speed_in_Mbps, download_network_speed_in_Mbps, desired_uploading_time_in_seconds, desired_downloading_time_in_seconds, memory_summary_dict):
        rank_based_on_gpu_memory = self._get_rank_based_on_gpu_memory(args, base_model,  total_gpu_memory_size_in_GB, memory_summary_dict)
        rank_based_on_upload_network_speed = self._get_rank_based_on_network_speed(args, base_model, upload_network_speed_in_Mbps, desired_uploading_time_in_seconds)
        rank_based_on_download_network_speed = self._get_rank_based_on_network_speed(args, base_model, download_network_speed_in_Mbps, desired_downloading_time_in_seconds)
        return self._get_final_rank(rank_based_on_gpu_memory, rank_based_on_upload_network_speed, rank_based_on_download_network_speed)

    def _get_final_rank(self, rank_based_on_gpu_memory, rank_based_on_upload_network_speed, rank_based_on_download_network_speed):
        # TODO Liam
        return min(rank_based_on_gpu_memory, rank_based_on_upload_network_speed, rank_based_on_download_network_speed)
    
    def _get_rank_based_on_gpu_memory(self, args, base_model, total_gpu_memory_size_in_GB, memory_summary_dict):

        total_gpu_memory_size_in_bytes = self._get_total_gpu_memory_size_in_bytes(args, total_gpu_memory_size_in_GB)
        base_model_portion = self._get_base_model_portion(args, base_model, memory_summary_dict)
        lora_portion = total_gpu_memory_size_in_bytes - base_model_portion

        return self._get_rank_based_on_lora_portion(args, base_model, lora_portion, memory_summary_dict)

    def _get_base_model_portion(self, args, model, memory_summary_dict):
        
        
        # TODO Liam
        base_model_parameter_memory_size_in_bytes = self._get_base_model_parameter_memory_size_in_bytes(args, model)

        gradient_memory_bytes = base_model_parameter_memory_size_in_bytes * args.percentage_of_layers_in_memory

        base_model_activations_and_safety_margin_memory_size_in_bytes = self._get_base_model_activations_and_safety_margin_memory_size_in_bytes(args)
        base_model_optimizer_states_memory_size_in_bytes = self._get_base_model_optimizer_states_memory_size_in_bytes(args, base_model_parameter_memory_size_in_bytes)
        result = base_model_parameter_memory_size_in_bytes + base_model_activations_and_safety_margin_memory_size_in_bytes + gradient_memory_bytes + base_model_optimizer_states_memory_size_in_bytes
        
        if memory_summary_dict is not None:
            memory_summary_dict['base_model_parameter_memory_size_in_MB'] = self._bytes_to_mb(base_model_parameter_memory_size_in_bytes)
            memory_summary_dict['base_model_activations_gradients_and_safety_margin_memory_size_in_MB'] = self._bytes_to_mb(base_model_activations_and_safety_margin_memory_size_in_bytes) + self._bytes_to_mb(gradient_memory_bytes)
            memory_summary_dict['base_model_portion_in_MB'] = self._bytes_to_mb(result)
        return result

    def _bytes_to_mb(self, bytes_value):
        return round(bytes_value / 1024 / 1024, 2)

    def _get_num_of_adapted_matrices(self, args):
        return 2 # A and B matrices

    def _get_num_of_modules_per_layer(self, args):
        return len(args.lora_target_modules)

    # TODO Liam
    def _get_rank_based_on_lora_portion(self, args, model, lora_portion, memory_summary_dict):
        if lora_portion <= 0:
            print(f'Warning: GPU memory is too small to train the model')
            return 0

        B = args.batch_size
        mlp_ratio = args.mlp_ratio if args.mlp_ratio else 4
        H = self._get_hidden_dimension(args)
        sequence_length = self._get_sequence_length(args)
        args.lora_target_modules
        bytes_per_parameter = self._get_byte_per_parameter(args.precision)

    
        def is_normal_mod(module_name):
            return 'query' in module_name or 'key' in module_name or 'value' in module_name or 'attention.output.dense' in module_name or 'intermediate.dense' in module_name

        C = 2 # two matrices A and B
        
        def get_param_mem(r, module_name, bytes_per_parameter):
            
            
            if is_normal_mod(module_name):
                return H * r * C * bytes_per_parameter
            elif 'output.dense' in module_name:
                return mlp_ratio * H * r * C * bytes_per_parameter

        def get_optimizer_state_count(optim_type):
            if optim_type == 'Adam' or optim_type == 'Adamw':
                return 2
            elif optim_type == 'SGD':
                return 1
            raise NotImplementedError('unsupported optimizer type')

        def get_optimizer_state_mem(r, module_name, bytes_per_parameter, optim_type):
            return get_optimizer_state_count(optim_type) * get_param_mem(r, module_name, bytes_per_parameter)
            

        def get_gradient_mem(r, module_name, bytes_per_parameter):
            return get_param_mem(r, module_name, bytes_per_parameter)
        
        def get_fwd_betas(module_name):
            # TODO
            if is_normal_mod(module_name):
                return 1, 1.25 
            elif 'output.dense' in module_name:
                return 5, 1

        def get_forward_mem(r, module_name, bytes_per_parameter):
            
            beta1, beta2 = get_fwd_betas(module_name)
            return (beta1 * B * sequence_length * H + beta2 * B * sequence_length * r) * bytes_per_parameter

        
        
        D = H * C * bytes_per_parameter
        mD = mlp_ratio * D
        opt = get_optimizer_state_count(args.optimizer) * D
        grad = D
        beta1, beta2 = get_fwd_betas(module_name)
        b2BSb =  beta2 * B * sequence_length * bytes_per_parameter
        # TODO wrong: D + mD
        #total_dim = D + mD + opt + grad + grad + b2BSb
        total_dim = 0
        total_layers = 12
        # currently it does not support regex TODO
        # need to support layer.0.query, query
        

        for module in args.lora_target_modules:
            if is_normal_mod(module_name):
                total_dim += D + opt + grad
            else:
                total_dim += mD + opt + grad
        
        
        b1BSHb = beta1 * B * sequence_length * H * bytes_per_parameter
        return (lora_portion - b1BSHb) / (total_dim)



    def _get_rank_based_on_lora_portion_v1(self, args, model, lora_portion, memory_summary_dict):
        if lora_portion <= 0:
            print(f'Warning: GPU memory is too small to train the model')
            return 0
        
        # get rank based on lora_portion
        # lora_portion includes (1) parameter size, (2) activations and safety margin size, and (3) optimizer states size.
        
        # The way we achieve dynamic rank adjustment is masking.

        # (1) parameter memory size
        # Suppose we enable m number of modules on one layer.
        # Let r be the rank of one module.
        # Layer total rank is r * m.
        # Number of layers is L.
        # Hidden dimension is H.
        # Bytes per parameter is 4 for fp32, 2 for fp16.
        # C is the number of adapted matrices (A and B, so C = 2).
        # Parameter memory size is r * (C  * m * H * L * bytes per parameter).
        # Let total_dimension_size = C  * m * H * L * bytes per parameter.
        # Parameter memory size is r * total_dimension_size.

        num_modules_per_layer = self._get_num_of_modules_per_layer(args)
        H = self._get_hidden_dimension(args)
        def get_total_dimension_size(args, model):
            C = self._get_num_of_adapted_matrices(args)
            bytes_per_parameter = self._get_byte_per_parameter(args.precision)
            return C * num_modules_per_layer * H * args.num_of_layers_to_allocate_LoRA * bytes_per_parameter

        total_dimension_size = get_total_dimension_size(args, model)

        # (2) activations and safety margin memory size
        # sequence_length_per_batch = batch size * sequence_length
        # input_to_lora = sequence_length_per_batch * hidden_dimension
        # intermediate_lora = sequence_length_per_batch * r
        # activations_per_module = input_to_lora + intermediate_lora
        # peak_activations_bytes = activations_per_module * num_modules_per_layer * num_layers * dtype_bytes * (1 + workspace_margin)
        # = (hidden_dimension + r) * sequence_length_per_batch * num_modules_per_layer * num_layers * dtype_bytes * (1 + workspace_margin)
        # Let total_sequence_length_with_margin = sequence_length_per_batch * num_modules_per_layer * num_layers * dtype_bytes * (1 + workspace_margin).
        # peak_activations_bytes = (hidden_dimension + r) * total_sequence_length_with_margin.

        workspace_margin = args.overhead_and_safety_margin_factor
        def get_total_sequence_length_with_margin(args):
            sequence_length_per_batch = args.batch_size * self._get_sequence_length(args)
            
            dtype_bytes = self._get_byte_per_parameter(args.precision)
            num_layers = args.num_of_layers_to_allocate_LoRA
            return sequence_length_per_batch * num_modules_per_layer * num_layers * dtype_bytes * (1 + workspace_margin)

        total_sequence_length_with_margin = get_total_sequence_length_with_margin(args)

        # (3) optimizer states memory size
        # if using adam, 
        # For numerical stability, these states are almost always stored in 32-bit precision (4 bytes), even if the model is being trained in 16-bit (fp16/bf16)
        # (3) optimizer states memory size = 2 * (1) parameter memory size if model is trained in fp32, 4 * (1) parameter memory size if model is trained in fp16.
        # multiplier is 2 for fp32, 4 for fp16.
        # optimizer states memory size = multiplier * total_dimension_size * r.
        def get_multiplier(args):
            if args.precision == 'fp32':
                return 2
            elif args.precision == 'fp16':
                return 4
            else:
                raise ValueError(f'Invalid precision: {args.precision}')
        # (4) gradient memory size
        gradient_percentage = args.percentage_of_layers_in_memory
        # gradient_memory_size = r * total_dimension_size * gradient_percentage

        # (5) total memory size
        # total memory size = (1) + (2) + (3) + (4)
        # total memory size = r * total_dimension_size * (1 + gradient_percentage) + (h + r) * total_sequence_length_with_margin + multiplier * r * total_dimension_size
        # r = (total memory size - h * total_sequence_length_with_margin) / (total_dimension_size + total_sequence_length_with_margin + multiplier * total_dimension_size)
        
        multiplier = get_multiplier(args)
        result = int((lora_portion - H * total_sequence_length_with_margin) / (total_dimension_size * (1 + gradient_percentage) + total_sequence_length_with_margin + multiplier * total_dimension_size))
        result = min(result, H) # cap the rank by the hidden dimension
        
        # print the result in MB
        # Parameter memory size is r * total_dimension_size.
        lora_portion_parameter_size = result * total_dimension_size
        lora_portion_parameter_size_in_MB = self._bytes_to_mb(lora_portion_parameter_size)
        if memory_summary_dict is not None:
            memory_summary_dict['lora_portion_parameter_size_in_MB'] = lora_portion_parameter_size_in_MB

        # peak_activations_bytes = (hidden_dimension + r) * total_sequence_length_with_margin.
        lora_portion_activations_size = (H + result) * total_sequence_length_with_margin
        lora_portion_activations_size_in_MB = self._bytes_to_mb(lora_portion_activations_size)
        if memory_summary_dict is not None:
            memory_summary_dict['lora_portion_activations_size_in_MB_with_workspace_margin'] = lora_portion_activations_size_in_MB
        lora_portion_activations_size_in_MB /= (1 + workspace_margin)
        gradient_memory_size_in_MB = lora_portion_parameter_size_in_MB * gradient_percentage
        if memory_summary_dict is not None:
            memory_summary_dict['lora_portion_gradient_size_in_MB'] = gradient_memory_size_in_MB
            memory_summary_dict['lora_portion_activations_size_in_MB'] = lora_portion_activations_size_in_MB
            memory_summary_dict['lora_portion_activations_gradients_and_workspace_margin_in_MB'] = lora_portion_activations_size_in_MB * (1 + workspace_margin) + gradient_memory_size_in_MB
        # optimizer states memory size = multiplier * total_dimension_size * r.
        lora_portion_optimizer_states_size = result * multiplier * total_dimension_size
        lora_portion_optimizer_states_size_in_MB = self._bytes_to_mb(lora_portion_optimizer_states_size)
        if memory_summary_dict is not None:
            memory_summary_dict['lora_portion_optimizer_states_size_in_MB'] = lora_portion_optimizer_states_size_in_MB
        
        if result < 0:
            print(f'Warning: Memory is too small to train the model with positive rank')
            result = 0
        return result 
    
    # TODO Liam refactor
    def _get_hidden_dimension(self, args):
        if args.model == 'facebook/deit-small-patch16-224':
            return 384 # TODO Liam read from actual model
        else:
            raise NotImplementedError(f'Invalid model: {args.model}. Only facebook/deit-small-patch16-224 is supported.')

    def _get_total_gpu_memory_size_in_bytes(self, args, total_gpu_memory_size_in_GB):
        return total_gpu_memory_size_in_GB * 1024 * 1024 * 1024

    # TODO Liam refactor
    def _get_base_model_parameter_memory_size_in_bytes(self, args, base_model):
        '''
        base_model = AutoModelForImageClassification.from_pretrained('facebook/deit-small-patch16-224')
        '''
        
        parameter_size = 22_000_000 # TODO Liam: test
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

    def _get_num_of_layers(self, args):
        if args.model == 'facebook/deit-small-patch16-224':
            return 12
        else:
            raise NotImplementedError(f'Invalid model: {args.model}. Only facebook/deit-small-patch16-224 is supported.')

    def _get_num_of_heads(self, args):
        if args.model == 'facebook/deit-small-patch16-224':
            return 6
        else:
            raise NotImplementedError(f'Invalid model: {args.model}. Only facebook/deit-small-patch16-224 is supported.')
    
    # TODO Liam
    def _get_base_model_activations_and_safety_margin_memory_size_in_bytes(self, args):

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


        if args.model != 'facebook/deit-small-patch16-224':
            raise NotImplementedError(f'Invalid model: {args.model}. Only facebook/deit-small-patch16-224 is supported.')

        batch_size = args.batch_size
        sequence_length = self._get_sequence_length(args)  # 197 for deit-small
        hidden_dimension = self._get_hidden_dimension(args)
        num_layers = args.percentage_of_layers_in_memory * self._get_num_of_layers(args)
        num_heads = self._get_num_of_heads(args)
        intermediate_size = hidden_dimension * 4
        dtype_bytes = self._get_byte_per_parameter(args.precision)
        workspace_margin = args.overhead_and_safety_margin_factor

        base_beta1, base_beta2 = 8, 49 # obtained through profiling + regression, hard coded for now TODO
        return (base_beta1 * batch_size * sequence_length * hidden_dimension + base_beta2 * batch_size * sequence_length * sequence_length * num_heads) * dtype_bytes

    def _get_sequence_length(self, args):
        if args.model != 'facebook/deit-small-patch16-224':
            raise NotImplementedError(f'Invalid model: {args.model}. Only facebook/deit-small-patch16-224 is supported.')
        CLS_TOKEN = 1
        return args.image_height / args.patch_size * args.image_width / args.patch_size + CLS_TOKEN
        
    def _get_base_model_optimizer_states_memory_size_in_bytes(self, args, base_model_memory_size_in_bytes):
        '''
        Optimizer states include the part for base model and the part for LoRA.
        This function only calculates the memory size for the base model portion.
        '''

        if not args.train_classifier:
            return 0
        else:
            raise NotImplementedError('Not implemented yet.')
    
    # TODO Liam
    def _get_safety_margin_memory_size_in_bytes(self, args, model, base_model_memory_size, activations_memory_size, optimizer_states_memory_size):
        safety_margin_memory_size = (base_model_memory_size + activations_memory_size + optimizer_states_memory_size) * args.overhead_and_safety_margin_factor
        return safety_margin_memory_size

    def _get_rank_based_on_network_speed(self, args, model,network_speed_in_Mbps, desired_communication_time_in_seconds):
        bytes_per_second = network_speed_in_Mbps * 1_000_000 / 8
        parameter_size_in_bytes = desired_communication_time_in_seconds * bytes_per_second
        num_modules_per_layer = self._get_num_of_modules_per_layer(args)
        H = self._get_hidden_dimension(args)
        C = self._get_num_of_adapted_matrices(args)
        num_layers = args.num_of_layers_to_allocate_LoRA
        bytes_per_parameter = self._get_byte_per_parameter(args.precision)
        total_dimension_size = C * num_modules_per_layer * H * num_layers * bytes_per_parameter
        rank = int(parameter_size_in_bytes / total_dimension_size)
        return rank



