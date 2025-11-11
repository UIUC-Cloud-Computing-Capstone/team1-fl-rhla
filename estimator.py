FEDHELLO = 'FedHello'
OURS = 'Ours'

class RankEstimator:

    def get_rank_for_all_client_groups(self, args, model):
        rank_for_all_client_groups = []
        for i in range(len(args.heterogeneous_group)):
            total_gpu_memory_size_in_GB_for_one_client_group = args.gpu_memory_size_for_each_group_in_GB[i]
            upload_network_speed_in_Mbps_for_one_client_group = args.avg_upload_network_speed_for_each_group_in_Mbps[i]
            download_network_speed_in_Mbps_for_one_client_group = args.avg_download_network_speed_for_each_group_in_Mbps[i]
            desired_uploading_time_in_seconds_for_one_client_group = args.desired_uploading_time_for_each_group_in_seconds[i]
            desired_downloading_time_in_seconds_for_one_client_group = args.desired_downloading_time_for_each_group_in_seconds[i]
            rank_for_one_client_group = self._get_rank_for_one_client_group(args, model, total_gpu_memory_size_in_GB_for_one_client_group, upload_network_speed_in_Mbps_for_one_client_group, download_network_speed_in_Mbps_for_one_client_group, desired_uploading_time_in_seconds_for_one_client_group, desired_downloading_time_in_seconds_for_one_client_group)
            rank_for_all_client_groups.append(rank_for_one_client_group)
        print(f'rank budget per module for all client groups respectively: {str(rank_for_all_client_groups)}')
        return rank_for_all_client_groups

    def _get_rank_for_one_client_group(self, args, model, total_gpu_memory_size_in_GB, upload_network_speed_in_Mbps, download_network_speed_in_Mbps, desired_uploading_time_in_seconds, desired_downloading_time_in_seconds):
        if args.rank_estimator_method == FEDHELLO:
            return self._get_rank_based_on_gpu_memory(args, model, total_gpu_memory_size_in_GB)
        elif args.rank_estimator_method == OURS:
            return self._get_rank_based_on_all(args, model, total_gpu_memory_size_in_GB, upload_network_speed_in_Mbps, download_network_speed_in_Mbps, desired_uploading_time_in_seconds, desired_downloading_time_in_seconds)
        else:
            raise ValueError(f'Invalid rank estimator method: {args.rank_estimator_method}')

    def _get_rank_based_on_all(self, args, model, total_gpu_memory_size_in_GB, upload_network_speed_in_Mbps, download_network_speed_in_Mbps, desired_uploading_time_in_seconds, desired_downloading_time_in_seconds):
        rank_based_on_gpu_memory = self._get_rank_based_on_gpu_memory(args, model,  total_gpu_memory_size_in_GB)
        rank_based_on_upload_network_speed = self._get_rank_based_on_network_speed(args, model, upload_network_speed_in_Mbps, desired_uploading_time_in_seconds)
        rank_based_on_download_network_speed = self._get_rank_based_on_network_speed(args, model, download_network_speed_in_Mbps, desired_downloading_time_in_seconds)
        return self._get_final_rank(rank_based_on_gpu_memory, rank_based_on_upload_network_speed, rank_based_on_download_network_speed)

    def _get_final_rank(self, rank_based_on_gpu_memory, rank_based_on_upload_network_speed, rank_based_on_download_network_speed):
        # TODO Liam: add penalty? how?
        return min(rank_based_on_gpu_memory, rank_based_on_upload_network_speed, rank_based_on_download_network_speed)
    
    def _get_rank_based_on_gpu_memory(self, args, model, total_gpu_memory_size_in_GB):

        total_gpu_memory_size_in_bytes = self._get_total_gpu_memory_size_in_bytes(args, total_gpu_memory_size_in_GB)
        base_model_portion = self._get_base_model_portion(args, model)
        lora_portion = total_gpu_memory_size_in_bytes - base_model_portion

        return self._get_rank_based_on_lora_portion(args, model, lora_portion)

    def _get_base_model_portion(self, args, model):
        # parameter + activations + safety margin + optimizer states
        
        base_model_parameter_memory_size_in_bytes = self._get_base_model_parameter_memory_size_in_bytes(args, model)
        base_model_activations_and_safety_margin_memory_size_in_bytes = self._get_base_model_activations_and_safety_margin_memory_size_in_bytes(args)
        base_model_optimizer_states_memory_size_in_bytes = self._get_base_model_optimizer_states_memory_size_in_bytes(args, base_model_parameter_memory_size_in_bytes)
        result = base_model_parameter_memory_size_in_bytes + base_model_activations_and_safety_margin_memory_size_in_bytes + base_model_optimizer_states_memory_size_in_bytes
        print(f"base_model_parameter_memory_size_in_MB: {self._bytes_to_mb(base_model_parameter_memory_size_in_bytes)}")
        print(f"base_model_activations_and_safety_margin_memory_size_in_MB: {self._bytes_to_mb(base_model_activations_and_safety_margin_memory_size_in_bytes)}")
        print(f"base_model_optimizer_states_memory_size_in_MB: {self._bytes_to_mb(base_model_optimizer_states_memory_size_in_bytes)}")
        print(f"base_model_portion in MB estimated: {self._bytes_to_mb(result)}")
        return result

    def _bytes_to_mb(self, bytes_value):
        return bytes_value / 1024 / 1024

    def _get_rank_based_on_lora_portion(self, args, model, lora_portion):
        print(f"lora_portion_in_MB: {self._bytes_to_mb(lora_portion)}")
        if lora_portion <= 0:
            raise ValueError('GPU memory is too small to train the model')
        
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

        num_modules_per_layer = 2
        num_layers = 12
        H = self._get_hidden_dimension(args, model)
        def get_total_dimension_size(args, model):
            C = 2
            bytes_per_parameter = self._get_byte_per_parameter(args.precision)
            return C * num_modules_per_layer * H * num_layers * bytes_per_parameter

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

        def get_total_sequence_length_with_margin(args):
            sequence_length_per_batch = args.batch_size * self._get_sequence_length()
            
            dtype_bytes = self._get_byte_per_parameter(args.precision)
            workspace_margin = 0.2
            return sequence_length_per_batch * num_modules_per_layer * num_layers * dtype_bytes * (1 + workspace_margin)

        total_sequence_length_with_margin = get_total_sequence_length_with_margin(args)

        # (3) optimizer states memory size
        # if using adam, 
        # For numerical stability, these states are almost always stored in 32-bit precision (4 bytes), even if the model is being trained in 16-bit (fp16/bf16)
        # (3) optimizer states memory size = 2 * (1) parameter memory size if model is trained in fp32, 4 * (1) parameter memory size if model is trained in fp16.
        # multiplier is 2 for fp32, 4 for fp16.
        # optimizer states memory size = multiplier * a * r.

        # (4) total memory size
        # total memory size = (1) + (2) + (3)
        # total memory size = r * total_dimension_size + (h + r) * total_sequence_length_with_margin + multiplier * r * total_dimension_size = (total_dimension_size + total_sequence_length_with_margin + multiplier * total_dimension_size) * r + h * total_sequence_length_with_margin
        # r = (total memory size - h * total_sequence_length_with_margin) / (total_dimension_size + total_sequence_length_with_margin + multiplier * total_dimension_size)
        
        multiplier = 2
        result = int((lora_portion - H * total_sequence_length_with_margin) / (total_dimension_size + total_sequence_length_with_margin + multiplier * total_dimension_size))
        result = min(result, H) # cap the rank by the hidden dimension
        return result 
    
    def _get_hidden_dimension(self, args, model):
        # TODO Liam
        # return the hidden dimension of the model
        return 384

    def _get_total_gpu_memory_size_in_bytes(self, args, total_gpu_memory_size_in_GB):
        return total_gpu_memory_size_in_GB * 1024 * 1024 * 1024

    def _get_base_model_parameter_memory_size_in_bytes(self, args, model):
        '''
        model = AutoModelForImageClassification.from_pretrained('facebook/deit-small-patch16-224')
        '''
        
        parameter_size = 22_000_000 # TODO Abdul: Please check documentation and only include parameters of base model without LoRA
        
        byte_per_parameter = self._get_byte_per_parameter(args.precision)

        return parameter_size * byte_per_parameter

    def _get_byte_per_parameter(self, precision):
        if precision == 'fp32':
            return 4
        elif precision == 'fp16':
            return 2
        else:
            raise ValueError(f'Invalid precision: {precision}')
    
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
        batch_size = args.batch_size
        
        # we use facebook/deit-small-patch16-224
        
        batch_size = args.batch_size
        sequence_length = self._get_sequence_length()  # 197 for deit-small
        hidden_dimension = 384
        num_layers = 12
        num_heads = 6
        intermediate_size = hidden_dimension * 4  # 1536
        dtype_bytes = self._get_byte_per_parameter(args.precision)
        workspace_margin = 0.2
    
        # Per-layer activation memory breakdown:
        # 1. Input to layer (for residual): B × S × D
        input_activations = batch_size * sequence_length * hidden_dimension
        
        # 2. Attention output: B × S × D
        attn_output = batch_size * sequence_length * hidden_dimension
        
        # 3. MLP intermediate (after first dense, before second): B × S × 4D
        mlp_intermediate = batch_size * sequence_length * intermediate_size
        
        # 4. Attention scores (QK^T): B × num_heads × S × S
        attn_scores = batch_size * num_heads * sequence_length * sequence_length

        # 5. Attention probabilities: B × num_heads × S × S
        attn_probabilities = batch_size * num_heads * sequence_length * sequence_length
        
        # Total per layer
        activations_per_layer = (input_activations + attn_output + mlp_intermediate + attn_scores + attn_probabilities)
        
        # Peak memory = all layers simultaneously during backprop
        # Note: In practice, some activations can be recomputed (gradient checkpointing)
        # but for conservative estimate, assume all are stored
        peak_activations_all_layers = activations_per_layer * num_layers
        
        # Convert to bytes
        peak_activations_bytes = peak_activations_all_layers * dtype_bytes
        
        # Add workspace margin
        print(f"peak_activations_MB: {self._bytes_to_mb(peak_activations_bytes)}")
        return peak_activations_bytes * (1 + workspace_margin)

    def _get_sequence_length(self):
        #if model_name == 'facebook/deit-small-patch16-224':
        H = 224
        P = 16
        W = 224
        CLS_TOKEN = 1
        # number of patches ((H / P) × (W / P)) + CLS token
        return H / P * W / P + CLS_TOKEN
        
    def _get_base_model_optimizer_states_memory_size_in_bytes(self, args, base_model_memory_size_in_bytes):
        '''
        Optimizer states include the part for base model and the part for LoRA.
        This function only calculates the memory size for the base model portion.
        '''

        if args.optimizer == 'adamw' or args.optimizer == 'adam':
            # AdamW keeps 2 extra states (m, v) per parameter.
            return base_model_memory_size_in_bytes * 2
    
    def _get_safety_margin_memory_size_in_bytes(self, args, model, base_model_memory_size, activations_memory_size, optimizer_states_memory_size):
        if args.alpha is None:
            args.alpha = 0.2
        safety_margin_memory_size = (base_model_memory_size + activations_memory_size + optimizer_states_memory_size) * args.alpha
        return safety_margin_memory_size

    def _get_rank_based_on_network_speed(self, args, model,network_speed_in_Mbps, desired_communication_time_in_seconds):
        # TODO Abdul review and add unit test for this function

    
        bytes_per_second = network_speed_in_Mbps * 1_000_000 / 8
        parameter_size_in_bytes = desired_communication_time_in_seconds * bytes_per_second
        num_modules_per_layer = 2
        H = self._get_hidden_dimension(args, model)
        C = 2 # A and B matrices
        num_layers = 12
        bytes_per_parameter = self._get_byte_per_parameter(args.precision)
        total_dimension_size = C * num_modules_per_layer * H * num_layers * bytes_per_parameter
        rank = int(parameter_size_in_bytes / total_dimension_size)
        return rank

# TODO Liam: refactor heterogeneous_group0_lora etc in YAML

# Test cases
# Group 1: large memory size, extremely bad network -> FedHello will give higher rank, but our method will give lower rank
# Group 2: large memory size, good network -> both methods will give higher rank
# Group 3: small memory size, good network -> both methods will give lower rank


# For group 1:
# FedHello, suppose the training time per round is 1 minute, and the communitcation time per round is 1 minute due to bad network



