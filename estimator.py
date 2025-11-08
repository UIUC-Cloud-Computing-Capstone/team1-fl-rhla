FEDHELLO = 'FedHello'
OURS = 'Ours'

class RankEstimator:


    def get_rank(self, args, model, total_gpu_memory_size_in_GB, upload_network_speed_in_Mbps, download_network_speed_in_Mbps):
        if args.rank_estimator_method == FEDHELLO:
            return self._get_rank_based_on_gpu_memory(args, model, total_gpu_memory_size_in_GB)
        elif args.rank_estimator_method == OURS:
            return self._get_rank_based_on_all(args, model, total_gpu_memory_size_in_GB, upload_network_speed_in_Mbps, download_network_speed_in_Mbps)
        else:
            raise ValueError(f'Invalid rank estimator method: {args.rank_estimator_method}')

    def _get_rank_based_on_all(self, args, model, total_gpu_memory_size_in_GB, upload_network_speed_in_Mbps, download_network_speed_in_Mbps):
        rank_based_on_gpu_memory = self._get_rank_based_on_gpu_memory(args, model,  total_gpu_memory_size_in_GB)
        rank_based_on_upload_network_speed = self._get_rank_based_on_upload_network_speed(args, upload_network_speed_in_Mbps)
        rank_based_on_download_network_speed = self._get_rank_based_on_download_network_speed(args, download_network_speed_in_Mbps)
        return self._get_final_rank(rank_based_on_gpu_memory, rank_based_on_upload_network_speed, rank_based_on_download_network_speed)

    def _get_final_rank(self, rank_based_on_gpu_memory, rank_based_on_upload_network_speed, rank_based_on_download_network_speed):
        # TODO Liam: add penalty? how?
        return min(rank_based_on_gpu_memory, rank_based_on_upload_network_speed, rank_based_on_download_network_speed)
    
    def _get_rank_based_on_gpu_memory(self, args, model, total_gpu_memory_size_in_GB):

        total_gpu_memory_size_in_bytes = self._get_total_gpu_memory_size_in_bytes(args, total_gpu_memory_size_in_GB)
        base_model_portion = self._get_base_model_portion(args, model)
        lora_portion = total_gpu_memory_size_in_bytes - base_model_portion

        return self._get_rank_based_on_lora_portion(lora_portion)

    def _get_base_model_portion(self, args, model):
        # parameter + activations + safety margin + optimizer states
        
        base_model_parameter_memory_size_in_bytes = self._get_base_model_parameter_memory_size_in_bytes(args, model)
        base_model_activations_and_safety_margin_memory_size_in_bytes = self._get_base_model_activations_and_safety_margin_memory_size_in_bytes(args)
        base_model_optimizer_states_memory_size_in_bytes = self._get_base_model_optimizer_states_memory_size_in_bytes(args, base_model_parameter_memory_size_in_bytes)
        return base_model_parameter_memory_size_in_bytes + base_model_activations_and_safety_margin_memory_size_in_bytes + base_model_optimizer_states_memory_size_in_bytes

    def _get_rank_based_on_lora_portion(self, lora_portion):
        # TODO Liam: implement this
        # get rank based on lora_portion
        # lora_portion includes (1) parameter size, (2) activations and safety margin size, and (3) optimizer states size.
        
        # The way we achieve dynamic rank adjustment is using truncation.
        # Truncating rank can save (2) activations and (3) optimizer states size, but it will not save (1) parameter size. 




        
        pass

    def _get_total_gpu_memory_size_in_bytes(self, args, total_gpu_memory_size_in_GB):
        return total_gpu_memory_size_in_GB * 1024 * 1024 * 1024

    def _get_base_model_parameter_memory_size_in_bytes(self, args, model):
        '''
        model = AutoModelForImageClassification.from_pretrained('facebook/deit-small-patch16-224').state_dict()
        '''
        
        parameter_size = 0 # TODO Abdul: Please check documentation and only include parameters of base model without LoRA
        
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
        # TODO Liam
        # is this correct?
        # sanity check
        # unit test

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
        
        sequence_length = self._get_sequence_length()
        hidden_dimension = 384
        dtype_bytes = self._get_byte_per_parameter(args.precision)
        workspace_margin = 0.2
        
        # method 1
        #intermediate_size = hidden_dimension * 4
        #A_resid = batch_size * sequence_length * hidden_dimension * dtype_bytes
        #A_mlp = batch_size * sequence_length * intermediate_size * dtype_bytes
        #peak_activations = (A_resid + A_mlp) * (1 + workspace_margin)
        
        # method 2
        #num_blocks = 12
        #num_heads = 6
        # bytes_per_block = batch_size * sequence_length * hidden_dimension * dtype_bytes
        # total_forward = bytes_per_block * num_blocks
        # attn_scores = batch_size * num_heads * sequence_length * sequence_length * dtype_bytes
        # peak_activations = (total_forward + attn_scores) * (1 + workspace_margin)

        # 500 MB
        peak_activations = 500 * 1024 * 1024
        return peak_activations * (1 + workspace_margin)

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

        if args.optimizer == 'adamw':
            # AdamW keeps 2 extra states (m, v) per parameter.
            return base_model_memory_size_in_bytes * 2
    
    def _get_safety_margin_memory_size_in_bytes(self, args, model, base_model_memory_size, activations_memory_size, optimizer_states_memory_size):
        if args.alpha is None:
            args.alpha = 0.2
        safety_margin_memory_size = (base_model_memory_size + activations_memory_size + optimizer_states_memory_size) * args.alpha
        return safety_margin_memory_size

    def _get_rank_based_on_upload_network_speed(self, args, upload_network_speed_in_Mbps):
        # TODO Abdul


        # 1. Based on which group this client belongs to, desired_uploading_time_for_each_group_in_seconds and upload_network_speed_in_Mbps, get parameter_size_in_bytes
        # 2. Based on the parameter_size_in_bytes, and args.precision, get rank
        # 3. add unit test for this function
        pass

    def _get_rank_based_on_download_network_speed(self, args, download_network_speed_in_Mbps):
        # TODO Abdul
        # 1. Based on which group this client belongs to, desired_downloading_time_for_each_group_in_seconds and download_network_speed_in_Mbps, get parameter_size_in_bytes
        # 2. Based on the parameter_size_in_bytes, and args.precision, get rank
        # 3. add unit test for this function
        pass

# TODO Liam: refactor heterogeneous_group0_lora etc in YAML

# Test cases
# Group 1: large memory size, extremely bad network -> FedHello will give higher rank, but our method will give lower rank
# Group 2: large memory size, good network -> both methods will give higher rank
# Group 3: small memory size, good network -> both methods will give lower rank


# For group 1:
# FedHello, suppose the training time per round is 1 minute, and the communitcation time per round is 1 minute due to bad network



