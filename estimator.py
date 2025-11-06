FEDHELLO = 'FedHello'
OURS = 'Ours'

class RankEstimator:


    def get_rank(self, method, model, total_gpu_memory_size, upload_network_speed, download_network_speed):
        if method == FEDHELLO:
            return self._get_rank_based_on_gpu_memory(model, total_gpu_memory_size)
        elif method == OURS:
            return self._get_rank_based_on_all(model, total_gpu_memory_size, upload_network_speed, download_network_speed)
        else:
            raise ValueError(f'Invalid method: {method}')

    def _get_rank_based_on_all(self, model, total_gpu_memory_size, upload_network_speed, download_network_speed):
        rank_based_on_gpu_memory = self._get_rank_based_on_gpu_memory(total_gpu_memory_size, model)
        rank_based_on_upload_network_speed = self._get_rank_based_on_upload_network_speed(upload_network_speed)
        rank_based_on_download_network_speed = self._get_rank_based_on_download_network_speed(download_network_speed)
        return self._get_final_rank(rank_based_on_gpu_memory, rank_based_on_upload_network_speed, rank_based_on_download_network_speed)

    def _get_final_rank(self, rank_based_on_gpu_memory, rank_based_on_upload_network_speed, rank_based_on_download_network_speed):
        # TODO Liam: add penalty? how?
        return min(rank_based_on_gpu_memory, rank_based_on_upload_network_speed, rank_based_on_download_network_speed)
    
    def _get_rank_based_on_gpu_memory(self, model, total_gpu_memory_size):
        # TODO Liam: implement this
        pass

    def _get_rank_based_on_upload_network_speed(self, upload_network_speed):
        # TODO Abdul


        # 1. Based on which group this client belongs to, desired_uploading_time_for_each_group_in_seconds and avg_upload_network_speed_for_each_group_in_Mbps, get parameter_size_in_bytes
        # 2. Based on the parameter_size_in_bytes, and precision, get rank
        # 3. add unit test for this function
        pass

    def _get_rank_based_on_download_network_speed(self, download_network_speed):
        # TODO Abdul
        pass

# TODO Liam: refactor heterogeneous_group0_lora etc in YAML

# TODO Liam: consider how to init rank properly

# Test cases
# Group 1: large memory size, extremely bad network -> FedHello will give higher rank, but our method will give lower rank
# Group 2: large memory size, good network -> both methods will give higher rank
# Group 3: small memory size, good network -> both methods will give lower rank


# For group 1:
# FedHello, suppose the training time per round is 1 minute, and the communitcation time per round is 1 minute due to bad network



