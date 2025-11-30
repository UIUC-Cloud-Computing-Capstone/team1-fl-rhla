import torch
from torch.nn import Module
from torch.utils.data import DataLoader
from torch.distributions import Categorical
import re
import sys
import time
from torch import Tensor
from typing import Dict, Optional
from sklearn.cluster import KMeans
import numpy as np

def move_dict_to_device(dict_of_tensors, device):
    """
    Move a dictionary of tensors to the specified device.
    
    Args:
        dict_of_tensors (dict): Dictionary containing tensors.
        device (str or torch.device): Device to move tensors to, e.g., 'cuda:0', 'cpu'.
    
    Returns:
        dict: Dictionary containing tensors moved to the specified device.
    """
    # Iterate over the items in the dictionary
    for key, tensor in dict_of_tensors.items():
        # Move each tensor to the specified device
        dict_of_tensors[key] = tensor.to(device)
    
    return dict_of_tensors

class FIMCalculator:
    def __init__(self, args, model, test_data):
        self.model_name = 'test_model_name'
        self.model = model
        self.test_data = test_data
        self.device = args.accelerator.device
        self.model.to(self.device)
        self.num_samples = len(self.test_data) 
        self.args = args

    def compute_fim(self, empirical=True, verbose=True, every_n=None):
        # ipdb.set_trace()
        all_fims = self.fim_diag(self.model, 
                                 self.test_data, 
                                 samples_no=self.num_samples, 
                                 empirical=empirical, 
                                 device=self.device, 
                                 verbose=verbose, 
                                 every_n=every_n)
        
        fim_diag_by_layer = self.aggregate_fisher_information(all_fims, self.model_name, self.args.lora_layer)
        return fim_diag_by_layer

    @staticmethod
    def fim_diag(model: Module,
                 data_loader: DataLoader,
                 samples_no: int = None,
                 empirical: bool = False,
                 device: torch.device = None,
                 verbose: bool = False,
                 every_n: int = None) -> Dict[int, Dict[str, Tensor]]:
        model.eval()
        fim = {}
        for name, param in model.named_parameters():
            if param.requires_grad and 'classifier' not in name:
                fim[name] = torch.zeros_like(param)
        
        seen_no = 0
        last = 0
        tic = time.time()

        all_fims = dict({})
        while samples_no is None or seen_no < samples_no:
            data_iterator = iter(data_loader)
            try:
                batch = next(data_iterator)
            except StopIteration:
                if samples_no is None:
                    break
                data_iterator = iter(data_loader)
                batch = next(data_iterator)
            move_dict_to_device(batch, device)
            logits = model(**batch).logits

            if empirical:
                outdx = batch['labels'].unsqueeze(1)
            else:
                outdx = Categorical(logits=logits).sample().unsqueeze(1).detach()
            outdx = outdx.to(torch.int64) # added to fix stsb error
            samples = logits.gather(1, outdx)

            if 'pixel_values' in batch:
                idx, batch_size = 0, batch['pixel_values'].size(0)
            elif 'input_ids' in batch:
                idx, batch_size = 0, batch['input_ids'].size(0)

            while idx < batch_size and (samples_no is None or seen_no < samples_no):
                model.zero_grad()
                torch.autograd.backward(samples[idx], retain_graph=True)
                for name, param in model.named_parameters():
                    if param.requires_grad and 'classifier' not in name:
                        fim[name] += (param.grad * param.grad)
                        fim[name].detach_()
                seen_no += 1
                idx += 1

                if verbose and seen_no % 100 == 0:
                    toc = time.time()
                    fps = float(seen_no - last) / (toc - tic)
                    tic, last = toc, seen_no
                    sys.stdout.write(f"\rSamples: {seen_no:5d}. Fps: {fps:2.4f} samples/s.")

                if None and seen_no % every_n == 0:
                    all_fims[seen_no] = {n: f.clone().div_(seen_no).detach_()
                                        for (n, f) in fim.items()}

        if verbose:
            if seen_no > last:
                toc = time.time()
                fps = float(seen_no - last) / (toc - tic)
            sys.stdout.write(f"\rSamples: {seen_no:5d}. Fps: {fps:2.5f} samples/s.\n")

        for name, grad2 in fim.items():
            grad2 /= float(seen_no)

        all_fims[seen_no] = fim

        return all_fims

    @staticmethod
    def aggregate_fisher_information(all_fims, model_name, lora_layer):
        latest_fim_diag = all_fims[max(all_fims.keys())]
        fim_diag_by_layer = [0]*lora_layer
        
        #print(latest_fim_diag)
        #print('##################################################')
        for param_name, param_fim_diag in latest_fim_diag.items():
            # TODO (Done): check the layer name actually follow this convention.
            layer_name = int(re.search(r'\.layer\.(\d+)\.', param_name).group(1))
            # layer_name is the model index
            # combined lora_A lora_B gradient
            fim_diag_by_layer[layer_name] += torch.norm(param_fim_diag, p='fro').pow(2).item()
        return fim_diag_by_layer

    @staticmethod
    def bottom_k_layers(input_list, k):
        sorted_items = sorted(input_list)
        keys = [sorted_items]

        values = np.array([v for v in input_list], dtype=float).reshape(-1, 1)

        kmeans = KMeans(n_clusters=3, random_state=0)
        kmeans.fit(values)

        # Relabel clusters by ascending centroid
        centers = kmeans.cluster_centers_.ravel()          # shape: (n_clusters,)
        order = np.argsort(centers)                        # e.g., [2, 0, 1]
        remap = {old: new for new, old in enumerate(order)}# smallest center -> 0, etc.
        cluster_labels = np.array([remap[l] for l in kmeans.labels_], dtype=int)
        # TODO (Remap done): the label needs to associated with the fim score! layer with smallest fim score should be label-2. So with the highest probablity to be selected as the blocked layer.
        return keys, cluster_labels

