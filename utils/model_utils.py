from model.cnn import CNNFmnist, CNNFEmnist, CNNSvhn, CNNCifar
from model.mlp import MLP
from model.recurrent import CharLSTM, RNN_FedShakespeare
from model.vgg import vgg19_bn
from model.resnet import ResNet9FashionMNIST, ResNet18, ReducedResNet18, \
                         ResNet34, ResNet50, ResNet101, ResNet152, \
                         CIFARResNet20, SVHNResNet20
# from model.bert import Bert
from transformers import AutoModelForCausalLM, BloomForSequenceClassification, AutoTokenizer, default_data_collator, get_linear_schedule_with_warmup
# from peft import get_peft_model, PromptTuningInit, PromptTuningConfig, TaskType
# from peft import PeftModel, PeftConfig
from peft import LoraConfig, get_peft_model, LoKrConfig

import torch
import torch.nn as nn
import copy

from transformers import AutoModelForImageClassification, AutoModelForSequenceClassification, AutoModelForMultipleChoice
from peft import LoraConfig, get_peft_model

################################### model setup ########################################
def model_setup(args):
    """
    Set up and configure models for federated learning with LoRA (Low-Rank Adaptation).
    
    This function initializes different types of pre-trained models and applies LoRA configuration
    for efficient fine-tuning in federated learning scenarios. It supports both text classification
    (BERT) and image classification (Vision Transformer) models.
    
    Args:
        args: Configuration object containing model parameters. Expected attributes:
            - model (str): Model identifier ('bert-base-uncased' or 'google/vit-base-patch16-224-in21k')
            - num_classes (int): Number of output classes for classification
            - device (torch.device): Device to move the model to (CPU/GPU)
            - label2id (dict): Mapping from labels to IDs (for ViT models)
            - id2label (dict): Mapping from IDs to labels (for ViT models)
    
    Returns:
        tuple: A tuple containing:
            - args: The input arguments (unchanged)
            - net_glob: The configured model with LoRA adapters applied
            - global_model: Deep copy of the model's state dictionary for federated learning
            - model_dim: Total number of parameters in the model
    
    Supported Models:
        1. BERT ('bert-base-uncased'):
           - Uses AutoModelForSequenceClassification for text classification
           - LoRA config: r=6, alpha=6, targets query/value modules
           - Dropout: 0.1, no bias adaptation
        
        2. Vision Transformer ('google/vit-base-patch16-224-in21k'):
           - Uses facebook/deit-small-patch16-224 as base model
           - LoRA config: r=48, alpha=48, targets query/value modules
           - Dropout: 0.1, no bias adaptation
           - Saves classifier module for fine-tuning
    
    Raises:
        SystemExit: If an unrecognized model is specified
    
    Example:
        >>> args = argparse.Namespace()
        >>> args.model = 'bert-base-uncased'
        >>> args.num_classes = 10
        >>> args.device = torch.device('cuda')
        >>> args, model, global_state, dim = model_setup(args)
    """
    if args.model == 'bert-base-uncased':
        model = AutoModelForSequenceClassification.from_pretrained(
            'google/bert_uncased_L-12_H-128_A-2', # https://huggingface.co/google/bert_uncased_L-4_H-256_A-4
            num_labels=args.num_classes
        )
        config = LoraConfig(
            r=args.lora_max_rank,
            lora_alpha=args.lora_max_rank,
            target_modules=["query", "value"],
            lora_dropout=0.1,
            bias="none"
        )

        if args.LOKR:
            config = LoKrConfig(
                r=args.lora_max_rank,
                alpha=args.lora_alpha,
                target_modules=["query", "value"],
            )

        net_glob = get_peft_model(model, config)
        net_glob.to(args.device)
    elif args.model == 'google/vit-base-patch16-224-in21k':
        model = AutoModelForImageClassification.from_pretrained(
            'facebook/deit-small-patch16-224',
            label2id=args.label2id,
            id2label=args.id2label,
            ignore_mismatched_sizes=True,  # provide this in case you're planning to fine-tune an already fine-tuned checkpoint
        )

        config = LoraConfig(
            r=args.lora_max_rank,
            lora_alpha=args.lora_max_rank,
            target_modules=["query", "value"],
            lora_dropout=0.1,
            bias="none",
            modules_to_save=["classifier"],
        )

        if args.LOKR:
            config = LoKrConfig(
                r=args.lora_max_rank,
                alpha=args.lora_alpha,
                target_modules=["query", "value"],
                modules_to_save=["classifier"],
            )

        net_glob = get_peft_model(model, config)
        net_glob.to(args.device)
    else:
        exit('Error: unrecognized model')

    global_model = copy.deepcopy(net_glob.state_dict())
    ##print(global_model.keys())
    # Set all the lora matrix to 0.
    #for k in global_model.keys():
    #    if 'lora_A' in k:
    #        global_model[k][:,:] = 0
    #    elif 'lora_B' in k:
    #        global_model[k][:,:] = 0
    #net_glob.load_state_dict(global_model)

    return args, net_glob, global_model, model_dim(global_model)

def model_dim(model):
    '''
    compute model dimension
    '''
    flat = [torch.flatten(model[k]) for k in model.keys()]
    s = 0
    for p in flat: 
        s += p.shape[0]
    return s


def model_clip(model, clip):
    '''
    clip model update
    '''
    model_norm=[]
    for k in model.keys():
        if 'num_batches_tracked' in k or 'running_mean' in k or 'running_var' in k:
            continue
        model_norm.append(torch.norm(model[k]))
        
    total_norm = torch.norm(torch.stack(model_norm))
    clip_coef = clip / (total_norm + 1e-8)
    if clip_coef < 1:
        for k in model.keys():
            if 'num_batches_tracked' in k or 'running_mean' in k or 'running_var' in k:
                continue
            model[k] = model[k] * clip_coef
    return model, total_norm

def save_checkpoint(state, is_best, filename='checkpoint.pth.tar'):
    """
    Save the training model
    """
    torch.save(state, filename)

def get_trainable_values(net,mydevice=None):
    ' return trainable parameter values as a vector (only the first parameter set)'
    trainable=filter(lambda p: p.requires_grad, net.parameters())
    paramlist=list(trainable) 
    N=0
    for params in paramlist:
        N+=params.numel()
    if mydevice:
        X=torch.empty(N,dtype=torch.float).to(mydevice)
    else:
        X=torch.empty(N,dtype=torch.float)
    X.fill_(0.0)
    offset=0
    for params in paramlist:
        numel=params.numel()
        with torch.no_grad():
            X[offset:offset+numel].copy_(params.data.view_as(X[offset:offset+numel].data))
        offset+=numel

    return X

def put_trainable_values(net,X):
    ' replace trainable parameter values by the given vector (only the first parameter set)'
    trainable=filter(lambda p: p.requires_grad, net.parameters())
    paramlist=list(trainable)
    offset=0
    for params in paramlist:
        numel=params.numel()
        with torch.no_grad():
            params.data.copy_(X[offset:offset+numel].data.view_as(params.data))
        offset+=numel
