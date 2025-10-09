import numpy as np
import os
import dill
import torch
from torch.utils.data import Dataset
from transformers import AutoTokenizer, default_data_collator
from tqdm import tqdm
from datasets import load_dataset
from torchvision.transforms import (
    Compose,
    Normalize,
    RandomHorizontalFlip,
    RandomResizedCrop,
    Resize,
    ToTensor,
)
from transformers import AutoImageProcessor
from transformers import AutoTokenizer
#from .noniid_patitioner import ClassWisePartitioner



# split for federated settings
class DatasetSplit(Dataset):
    def __init__(self, dataset, idxs, args):
        self.dataset = dataset
        self.idxs = list(idxs)
        self.args = args

    def __len__(self):
        return len(self.idxs)

    def __getitem__(self, item):
        if self.args.dataset == 'cifar100':
            data_item = self.dataset[int(self.idxs[item])]
            image = data_item['img']
            label = data_item['fine_label']
            pixel_values = data_item['pixel_values']
            return image, label, pixel_values
        else:
            image, label = self.dataset[self.idxs[item]]
            return image, label

def load_partition(args):
    dict_users = []
    # read dataset
    if args.dataset == 'cifar100':
        path = './data/dataset/cifar100'
        dataset = load_dataset("cifar100", keep_in_memory=True)
        args.labels = dataset['train'].features["fine_label"].names
        args.label2id, args.id2label = dict(), dict()
        for i, label in enumerate(args.labels):
            args.label2id[label] = i
            args.id2label[i] = label

        image_processor = AutoImageProcessor.from_pretrained('google/vit-base-patch16-224-in21k')
        normalize = Normalize(mean=image_processor.image_mean, std=image_processor.image_std)
        if args.model == 'google/vit-base-patch16-224-in21k':
            train_transforms = Compose(
                [
                    RandomResizedCrop(image_processor.size["height"]),
                    RandomHorizontalFlip(),
                    # Grayscale(),
                    ToTensor(),
                    normalize,
                ]
            )

            val_transforms = Compose(
                [
                    Resize(image_processor.size["height"]),
                    ToTensor(),
                    normalize,
                ]
            )


            proxy_transforms = Compose(
                [
                    Resize((image_processor.size["height"], image_processor.size["width"])),
                    ToTensor(),
                    normalize,
                ]
            )
        else:
            train_transforms = Compose(
                [
                    # RandomResizedCrop(image_processor.size["height"]),
                    RandomResizedCrop((32, 32)),
                    RandomHorizontalFlip(),
                    # Grayscale(),
                    ToTensor(),
                    normalize,
                ]
            )

            val_transforms = Compose(
                [
                    # Resize(image_processor.size["height"]),
                    Resize((32, 32)),
                    ToTensor(),
                    normalize,
                ]
            )

        def preprocess_train(example_batch):
            """Apply train_transforms across a batch."""
            example_batch["pixel_values"] = [train_transforms(image.convert("RGB")) for image in example_batch["img"]]
            return example_batch

        def preprocess_val(example_batch):
            """Apply val_transforms across a batch."""
            example_batch["pixel_values"] = [val_transforms(image.convert("RGB")) for image in example_batch["img"]]
            return example_batch
        
        def preprocess_proxy(example_batch):
            """Apply val_transforms across a batch."""
            example_batch["pixel_values"] = [proxy_transforms(image.convert("RGB")) for image in example_batch["img"]]
            return example_batch
        
        if 'depthffm_fim' in args.model_heterogeneity:
            if args.model_heterogeneity == 'depthffm_fim' or args.model_heterogeneity == 'depthffm_fim_extradata' or args.model_heterogeneity == 'depthffm_fim_extradata-ft':
                # FIM dataset
                dataset_fim = dataset['train'] 
                selected_indices_fim = []
                #############
                ## uniform ##
                #############
                '''
                # Get the unique fine labels and their counts
                labels_count_fim = defaultdict(int)
                for label in dataset_fim['fine_label']:
                    labels_count_fim[label] += 1
                # Determine the number of samples to select from each category (0.2% of the count)
                num_samples_per_category_fim = {label: int(0.002 * count) for label, count in labels_count_fim.items()}
                # Randomly select samples from each category
                for label, count in num_samples_per_category_fim.items():
                    indices_fim = np.where(np.array(dataset_fim['fine_label']) == label)[0]
                    selected_indices_fim.extend(np.random.choice(indices_fim, count, replace=False))
                '''
                #############
                ## random ###
                #############
                selected_indices_fim.extend(np.random.choice(list(range(len(dataset_fim))), 100, replace=False))
                args.logger.info('selected_indices_fim:')
                args.logger.info(selected_indices_fim)
                # Create the subset dataset
                dataset_fim = dataset_fim.select(selected_indices_fim)
                # remove overlap
                dataset['train'] = dataset['train'].select(list(set(range(len(dataset['train']))) - set(selected_indices_fim)))
                
                dataset_fim = dataset_fim.with_transform(preprocess_val)

            # elif args.model_heterogeneity == 'depthffm_fim_external':
            #     #############
            #     ## proxy ####
            #     #############
            #     dataset_fim = load_dataset('TNILab/cifar100_proxydata')
            #     dataset_fim = dataset_fim.rename_column('label', 'fine_label')
            #     dataset_fim = dataset_fim.rename_column('image', 'img')
            #     dataset_fim = dataset_fim['train']

            #     dataset_fim = dataset_fim.with_transform(preprocess_proxy)

        dataset_train = dataset["train"].with_transform(preprocess_train)
        dataset_test = dataset["test"].with_transform(preprocess_val)
        
        args.num_classes = 100

        pik_path = os.path.join(path,'cifar100_dict_users.pik')
        if os.path.isfile(pik_path):
            with open(pik_path, 'rb') as f: 
                dict_users = dill.load(f) 
        if len(dict_users) < 1:
            if args.iid:
                dict_users = iid(dataset_train, args.num_users)
                if args.freeze_datasplit:
                    with open(pik_path, 'wb') as f: 
                        dill.dump(dict_users, f)
            else:
                # if args.noniid_type == 'pathological':
                #     least_samples = 10 if args.pat_num_cls > 3 else 16
                #     split = ClassWisePartitioner(rng=np.random.RandomState(2),
                #                                  n_class_per_share=args.pat_num_cls,
                #                                  min_n_sample_per_share=least_samples,
                #                                  partition_mode=args.partition_mode,
                #                                  verbose=True)
                #     _tr_labels = dataset['train']['fine_label']  # labels in the original order
                #     args._idx_by_user, _user_ids_by_cls = split(_tr_labels, args.num_users,
                #                                         return_user_ids_by_class=True)
                #     print(f" train split size: {[len(idxs) for idxs in args._idx_by_user]}")
                #     args._tr_labels = np.array(_tr_labels)
                #     print(f"    | train classes: "
                #         f"{[f'{np.unique(args._tr_labels[idxs]).tolist()}' for idxs in args._idx_by_user]}")
                #     dict_users = {index: set(inner_list) for index, inner_list in enumerate(args._idx_by_user)}
                if args.noniid_type == 'dirichlet':
                    least_samples = 50
                    N = len(dataset['train'])
                    K = args.num_classes
                    dataidx_map = {}
                    min_size = 0
                    assigned_ids = []
                    idx_batch = [[] for _ in range(args.num_users)]

                    if args.partition_mode == 'dir':
                        while min_size < least_samples:
                            for k in range(K):
                                idx_k = np.where(np.array(dataset['train']['fine_label']) == k)[0]
                                np.random.shuffle(idx_k)
                                proportions = np.random.dirichlet(np.repeat(args.dir_cls_alpha, args.num_users))
                                proportions = np.array([p*(len(idx_j)<N/args.num_users) for p,idx_j in zip(proportions,idx_batch)])
                                proportions = proportions/proportions.sum()
                                proportions = (np.cumsum(proportions)*len(idx_k)).astype(int)[:-1]
                                idx_batch = [idx_j + idx.tolist() for idx_j,idx in zip(idx_batch,np.split(idx_k,proportions))]
                                min_size = min([len(idx_j) for idx_j in idx_batch])
                    elif args.partition_mode == 'uni':
                        for i in tqdm(range(args.num_users), total=args.num_users):
                            weights = torch.zeros(N)
                            proportions = np.random.dirichlet(np.repeat(args.dir_cls_alpha, K))
                            for k in range(K):
                                idx_k = np.where(np.array(dataset['train']['fine_label']) == k)[0]
                                weights[idx_k]=proportions[k]
                            weights[assigned_ids] = 0.0
                            idx_batch[i] = (torch.multinomial(weights, int(N/args.num_users), replacement=False)).tolist()
                            assigned_ids+=idx_batch[i]

                    for j in range(args.num_users):
                        np.random.shuffle(idx_batch[j])
                        dataidx_map[j] = idx_batch[j]
                    
                    args._idx_by_user = [dataidx_map[key] for key in dataidx_map]
                    
                    print(f" train split size: {[len(idxs) for idxs in args._idx_by_user]}")
                    args._tr_labels = np.array(dataset['train']['fine_label'])
                    print(f"    | train classes: "
                        f"{[f'{np.unique(args._tr_labels[idxs]).tolist()}' for idxs in args._idx_by_user]}")
                    dict_users = {index: set(inner_list) for index, inner_list in enumerate(args._idx_by_user)}

                # dict_users = noniid_arrow(dataset_train, num_users=args.num_users, class_num=args.noniid_num_cls, iid_data_amplify=args.noniid_data_amplify)
                if args.freeze_datasplit:
                    with open(pik_path, 'wb') as f: 
                        dill.dump(dict_users, f)

    elif args.dataset == 'ledgar':
        path = './data/dataset/ledgar'
        args.num_classes = 100

        # Load the dataset
        raw_train_datasets = load_dataset('lex_glue', 'ledgar', split="train")
        raw_test_datasets = load_dataset('lex_glue', 'ledgar', split="test")

        if 'depthffm_fim' in args.model_heterogeneity:
            if args.model_heterogeneity == 'depthffm_fim' or args.model_heterogeneity == 'depthffm_fim_extradata' or args.model_heterogeneity == 'depthffm_fim_extradata-ft':
                # FIM dataset
                dataset_fim = raw_train_datasets # full test dataset, we then use 10% of them which distributed uniformly on each category
                selected_indices_fim = []
                #############
                ## random ###
                #############
                selected_indices_fim.extend(np.random.choice(list(range(len(dataset_fim))), 50, replace=False))
                args.logger.info('selected_indices_fim:')
                print(selected_indices_fim)
                # Create the subset dataset
                raw_fim_datasets = dataset_fim.select(selected_indices_fim)
                # remove overlap
                raw_train_datasets = raw_train_datasets.select(list(set(range(len(raw_train_datasets))) - set(selected_indices_fim)))
            elif args.model_heterogeneity == 'depthffm_fim_external':
                #############
                ## proxy ####
                #############
                raw_fim_datasets = load_dataset("TNILab/ledgar_proxydata")['train']

        tokenizer = AutoTokenizer.from_pretrained("bert-base-uncased",
                                                   do_lower_case=True,
                                                   use_fast=True)

        def tokenize_function(examples):
            # Tokenize the texts
            batch = tokenizer(
                examples["text"],
                padding="max_length",
                max_length=512,
                truncation=True,
            )
            batch["labels"] = [list(range(args.num_classes)).index(label) for label in examples["label"]]

            return batch

        tokenized_train_datasets = raw_train_datasets.map(tokenize_function, batched=True)
        tokenized_test_datasets = raw_test_datasets.map(tokenize_function, batched=True)
        if 'depthffm_fim' in args.model_heterogeneity:
            tokenized_fim_datasets = raw_fim_datasets.map(tokenize_function, batched=True)

        args.data_collator = default_data_collator
        tokenized_train_datasets.set_format('torch')
        tokenized_test_datasets.set_format('torch')
        if 'depthffm_fim' in args.model_heterogeneity:
            tokenized_fim_datasets.set_format('torch')
        
        dataset_train = tokenized_train_datasets
        dataset_test = tokenized_test_datasets
        if 'depthffm_fim' in args.model_heterogeneity:
            dataset_fim = tokenized_fim_datasets
        
        pik_path = os.path.join(path,'ledgar_dict_users.pik')
        if os.path.isfile(pik_path):
            with open(pik_path, 'rb') as f: 
                dict_users = dill.load(f) 
        if len(dict_users) < 1:
            if args.iid:
                dict_users = iid(dataset_train, args.num_users)
                if args.freeze_datasplit:
                    with open(pik_path, 'wb') as f: 
                        dill.dump(dict_users, f)
            else:
                # if args.noniid_type == 'pathological':
                #     least_samples = 0 if args.pat_num_cls > 3 else 0
                #     split = ClassWisePartitioner(rng=np.random.RandomState(2),
                #                                 n_class_per_share=args.pat_num_cls,
                #                                 min_n_sample_per_share=least_samples,
                #                                 partition_mode=args.partition_mode,
                #                                 verbose=True)
                #     _tr_labels = dataset_train['labels']  # labels in the original order
                #     args._idx_by_user, _user_ids_by_cls = split(_tr_labels, args.num_users,
                #                                         return_user_ids_by_class=True)
                #     print(f" train split size: {[len(idxs) for idxs in args._idx_by_user]}")
                #     args._tr_labels = np.array(_tr_labels)
                #     print(f"    | train classes: "
                #         f"{[f'{np.unique(args._tr_labels[idxs]).tolist()}' for idxs in args._idx_by_user]}")
                #     dict_users = {index: set(inner_list) for index, inner_list in enumerate(args._idx_by_user)}
                
                if args.noniid_type == 'dirichlet':
                    least_samples = 50
                    N = len(dataset_train)
                    K = args.num_classes
                    dataidx_map = {}
                    min_size = 0
                    assigned_ids = []
                    idx_batch = [[] for _ in range(args.num_users)]

                    if args.partition_mode == 'dir':
                        while min_size < least_samples:
                            for k in range(K):
                                idx_k = np.where(np.array(dataset_train['labels']) == k)[0]
                                np.random.shuffle(idx_k)
                                proportions = np.random.dirichlet(np.repeat(args.dir_cls_alpha, args.num_users))
                                proportions = np.array([p*(len(idx_j)<N/args.num_users) for p,idx_j in zip(proportions,idx_batch)])
                                proportions = proportions/proportions.sum()
                                proportions = (np.cumsum(proportions)*len(idx_k)).astype(int)[:-1]
                                idx_batch = [idx_j + idx.tolist() for idx_j,idx in zip(idx_batch,np.split(idx_k,proportions))]
                                min_size = min([len(idx_j) for idx_j in idx_batch])
                    elif args.partition_mode == 'uni':
                        for i in tqdm(range(args.num_users), total=args.num_users):
                            weights = torch.zeros(N)
                            proportions = np.random.dirichlet(np.repeat(args.dir_cls_alpha, K))
                            for k in range(K):
                                idx_k = np.where(np.array(dataset_train['labels']) == k)[0]
                                weights[idx_k]=proportions[k]
                            weights[assigned_ids] = 0.0
                            idx_batch[i] = (torch.multinomial(weights, int(N/args.num_users), replacement=False)).tolist()
                            assigned_ids+=idx_batch[i]

                    for j in range(args.num_users):
                        np.random.shuffle(idx_batch[j])
                        dataidx_map[j] = idx_batch[j]
                    
                    args._idx_by_user = [dataidx_map[key] for key in dataidx_map]
                    
                    print(f" train split size: {[len(idxs) for idxs in args._idx_by_user]}")
                    args._tr_labels = np.array(dataset_train['labels'])
                    print(f"    | train classes: "
                        f"{[f'{np.unique(args._tr_labels[idxs]).tolist()}' for idxs in args._idx_by_user]}")
                    dict_users = {index: set(inner_list) for index, inner_list in enumerate(args._idx_by_user)}

                elif args.noniid_type == 'quantity_skew':
                    args._tr_labels = np.array(dataset_train['labels'])
                    N = len(dataset_train)
                    least_samples = 50
                    min_size = 0
                    try_limit = 50
                    try_cnt = 0
                    while min_size < least_samples:
                        partition = (np.random.dirichlet(args.num_users * [args.dir_par_beta]) * N).astype(int)
                        min_size = np.min(partition)
                        try_cnt += 1
                        if try_cnt > try_limit and least_samples > 10:
                            least_samples -= 10
                        assert least_samples >= 0, 'least_samples should be larger than 0'

                    remaining_ids = list(range(N))
                    args._idx_by_user = []
                    # Create IDs based on the partition
                    for i, size in enumerate(partition):
                        # Sample IDs with replacement
                        sampled_ids = np.random.choice(remaining_ids, size=size, replace=False)
                        remaining_ids = list(set(remaining_ids) - set(sampled_ids))
                        args._idx_by_user.append(sampled_ids)
                    
                    print(f" train split size: {[len(idxs) for idxs in args._idx_by_user]}")
                    print(f"    | train classes: "
                        f"{[f'{np.unique(args._tr_labels[idxs]).tolist()}' for idxs in args._idx_by_user]}")
                    dict_users = {index: set(inner_list) for index, inner_list in enumerate(args._idx_by_user)}

                if args.freeze_datasplit:
                    with open(pik_path, 'wb') as f: 
                        dill.dump(dict_users, f)
    
    else:
        exit('Error: unrecognized dataset')
    
    # return args, dataset_train, dataset_test, dataset_val, dataset_public, dict_users
    if 'depthffm_fim' in args.model_heterogeneity:
        return args, dataset_train, dataset_test, None, None, dict_users, dataset_fim
    else:
        return args, dataset_train, dataset_test, None, None, dict_users, None

###################### utils #################################################
## IID assign data samples for num_users (mnist, svhn, fmnist, emnist, cifar)
def iid(dataset, num_users):
    """
    Sample I.I.D. client data from dataset
    :param dataset:
    :param num_users:
    :return: dict of image index
    """
    print("Assigning training data samples (iid)")
    num_items = int(len(dataset)/num_users)
    dict_users, all_idxs = {}, [i for i in range(len(dataset))]
    for i in range(num_users):
        dict_users[i] = set(np.random.choice(all_idxs, num_items, replace=False))
        all_idxs = list(set(all_idxs) - dict_users[i])
    return dict_users