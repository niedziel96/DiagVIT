import argparse
import os

from diaglib import config
from diaglib.data.vit.dataset import AbstractDiagSetDataset #, EvaluationDiagSetDataset # to change 
#from diaglib.learn.cnn.classify.model_classes import MODEL_CLASSES # no idea 
#from diaglib.learn.cnn.classify.trainers import Trainer # to replace 
import random


### Base Packages
#from __future__ import print_function
import argparse
import pdb
import os
import math

### Numerical Packages
import numpy as np
import pandas as pd

### Internal Imports
#from datasets.dataset_generic import Generic_WSI_Classification_Dataset, Generic_MIL_Dataset
from diaglib.learn.utils.file_utils import save_pkl, load_pkl # to add 
from diaglib.learn.utils.utils import *
from diaglib.learn.trainer_vit import train
from diaglib.learn.logger import create_logger

### PyTorch Imports
import torch
from torch.utils.data import DataLoader, sampler
import torch.nn as nn
import torch.nn.functional as F
import torch.distributed as dist
import torchvision 

##### Train-Val-Test Loop for 10-Fold CV
def main(args, dataset):
    ### Creates Results Directory (if not previously created)
    if not os.path.isdir(args.results_dir):
        os.mkdir(args.results_dir)

    ### Which folds to evaluates + iterate
    if args.k_start == -1:
        start = 0
    else:
        start = args.k_start
    if args.k_end == -1:
        end = args.k
    else:
        end = args.k_end

    ### 10-Fold CV Loop.
    all_test_auc, all_val_auc = [], []
    all_test_acc, all_val_acc= [], []
    
    if args.default_partition:
        folds = np.arange(0, 1)
    else: 
        folds = np.arange(start, end)
        
    for i in folds:
               
        if args.default_partition:
            train_dataset, test_dataset, val_dataset = dataset
            tr_dataset = train_dataset.get_split_from_df(i, split_key='default')
            t_dataset = test_dataset.get_split_from_df(i, split_key='default')
            v_dataset = val_dataset.get_split_from_df(i, split_key='default')
            datasets = (tr_dataset, t_dataset, v_dataset)
        else: 
            train_dataset = dataset.get_split_from_df(i, split_key='train')
            test_dataset = dataset.get_split_from_df(i, split_key='test')
            val_dataset = dataset.get_split_from_df(i, split_key='val')
            datasets = (train_dataset, test_dataset, val_dataset)
            
        results, test_auc, val_auc, test_acc, val_acc  = train(datasets, i, args)
        all_test_auc.append(test_auc)
        all_val_auc.append(val_auc)
        all_test_acc.append(test_acc)
        all_val_acc.append(val_acc)

        ### Writes results to PKL File
        filename = os.path.join(args.results_dir, 'split_{}_results.pkl'.format(i))
        save_pkl(filename, results)

    ### Saves results as a CSV file
    final_df = pd.DataFrame({'folds': folds, 'test_auc': all_test_auc, 'val_auc': all_val_auc, 'test_acc': all_test_acc, 'val_acc' : all_val_acc})

    if len(folds) != args.k:
        save_name = 'summary_partial_{}_{}.csv'.format(start, end)
    else:
        save_name = 'summary.csv'
    final_df.to_csv(os.path.join(args.results_dir, save_name))

parser = argparse.ArgumentParser()

# dataset related 
parser.add_argument('-tissue_tag', type=str, choices=config.TISSUE_TAGS, required=True) # Type of tissue to work with 
parser.add_argument('-default_partition', type=bool, default=True) # partition - on default it uses partitions from original paper, otherwise it creates new folds 
parser.add_argument('-image_size', type=int, default=256) # used in dataset 
parser.add_argument('-patch_size', type=int, default=224) # used in dataset
parser.add_argument('-augument', type=bool, default=True) # augument images  
parser.add_argument('-subtract_mean', type=bool, default=True) # subtract_mean for images (image net)   
parser.add_argument('-label_dictionary', default=None) # labels dictionary to use
parser.add_argument('-shuffling', type=bool, default=True) 
parser.add_argument('-class_ratios', default=None) 
parser.add_argument('-seed', type=int, default=7) 
parser.add_argument('-input_data_csv', type=str, default=None) # path to csv file with already prepared files table 
parser.add_argument('-drop_out', type=float, default=None, help='set dropout rate')
parser.add_argument('-att_dropout', type=float, default=None, help='set attention dropout rate')

# optimizer and loader related 
parser.add_argument('-batch_size', type=int, default=32) # TBC 
parser.add_argument('-weight_decay', type=float, default=0.05) # used in utils.utils
parser.add_argument('-optimizer', type=str, default='sgd', choices=['adam', 'sgd']) # used in utils.utils

# model and train related 
parser.add_argument('-pretrained', type=bool, default=False) # used in trainer_vit
parser.add_argument('-early_stopping', type=bool, default=False) # used in trainer_vit 
parser.add_argument('-max_epochs', type=int, default=300) # used in trainer_vit 
parser.add_argument('-warmup_epochs', type=int, default=20) # used in trainer_vit - lr_scheduler 
parser.add_argument('-decay_epochs', type=int, default=30) # used in trainer_vit - lr_scheduler 
parser.add_argument('-multi_steps', type=list, default=[]) # used in trainer_vit - lr_scheduler 
parser.add_argument('-scheduler_name', type=str, default='cosine') # used in trainer_vit - lr_scheduler 
parser.add_argument('-warmup_prefix', type=bool, default=True) #  used in trainer_vit - lr_scheduler 
parser.add_argument('-min_lr', type=int, default=5e-6) # used in trainer_vit - lr_scheduler 
parser.add_argument('-warmup_lr', type=int, default=5e-7) # used in trainer_vit - lr_scheduler 
parser.add_argument('-dacay_rate', type=int, default=0.1) # used in trainer_vit - lr_scheduler 
parser.add_argument('-gamma', type=int, default=0.1) # used in trainer_vit - lr_scheduler 
parser.add_argument('-label_smoothing', type=int, default=0) # used in trainer_vit - lr_scheduler NOT TESTED / NOT USED 
parser.add_argument('-mixup', type=int, default=0) # used in trainer_vit - lr_scheduler NOT TESTED / NOT USED 
parser.add_argument('-use_autocast', type=bool, default=True) #  used in trainer_vit - lr_scheduler 
parser.add_argument('-clip_grad', type=int, default=5.0) # used in trainer_vit - clip gradient
parser.add_argument('-use_tensorboard', type=bool, default=True) #  used in trainer_vit 

# distributed training
parser.add_argument("-local_rank", type=int, required=True, default = 0, help='local rank for DistributedDataParallel')

parser.add_argument('-learning_rate', type=float, default=5e-4) # used in utils.utils
parser.add_argument('-momentum', type=float, default=0.9) # used in utils.utils 
parser.add_argument('-weighted_sample', type=bool, default=True) # used in trainer_vit - to make weighted sampling in case of unbalanced datasets 
parser.add_argument('-already_initialized', type=bool, default=False) # for dataset
parser.add_argument('-n_classes', type=int, default=None) # used in trainer_vit 
parser.add_argument('-acc_steps', type=int, default=1) # used in trainer_vit - accumulation steps for gradient
parser.add_argument('-en_autocast', type=bool, default=True) # used in trainer_vit - to enable autocasting for training

#  used in train 
parser.add_argument('--log_data', action='store_true', default=True, help='log data using logging') # used in trainer 
parser.add_argument('--results_dir', type=str, default='./results', help='results directory (default: ./results)') # used in trainer 
parser.add_argument('--model_type', type=str, default='vit_base_b_16', help='Model type to use for training - vit_base_b_16 by default') # used in trainer 

parser.add_argument('--k', type=int, default=10, help='number of folds (default: 10)')
parser.add_argument('--k_start', type=int, default=-1, help='start fold (default: -1, last fold)')
parser.add_argument('--k_end', type=int, default=-1, help='end fold (default: -1, first fold)')


args = parser.parse_args()

if 'RANK' in os.environ and 'WORLD_SIZE' in os.environ:
    rank = int(os.environ["RANK"])
    world_size = int(os.environ['WORLD_SIZE'])
    print(f"RANK and WORLD_SIZE in environ: {rank}/{world_size}")
else:
    rank = -1
    world_size = -1
    
if torch.cuda.is_available():
    torch.cuda.set_device(args.local_rank)
    torch.distributed.init_process_group(backend='nccl', init_method='env://', world_size=world_size, rank=rank)
    torch.distributed.barrier()

seed = args.seed + dist.get_rank()
torch.manual_seed(seed)
torch.cuda.manual_seed(seed)
np.random.seed(seed)
random.seed(seed)
torch.backends.benchmark = True  # you can as well set it to False 
#torch.backends.cudnn.deterministic = True # for CNN you can uncomment this


if args.default_partition: 
    
    train_dataset = AbstractDiagSetDataset( 
            tissue_tag = args.tissue_tag, 
            partitions = ['train'],
            patch_size = args.patch_size,
            image_size = args.image_size,
            augment = args.augument, 
            subtract_mean = args.subtract_mean, 
            label_dictionary = args.label_dictionary, 
            shuffling = args.shuffling, 
            class_ratios = args.class_ratios,  
            seed = args.seed,
            input_data_table = None,
            input_data_csv = args.input_data_csv,
            agg_splits = [],
            already_initialized = args.already_initialized
            )
            
    test_dataset = AbstractDiagSetDataset( 
        tissue_tag = args.tissue_tag, 
        partitions = ['test'],
        patch_size = args.patch_size,
        image_size = args.image_size,
        augment = False, 
        subtract_mean = args.subtract_mean, 
        label_dictionary = args.label_dictionary, 
        shuffling = args.shuffling, 
        class_ratios = args.class_ratios,  
        seed = args.seed,
        input_data_table = None, 
        input_data_csv = args.input_data_csv,
        agg_splits = [],
        already_initialized = args.already_initialized
        )
        
    val_dataset = AbstractDiagSetDataset( 
            tissue_tag = args.tissue_tag, 
            partitions = ['validation'],
            patch_size = args.patch_size,
            image_size = args.image_size,
            augment = False, 
            subtract_mean = args.subtract_mean, 
            label_dictionary = args.label_dictionary, 
            shuffling = args.shuffling, 
            class_ratios = args.class_ratios,  
            seed = args.seed,
            input_data_table = None, 
            input_data_csv = args.input_data_csv,
            agg_splits = [],
            already_initialized = args.already_initialized
            )
    
    dataset = (train_dataset, test_dataset, val_dataset)

elif not args.default_partition and args.dataset_folder is not None:
    transform = torchvision.transforms.Compose(
        [torchvision.transforms.ToTensor(),
        torchvision.transforms.Normalize((0.5,), (0.5,)),
        torchvision.transforms.Resize((args.image_size,args.image_size)),
        ])
    
    training_set = torchvision.datasets.ImageFolder(f'{args.dataset_folder}/train', transform=transform)
    test_set = torchvision.datasets.ImageFolder(f'{args.dataset_folder}/test', transform=transform)
    dataset = (training_set, test_set)
           
else: 
    dataset = AbstractDiagSetDataset( 
            tissue_tag = args.tissue_tag, 
            partitions = None,
            patch_size = args.patch_size,
            image_size = args.image_size,
            augment = False, 
            subtract_mean = args.subtract_mean, 
            label_dictionary = args.subtract_mean, 
            shuffling = args.shuffling, 
            class_ratios = args.class_ratios,  
            seed = args.seed,
            input_data_table = None, 
            input_data_csv = args.input_data_csv,
            agg_splits = [],
            already_initialized = False
            )


if __name__ == "__main__":
    
    logger = create_logger(output_dir=args.result_dir, dist_rank=dist.get_rank(), name=f"{args.model_type}_pretrained:{args.pretrained}")
    
    # linear scale the learning rate according to total batch size, may not be optimal
    linear_scaled_lr = args.learning_rate * args.batch_size * dist.get_world_size() / 512.0
    linear_scaled_warmup_lr = args.warmup_lr * args.batch_size * dist.get_world_size() / 512.0
    linear_scaled_min_lr = args.min_lr * args.batch_size * dist.get_world_size() / 512.0
    # gradient accumulation also need to scale the learning rate
    if args.acc_steps > 1:
        linear_scaled_lr = linear_scaled_lr * args.acc_steps
        linear_scaled_warmup_lr = linear_scaled_warmup_lr * args.acc_steps
        linear_scaled_min_lr = linear_scaled_min_lr * args.acc_steps
    args.defrost()
    args.TRAIN.BASE_LR = linear_scaled_lr
    args.TRAIN.WARMUP_LR = linear_scaled_warmup_lr
    args.TRAIN.MIN_LR = linear_scaled_min_lr
    args.logger = logger
    args.freeze()
    
    results = main(args, dataset)
    print("finished!")
    print("end script")