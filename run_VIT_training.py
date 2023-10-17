import argparse
import logging
import os

from diaglib import config
from diaglib.data.vit.dataset import AbstractDiagSetDataset #, EvaluationDiagSetDataset # to change 
#from diaglib.learn.cnn.classify.model_classes import MODEL_CLASSES # no idea 
#from diaglib.learn.cnn.classify.trainers import Trainer # to replace 


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

### PyTorch Imports
import torch
from torch.utils.data import DataLoader, sampler
import torch.nn as nn
import torch.nn.functional as F


##### Train-Val-Test Loop for 10-Fold CV
def main(args):
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
       
       
        seed_torch(args.seed) ### Sets the Torch.Seed
        
        if args.default_partition:
            train_dataset = train_dataset.get_split_from_df(i, split_key='default')
            test_dataset = test_dataset.get_split_from_df(i, split_key='default')
            val_dataset = val_dataset.get_split_from_df(i, split_key='default')
            datasets = (train_dataset, test_dataset, val_dataset)
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


logging.basicConfig(level=logging.INFO)

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

# optimizer and loader related 
parser.add_argument('-batch_size', type=int, default=32) # TBC 
parser.add_argument('-weight_decay', type=float, default=0.0005) # used in utils.utils
parser.add_argument('-optimizer', type=str, default='sgd', choices=['adam', 'sgd']) # used in utils.utils

# model and train related 
parser.add_argument('-pretrained', type=bool, default=False) # used in trainer_vit
parser.add_argument('-early_stopping', type=bool, default=False) # used in trainer_vit 
parser.add_argument('-max_epochs', type=int, default=50) # used in trainer_vit 
parser.add_argument('-learning_rate', type=float, default=0.0001) # used in utils.utils
parser.add_argument('-momentum', type=float, default=0.9) # used in utils.utils 
parser.add_argument('-weighted_sample', type=bool, default=True) # used in trainer_vit - to make weighted sampling in case of unbalanced datasets 

#  used in train 
parser.add_argument('--log_data', action='store_true', default=True, help='log data using tensorboard') # used in trainer 
parser.add_argument('--results_dir',    type=str, default='./results', help='results directory (default: ./results)') # used in trainer 
parser.add_argument('--model_type',    type=str, default='vit_base_b_16', help='Model type to use for training - vit_base_b_16 by default') # used in trainer 

parser.add_argument('--k',              type=int, default=10, help='number of folds (default: 10)')
parser.add_argument('--k_start',        type=int, default=-1, help='start fold (default: -1, last fold)')
parser.add_argument('--k_end',          type=int, default=-1, help='end fold (default: -1, first fold)')


args = parser.parse_args()


device=torch.device("cuda" if torch.cuda.is_available() else "cpu")


##### Setting the seed + log settings
def seed_torch(seed=7):
    import random
    random.seed(seed)
    os.environ['PYTHONHASHSEED'] = str(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if device.type == 'cuda':
        torch.cuda.manual_seed(seed)
        torch.cuda.manual_seed_all(seed) # if you are using multi-GPU.
    torch.backends.cudnn.benchmark = False
    torch.backends.cudnn.deterministic = True

seed_torch(args.seed)

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
            already_initialized = False
            )
            
    test_dataset = AbstractDiagSetDataset( 
        tissue_tag = args.tissue_tag, 
        partitions = ['test'],
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
        
    val_dataset = AbstractDiagSetDataset( 
            tissue_tag = args.tissue_tag, 
            partitions = ['val'],
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
    results = main(args)
    print("finished!")
    print("end script")