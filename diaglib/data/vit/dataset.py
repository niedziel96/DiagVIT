import json
import logging
import numpy as np
import pandas as pd
from pathlib import Path
import torch 
 
from abc import ABC, abstractmethod
from diaglib import config
#from diaglib.data.diagset.loading import ndp, db, local
#from diaglib.data.diagset.loading.common import prepare_multipolygons
#from diaglib.data.diagset.paths import get_nested_path
from diaglib.config import IMAGENET_IMAGE_MEAN
#from diaglib.predict.maps.common import segment_foreground_vdsr
#from queue import Queue
#from shapely.geometry.polygon import Polygon
#from threading import Thread
from sklearn.model_selection import StratifiedShuffleSplit

class AbstractDiagSetDataset(ABC):
    def __init__(self, 
        tissue_tag, # either J or P or S
        partitions, # selected partition
        patch_size=(224, 224), # default patch size 
        image_size=(256, 256), # default image size 
        augment=True, # default true for flip image 
        subtract_mean=True, # default true for weights pretrained on imagenet 
        label_dictionary=None, # for preprepared label dict 
        shuffling=True, # default shuffling 
        class_ratios = None, # none for class ratio  
        seed=7,
        input_data_table = None, # default none 
        input_data_csv = None,
        agg_splits = [],
        already_initialized = False,
        ):
    
        # Make sure the tissue tag is legit 
        assert tissue_tag in config.TISSUE_TAGS, "Tissue tag not in config, unknown entry!"

        # Assert for partition type 
        for partition in partitions:
            assert partition in ['train', 'validation', 'test', 'unannotated'], "Invalid partition!"

        # Initialize 
        self.tissue_tag = tissue_tag
        self.partitions = partitions
        self.patch_size = patch_size
        self.image_size = image_size
        self.augment = augment
        self.subtract_mean = subtract_mean
        self.label_dictionary = label_dictionary
        self.shuffling = shuffling
        self.class_ratios = class_ratios
        self.seed = seed 
        self.input_data_table = input_data_table
        self.input_data_csv = input_data_csv
        self.agg_splits = agg_splits
        self.already_initialized = already_initialized
            
        

        # Perform initial setup and data preparation only of ot was not initialized before (genereic dataset) 
        if not self.already_initialized:
            
            # assign data from config 
            self.root_blobs_path = Path(config.DIAGSET_BLOBS_PATH[tissue_tag])
            self.root_distributions_path = Path(config.DIAGSET_DISTRIBUTIONS_PATH[tissue_tag])
            assert self.root_blobs_path.exists(), "Specified path for blobs does not exists!"
            
            if self.label_dictionary is None:
                logging.getLogger('diaglib').info('Using default label dictionary...')
                print('self label empty -- getting it from config')
                self.label_dictionary = config.LABEL_DICTIONARIES[tissue_tag]
            else:
                print(type(label_dictionary))
                self.label_dictionary = label_dictionary
            
            # get numeric labels based on dict values 
            self.numeric_labels = list(set(self.label_dictionary.values()))
            
            # if there is already pre-prepared input data to be read - read it
            if self.input_data_csv is not None:  
                self.input_data_table = pd.read_csv(self.input_data_csv)
        
            # set empty variables for blob paths and class dist (for ratios) 
            self.blob_paths = {key: [] for key in self.numeric_labels}
            self.class_distribution = {key: 0 for key in self.numeric_labels}
            self.length = 0
            

            # get scan names for current dir 
            self.scan_names = [path.name for path in self.root_blobs_path.iterdir()]

            # for any partition type - get scan names and store in list 
            partition_scan_names = []
            for partition in self.partitions:
                partition_path = Path(config.DIAGSET_PARTITIONS_PATH[tissue_tag])

                if partition_path.exists():
                    df = pd.read_csv(f'{partition_path}/{partition}.csv')
                    partition_scan_names += df['scan_id'].astype(np.str).tolist()
                else:
                    raise ValueError('Partition file not found under "%s".' % partition_path)

            self.scan_names = [scan_name for scan_name in self.scan_names if scan_name in partition_scan_names]
            logging.getLogger('diaglib').info('Loading blob paths...')

            # for each scan name assign path and add to blobs 
            errored_dist = []
            for scan_name in self.scan_names:
                ignored = []
                for string_label in config.USABLE_LABELS[tissue_tag]:
                    numeric_label = self.label_dictionary[string_label]
                    
                    pth = Path((f'{self.root_blobs_path}/{scan_name}/{string_label}'))
                    
                    # check if path is existing 
                    if pth.exists():
                        blbs_list = [x for x in pth.iterdir()]
                        blob_names = map(lambda x: x.name, sorted(blbs_list))
                    else:
                        print(f'Not found for: {string_label}')
                        
                        ignored.append(string_label)
                        continue 

                    for blob_name in blob_names:
                        self.blob_paths[numeric_label].append(f'{self.root_blobs_path}/{scan_name}/{string_label}/{blob_name}')
        
        if self.already_initialized:
            print(' ----- reading already prepared input data table -----')
            self.input_data_table = pd.read_csv(f'inputs_data_table_{self.partitions[0]}.csv')
                    
                
        if self.input_data_table is None: 
            # create table to select from based on blob paths 
            self.input_data_table = self.conv_to_table(self.blob_paths) # creates a table |patch_path|label|
            print(self.input_data_table)
            # add sub-index to each file 
            self.input_data_table = self.add_sub_index(self.input_data_table)

        
            if self.shuffling:
                # for numeric_label in self.numeric_labels:
                    # np.random.shuffle(self.blob_paths[numeric_label])
                    self.input_data_table = self.input_data_table.sample(frac = 1, random_state = self.seed)

    def get_num_classes(self):
        labels_num = len(self.input_data_table.groupby(by=['label']).size())
        return labels_num
    
    def get_class_ratios(self):
        lenght = len(self.input_data_table)
        labels_size = self.input_data_table.groupby(by=['label']).size()
        labels_keys = labels_size.keys()

        class_ratios = {int(labels_keys[k]): v/lenght for k,v in enumerate(labels_size)}
        return class_ratios
        
    def conv_to_table(self, blob_dict):
        df = pd.DataFrame()
        for key, value in blob_dict.items():
            df = pd.concat([df, pd.DataFrame({'patch_path' : self.blob_paths[key], 'label': [int(key) for x in range(len(self.blob_paths[key]))]})], axis=0)
    
        df.reset_index(inplace = True)
        return df 
        
    def get_split_from_df(self, split_no, split_key='train'):
        
        if split_key == 'default':
            return SplitDataSet(self.input_data_table, input_data_csv=None, subtract_mean=True, seed=self.seed, augment=True)
    
        all_splits = self.agg_splits[split_no]
        split = all_splits[split_key]
        split = split.dropna().reset_index(drop=True)

        if len(split) > 0:
            mask = self.input_data_table['artificial_id'].isin(split.tolist())
            df_slice = self.input_data_table[mask].reset_index(drop=True)
            print("Traing Data Size ({%0.2f}): %d" % (self.prop, df_slice.shape[0]))
            
            if split_key == 'train':
                split = SplitDataSet(df_slice, input_data_csv=None, subtract_mean=True, seed=self.seed, augment=True)
            else: 
                split = SplitDataSet(df_slice, input_data_csv=None, subtract_mean=True, seed=self.seed, augment=False)
        else:
            split = None
        
        return split
        
    def make_strat_shuffle_split(self, n_splits, t_and_v_size):
        
        train_split = []
        tv_split = []
        test_split = []
        val_split = []
        self.agg_splits = []
        if self.split_type == 'shuffle':
            split = StratifiedShuffleSplit(n_splits=n_splits, test_size=t_and_v_size, random_state=self.seed)
            for train_index, test_valid_index in split.split(self.input_data_table , self.input_data_table.label):
                train_set = self.input_data_table.iloc[train_index]
                train_split.append(train_set)
                test_valid_set =  self.input_data_table.iloc[test_valid_index]
                tv_split.append(test_valid_set)
                
                sub_split = StratifiedShuffleSplit(n_splits=n_splits, test_size=0.5, random_state=self.seed) # split the same way test and validation in 50:50 ratio 
                for test_index, valid_index in sub_split.split(test_valid_set, test_valid_set.label):
                    test_set = test_valid_set.iloc[test_index]
                    test_split.append(test_set)
                    valid_set = test_valid_set.iloc[valid_index]
                    val_split.append(valid_set)
                    
            for i in range(n_splits):
                train, test, val = train_split[i].artificial_id, test_split[i].artificial_id, val_split[i].artificial_id
        
                # reset index for current 
                train_ids = train.reset_index()
                val_ids = val.reset_index()
                test_ids = test.reset_index()

                df = pd.concat([train_ids.artificial_id, val_ids.artificial_id, test_ids.artificial_id], ignore_index = True,axis=1)
                print(f'Preparing for {i}')
                df.columns = ['train','test','val']
                fname = f'splits_{i}.csv'
                print(fname)
                df.to_csv(f'{fname}')
                self.agg_splits.append(df)
            
        elif self.split_type == 'kfold':
            print('not implemented yet....')
    
    def add_sub_index(self, dt_table):
        
        n_dt_table = pd.DataFrame()
        # for each single image get len and assign idx 
        for idx in range(len(dt_table['patch_path'])):
            if idx != len(dt_table['patch_path']):
                tmp_img = np.load(dt_table.patch_path[idx])
                n_sub_idx = [x for x in range(len(tmp_img))]
                tmp_tbl = pd.DataFrame(np.repeat(dt_table.iloc[idx:idx+1].values, len(tmp_img), axis=0))
                tmp_tbl.rename(columns = {'1':'patch_path', '2':'label'}, inplace = True)
                tmp_tbl['subindex'] = n_sub_idx
                tmp_tbl['artificial_id'] = dt_table['patch_path'][idx] + '_' + tmp_tbl['subindex'].astype(str) # that one is unique for any patch - it's composed of file path (scan) and subindex. 
                print(f'processing {idx} out of {len(dt_table["patch_path"])}')
                n_dt_table = pd.concat([n_dt_table, tmp_tbl], axis=0)
        
        print(n_dt_table)
        f = open(f'inputs_data_table_{self.partitions[0]}.csv', 'wb')
        n_dt_table.to_csv(f, encoding='utf-8', index=False)
        f.close()
        
        return n_dt_table
    
    def __getitem__(self, idx):
        label = self.input_data_table['label'][idx]
        
        imgs = np.load(self.input_data_table['patch_path'][idx])
        image = imgs[self.input_data_table['subindex'][idx]].astype(np.float32)
        print(image.shape)
        if self.augment:
            image = self._augment(image)
            print(image.shape)
        else:
            x = (image.shape[0] - self.patch_size[0]) // 2
            y = (image.shape[1] - self.patch_size[1]) // 2

            image = image[x:(x + self.patch_size[0]), y:(y + self.patch_size[1])]

        if self.subtract_mean:
            image -= IMAGENET_IMAGE_MEAN
        
        feature = torch.from_numpy(image)
        
        feature = feature.unsqueeze(0)
        feature = feature.permute(0, 3, 1, 2)
        
        print(f'-- {feature.shape} --')
        return feature, label 
           

    def prepare_images(self, blob_path):
        images = np.load(blob_path)

        if self.shuffling:
            np.random.shuffle(images)

        prepared_images = []

        for i in range(len(images)):
            image = images[i].astype(np.float32)

            if self.augment:
                image = self._augment(image)
            else:
                x = (image.shape[0] - self.patch_size[0]) // 2
                y = (image.shape[1] - self.patch_size[1]) // 2

                image = image[x:(x + self.patch_size[0]), y:(y + self.patch_size[1])]

            if self.subtract_mean:
                image -= IMAGENET_IMAGE_MEAN

            prepared_images.append(image)

        prepared_images = np.array(prepared_images)

        return prepared_images

    def _augment(self, image):
        x_max = image.shape[0] - self.patch_size[0]
        y_max = image.shape[1] - self.patch_size[1]

        x = np.random.randint(x_max)
        y = np.random.randint(y_max)

        image = image[x:(x + self.patch_size[0]), y:(y + self.patch_size[1])]

        if np.random.choice([True, False]):
            image = np.fliplr(image).copy() # walkaround for negative tensor problems 

        image = np.rot90(image, k=np.random.randint(4)).copy() # walkaround for negative tensor problems

        return image


class SplitDataSet(AbstractDiagSetDataset):
    def __init__(self,
        input_data_table, 
        input_data_csv = None,
        subtract_mean = True, 
        seed = 7,
        augment = True,
        already_initialized = True,
        patch_size = (224, 224)
        ):
    
        # Make sure the tissue tag is legit 

        # Initialize 
        self.augment = augment
        self.subtract_mean = subtract_mean
        self.seed = seed 
        self.input_data_table = input_data_table
        self.input_data_csv = input_data_csv
        self.already_initialized = already_initialized
        self.patch_size = patch_size
        
    def __len__(self):
        return len(self.input_data_table)


## ALL BELOW IS NOT MINE, NOT SURE IF I EVEN NEED IT?? 

