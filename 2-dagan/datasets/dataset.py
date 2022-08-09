from __future__ import print_function, division
import os
import pandas as pd

import torch
from torch.utils.data import Dataset


class PatchDatasetFactory:

    def __init__(self,
        data_dir, 
        split_dir, 
        seed = 7, 
        print_info = True
        ):
        r"""
        PatchDatasetFactory 

        Args:
            data_dir (string): Path to patch dataset
            split_dir (string): Path to split info csv 
            seed (int): random seed for shuffling the data
            print_info (boolean): Whether to print a summary of the dataset
        """

        #---> self
        self.data_dir = data_dir
        self.split_dir = split_dir
        self.seed = seed
        self.print_info = print_info
        self.train_ids, self.val_ids, self.test_ids  = self._train_val_test_split()

        #---> summarize
        self._summarize()

    def _train_val_test_split(self):
        # @TODO: read csv with split info, ie which samples belong to train/test/val. 
        return None, None, None

    def _summarize(self):
        if self.print_info:
            print('Init patch dataset factory...')
            print("number of cases {}".format(0))

    def return_splits(self, args, csv_path: str=None):
        assert csv_path 
        all_splits = pd.read_csv(csv_path)
        train_split = self._get_split_from_df(args, all_splits=all_splits, split_key='train')
        val_split = self._get_split_from_df(args, all_splits=all_splits, split_key='val')
        test_split = self._get_split_from_df(args, all_splits=all_splits, split_key='test')

        return train_split, val_split, test_split

    def _get_split_from_df(self, args, all_splits: dict={}, split_key: str='train', scaler=None):
        split = all_splits[split_key]
        split = split.dropna().reset_index(drop=True)

        if len(split) > 0:
            #---> create patch dataset
            split_dataset = PatchDataset(
                data_dir=self.data_dir,
                num_classes=self.num_classes,
            )
        else:
            split_dataset = None
        
        return split_dataset
    

class PatchDataset(Dataset):

    def __init__(self,
        data_dir, 
        num_classes=8
        ): 

        super(PatchDataset, self).__init__()

        #---> self
        self.data_dir = data_dir
        self.num_classes = num_classes

    def __getitem__(self, idx):
        original, augmentation  = self._load_embs_from_path(self.data_dir, self.slide_ids[idx])
        return original, augmentation 

    def _load_embs_from_path(self, slide_id):
        """
        Load a pair of patch embeddings. 
        """
        path = os.path.join(self.data_dir, 'pt_files', '{}.pt'.format(slide_id.rstrip('.svs')))
        images = torch.load(path)
        original = images[0]
        augmentation = images[1]  # @TODO: randomly fetch an augmentation. 
        return original, augmentation 
    
    def __len__(self):
        return len(self.slide_ids)  
