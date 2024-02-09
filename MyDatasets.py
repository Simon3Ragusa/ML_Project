# -*- coding: utf-8 -*-
"""
Created on Mon Feb  5 17:24:38 2024

@author: simon
"""

#DEPENDENCIES
import os, random
os.environ['KMP_DUPLICATE_LIB_OK']='True'
from torch.utils.data import Dataset, DataLoader
from torchvision import transforms
from torchvision.datasets import lfw
from sklearn.model_selection import train_test_split
from matplotlib import pyplot as plt
import numpy as np

BASE_FOLDER = "lfw_data"
IMAGE_FOLDER = "lfw_data\\lfw-py\\lfw_funneled"
FILES_FOLDER = "files"
FILES = {'train': 'pairsDevTrain.txt',
         'test' : 'pairsDevTest.txt'}

def _get_matches(split):
    with open(os.path.join(FILES_FOLDER, FILES[split])) as f:
        lines = f.readlines()
        n_pairs = int(lines[0])
        
        matched_pairs = [line.strip().split("\t") for line in lines[1 : 1 + n_pairs]]
        unmatched_pairs = [line.strip().split("\t") for line in lines[1 + n_pairs : 1 + 2*n_pairs]]
        
        return {'match': matched_pairs, 'mismatch': unmatched_pairs}
    
        
class SiameseDataset(Dataset):
    
    def __init__(self, split = 'train', transform = None):
        super(SiameseDataset, self).__init__()
        
        self.split = split
        self.transform = transform
        self.lfw_data = lfw._LFW(root=BASE_FOLDER,
                                  split=split,
                                  image_set='funneled',
                                  view='people',
                                  download=True,
                                  transform=transform)
        
        self.pair_names, self.data, self.targets = self._get_pairs(split)
        
        self.people_label = lfw.LFWPeople(root=BASE_FOLDER,
                                          split=split,
                                          transform=transform)

    def _get_pairs(self, split = 'train'):
        pair_names, data, targets = [], [], []
        
        matched_pairs = _get_matches(split)['match']
        unmatched_pairs = _get_matches(split)['mismatch']
        
        '''
        Scorro i match per ottenere una lista indicizzata delle coppie
        '''
        for pair in matched_pairs:
            id_anchor = pair[0]
            anchor = self.lfw_data._get_path(id_anchor, pair[1])
            positive = self.lfw_data._get_path(id_anchor, pair[2])
            
            pair_names.append((id_anchor, id_anchor))
            data.append((anchor, positive))
            targets.append(1)
            
        '''
        Scorro i mismatch per ottenere le non-coppie nella lista di prima
        '''
        for pair in unmatched_pairs:
            id_anchor = pair[0]
            anchor = self.lfw_data._get_path(id_anchor, pair[1])
            id_negative = pair[2]
            negative = self.lfw_data._get_path(id_negative, pair[3])
            
            pair_names.append((id_anchor, id_negative))
            data.append((anchor, negative))
            targets.append(0)
            
        return pair_names, data, targets
        
        
    def __getitem__(self, index):
        '''
        Parameters
        ----------
        index : int

        Returns
        -------
        (anchor, validation, target (1 = match / 0 = mismatch))

        '''
        anchor, validation = self.data[index]
        anchor, validation = self.lfw_data._loader(anchor), self.lfw_data._loader(validation)
        target = self.targets[index]
        
        if self.transform is not None:
            anchor, validation = self.transform(anchor), self.transform(validation)
        
        return anchor, validation, target
                
    
    def __len__(self):
        return len(self.data)
    
    def get_loader(self):
        if(self.split == 'train'):
            return {'pairsTrain': DataLoader(self, batch_size=32, shuffle = True, num_workers=0),
                    'train': DataLoader(self.people_label, batch_size = 32, shuffle = True, num_workers=0)}
        else:
            return {'pairsTest': DataLoader(self, batch_size = 32, num_workers=0),
                    'test': DataLoader(self.people_label, batch_size=32, num_workers=0)}