# -*- coding: utf-8 -*-
"""
Created on Mon Feb  5 17:24:38 2024

@author: simon
"""

#DEPENDENCIES
from torch.utils.data import DataLoader
from torchvision.datasets import LFWPairs, LFWPeople
from sklearn.model_selection import train_test_split

BASE_FOLDER = "lfw_data"
        
class PairsDataset():
    
    def __init__(self, transform = False):
        
        if(transform):
            self.transform = transform
        else:
            self.transform = None
                
        train_val_pairs = LFWPairs(root = BASE_FOLDER,
                                    split = 'train',
                                    transform = self.transform,
                                    download = 'True')
        
        self.test_pairs = LFWPairs(root = BASE_FOLDER,
                                   split = 'test',
                                   transform = self.transform,
                                   download = 'True')
        
        self.people_labels = LFWPeople(root=BASE_FOLDER,
                                      split='test',
                                      transform=transform,
                                      download = 'True')
        
        self.train_pairs, self.val_pairs = train_test_split(train_val_pairs, test_size = 0.14)
        
    def get_loaders(self, batch_size):
        return {'train': DataLoader(self.train_pairs, batch_size = batch_size, shuffle = True, num_workers = 0),
                'valid': DataLoader(self.val_pairs, batch_size = batch_size, shuffle = False, num_workers = 0),
                'test': DataLoader(self.test_pairs, batch_size = batch_size, shuffle = False, num_workers = 0),
                'test_people': DataLoader(self.people_labels, batch_size = batch_size, shuffle = False, num_workers = 0)}
    
    def get_trainset(self):
        return self.train_pairs
    
    def get_valset(self):
        return self.val_pairs
    
    def get_testset(self):
        return self.test_pairs
    
    def get_people_labels(self):
        return self.people_labels