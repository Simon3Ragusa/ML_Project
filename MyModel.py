# -*- coding: utf-8 -*-
"""
Created on Thu Feb  1 20:52:15 2024

@author: simon
"""

from os.path import join
from torch import nn
import torch
from torch.optim import SGD
from torch.utils.tensorboard import SummaryWriter
from sklearn.metrics import accuracy_score

class EmbeddingNet(nn.Module):
    
    def __init__(self):
        super(EmbeddingNet, self).__init__()
        
        self.feature_extractor = nn.Sequential(
            nn.Conv2d(3, 16, 5, padding = 2),
            nn.MaxPool2d(2),
            nn.ReLU(),
            
            nn.BatchNorm2d(16),
            nn.Conv2d(16, 32, 5, padding = 2),
            nn.MaxPool2d(2),
            nn.ReLU(),
            
            nn.BatchNorm2d(32),
            nn.Conv2d(32, 64, 3, padding = 1),
            nn.MaxPool2d(2),
            nn.ReLU(),
            
            nn.BatchNorm2d(64),
            nn.Conv2d(64, 128, 3, padding = 1),
            nn.ReLU(),
            
            nn.BatchNorm2d(128),
            nn.Conv2d(128, 256, 3, padding = 1),
            nn.MaxPool2d(2),
            nn.ReLU())
        
        
        self.embedding = nn.Sequential(
            nn.BatchNorm1d(9216),
            nn.Linear(9216, 4608),
            nn.ReLU(),
            
            nn.BatchNorm1d(4608),
            nn.Linear(4608, 2304),
            nn.ReLU(),
            
            nn.BatchNorm1d(2304),
            nn.Linear(2304, 1152),
            nn.ReLU(),
            
            nn.BatchNorm1d(1152),
            nn.Linear(1152, 256)
            )       
        
    def forward(self, x):
        x = self.feature_extractor(x)
        x = self.embedding(x.view(x.shape[0], -1))
        return x
    
class ContrastiveLoss(nn.Module):
    def __init__(self, m = 2):
        super(ContrastiveLoss, self).__init__()
        #Margine
        self.m = m
        
    def forward(self, anchor_img, validation_img, label):
        d = nn.functional.pairwise_distance(anchor_img, validation_img)
        loss = 0.5 * label.float() * torch.pow(d,2) + \
            0.5 * (1 - label.float()) * torch.pow(torch.clamp(self.m - d, min = 0), 2)
            
        return loss.mean()
 
class AverageValueMeter():
    def __init__(self):
        self.reset()
        
    def reset(self):
        self.sum = 0
        self.num = 0

    def add(self, value, num):
        self.sum += value*num
        self.num += num
    
    def value(self):
        try:
            return self.sum/self.num
        except:
            return None
  
        
#RETE CUSTOM
class SiameseNetworkTask():
    def __init__(self,
                 embedding_net,
                 lr = 0.001,
                 momentum = 0.99,
                 margin = 2):
        
        self.criterion = ContrastiveLoss(margin)
        self.embedding_net = embedding_net
        self.optimizers = SGD(self.embedding_net.parameters(), lr, momentum = momentum, weight_decay=0.1)
        
        self.loss_meter = AverageValueMeter()
        self.acc_meter = AverageValueMeter()
        
    def training_step(self, train_loader, val_loader, exp_name = 'loggy', epochs = 50,
                      logdir = 'C:\\Users\\simon\\Desktop\\ML_Project\\mylogs'):
        
        writer = SummaryWriter(join(logdir, exp_name))
        
        device = 'cuda' if torch.cuda.is_available else 'cpu'
        self.embedding_net.to(device)
        
        loader = {
            'train': train_loader,
            'test': val_loader
        }
        
        global_step = 0
        for e in range(epochs):
            for mode in ['train', 'test']:
                print(mode)
                
                self.loss_meter.reset()
                self.acc_meter.reset()
                self.embedding_net.train() if mode == 'train' else self.embedding_net.eval()
                
                with torch.set_grad_enabled(mode == 'train'):
                    for i, batch in enumerate(loader[mode]):
                        anchor, valid, label = batch
                        anchor = anchor.to(device)
                        valid = valid.to(device)
                        label = label.to(device)
                        
                        emb1 = self.embedding_net(anchor)
                        emb2 = self.embedding_net(valid)
                        
                        n = anchor.shape[0]
                        global_step += n
                        
                        loss = self.criterion(emb1, emb2, label)
                        preds = self.predict(emb1, emb2)
        
                        acc = accuracy_score(label.to('cpu'), preds.to('cpu'))
                        
                        if mode == 'train':
                            loss.backward()
                            self.optimizers.step()
                            self.optimizers.zero_grad()
                            
                        self.loss_meter.add(loss.item(), n)
                        self.acc_meter.add(acc, n)
                        
                        if mode == 'train':
                            writer.add_scalar('loss/train', self.loss_meter.value(), global_step = global_step)
                            writer.add_scalar('accuracy/train', self.acc_meter.value(), global_step = global_step)
                            
                        print("Batch: %d" % i + " ===> [Loss: %f]" % loss + "[Accuracy: %f]" % acc)
                        
                writer.add_scalar('loss/' + mode, self.loss_meter.value(), global_step = global_step)
                writer.add_scalar('accuracy/' + mode, self.acc_meter.value(), global_step=global_step)
                
            print("Fine epoca: %d" % e)
            
        return self.embedding_net
            
    
    def test_accuracy(self, test_loader, neighbors = 3):
    
        device = 'cuda' if torch.cuda.is_available else 'cpu'
        self.acc_meter.reset()
        self.embedding_net.to(device)
        self.embedding_net.eval()
        
        with torch.set_grad_enabled(False):
            for i, batch in enumerate(test_loader):
                anchor, valid, y_true = batch
                anchor = anchor.to(device)
                valid = valid.to(device)
                y_true = y_true.to(device)
                
                emb1 = self.embedding_net(anchor)
                emb2 = self.embedding_net(valid)
                
                n = anchor.shape[0]
                
                preds = self.predict(emb1, emb2)
                acc = accuracy_score(y_true.to('cpu'), preds.to('cpu'))
                
                self.acc_meter.add(acc, n)
                print("[Batch: %d" % i + "] ==> %f" % acc)
            
        acc = self.acc_meter.value()
        
        return acc
            
    
    def predict(self, emb1, emb2, threshold = 0.5):
        d = nn.functional.pairwise_distance(emb1, emb2)
        preds = (d < threshold).float()
        
        return preds      
        
            
