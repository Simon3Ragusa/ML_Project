# -*- coding: utf-8 -*-
"""
Created on Thu Feb  1 20:52:15 2024

@author: simon
"""
'''
from tensorflow.keras.models import Model
from tensorflow.keras.layers import Layer, Conv2D, Dense, MaxPooling2D, Input, Flatten
import tensorflow as tf
from tensorflow.keras.metrics import Precision, Recall
import torch

def embedding_block():
    inp = Input(shape=(100,100,3), name='input_image')
    c1 = Conv2D(64, (10,10), activation='relu')(inp)
    
    m1 = MaxPooling2D(64, (2,2), padding='same')(c1)
    c2 = Conv2D(128, (7,7), activation='relu')(m1)
    
    m2 = MaxPooling2D(64, (2,2), padding='same')(c2)
    c3 = Conv2D(128, (4,4), activation='relu')(m2)
    
    m3 = MaxPooling2D(64, (2,2), padding='same')(c3)
    c4 = Conv2D(256, (4,4), activation='relu')(m3)
    
    f1 = Flatten()(c4)
    d1 = Dense(4096, activation='sigmoid')(f1)
    
    return(Model(inputs=[inp], outputs=[d1], name='embedding'))

class Distancelayer(Layer):
    # Init method - inheritance
   def __init__(self):
       super().__init__()
      
   def call(self, input_embedding, validation_embedding):
       return tf.math.abs(input_embedding - validation_embedding)
   
embedding = embedding_block()
    
def make_siamese_model():
    
    # Anchor image input in the network
    input_image = Input(name='input_img', shape=(100,100,3))
    
    # Validation image in the network 
    validation_image = Input(name='validation_img', shape=(100,100,3))
    
    # Combine siamese distance components
    siamese_layer = Distancelayer()
    siamese_layer._name = 'distance'
    distances = siamese_layer(embedding(input_image), embedding(validation_image))
    
    # Classification layer 
    classifier = Dense(1, activation='sigmoid')(distances)
    
    return Model(inputs=[input_image, validation_image], outputs=classifier, name='SiameseNetwork')
'''

from os.path import join
from torch import nn
import torch
import pytorch_lightning as pl
from torch.optim import SGD
from torch.utils.tensorboard import SummaryWriter
from sklearn.metrics import accuracy_score

torch.cuda.empty_cache()
torch.cuda.set_per_process_memory_fraction(0.5)

'''
class EmbeddingNetwork(nn.Module):
    def __init__(self):
        super(EmbeddingNetwork, self).__init__()
        
        self.model = vgg16()
        self.model.classifier[6] = nn.Linear(4096, 256)
        self.margin = 2
        
        print("Rete di embedding creata")
        
    def forward(self, x):
        return self.model(x)
'''

class EmbeddingNet(nn.Module):
    #Ex-modello
    
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
            
        #Escono immagini da 256 x 12 x 12 = 36864
        
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
    '''
        
    def __init__(self):
        super(EmbeddingNet, self).__init__()
        
        self.feature_extractor = nn.Sequential(
            #First convolution
            nn.Conv2d(3, 64, 10),
            nn.MaxPool2d(2),
            nn.ReLU(),
            
            #Second convolution
            #nn.BatchNorm2d(64),
            nn.Conv2d(64, 128, 7),
            nn.MaxPool2d(2),
            nn.ReLU(),
            
            #Third convolution
            #nn.BatchNorm2d(128),
            nn.Conv2d(128, 128, 4),
            nn.MaxPool2d(2),
            nn.ReLU(),
            
        
            #Fourth convolution
            #nn.BatchNorm2d(128),
            nn.Conv2d(128, 256, 4),
            nn.ReLU())
        
            
        #Escono immagini da 256 x 12 x 12 = 36864
        
        self.embedding = nn.Sequential(
            #nn.BatchNorm1d(6400),
            nn.Linear(6400, 4096))
        '''
        
        
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
        #print("Calcolo della loss")
        d = nn.functional.pairwise_distance(anchor_img, validation_img)
        loss = 0.5 * (1 - label.float()) * torch.pow(d,2) + \
            0.5 * label.float() * torch.pow(torch.clamp(self.m - d, min = 0), 2)
            
        #print("Loss calcolata")
            
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
        self.optimizers = SGD(self.embedding_net.parameters(), lr, momentum = momentum)
        
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
                        
                        
                        if mode == 'train':
                            loss.backward()
                            self.optimizers.step()
                            self.optimizers.zero_grad()
                            
                        self.loss_meter.add(loss.item(), n)
                        
                        if mode == 'train':
                            writer.add_scalar('loss/train', self.loss_meter.value(), global_step = global_step)
                            
                        print("Batch: %d" % i + " ===> [Loss: %f]" % loss)
                        
                writer.add_scalar('loss/' + mode, self.loss_meter.value(), global_step = global_step)
                
            #torch.save(self.embedding_net.state_dict(), '%s-%d.pth' % (exp_name, e+1))
            
            print("Fine epoca: %d" % e)
            
        return self.embedding_net
            
    def predict(self, anchor, validation):
        
        emb1 = self.embedding_net(anchor)
        emb2 = self.embedding_net(validation)
        
        distance = nn.functional.pairwise_distance(emb1, emb2)
        
        return distance
            
                                      
class TripletNetworkTask():
    def __init__(self,
                 embedding_net,
                 lr = 0.001,
                 momentum = 0.99,
                 margin = 2):
        
        self.criterion = nn.TripletMarginLoss(margin)
        self.embedding_net = embedding_net
        self.optimizers = SGD(self.embedding_net.parameters(), lr, momentum = momentum)
        
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
                        anchor, positive, negative = batch
                        anchor = anchor.to(device)
                        positive = positive.to(device)
                        negative = negative.to(device)
                        
                        emb1 = self.embedding_net(anchor)
                        emb2 = self.embedding_net(positive)
                        emb3 = self.embedding_net(negative)
                        
                        n = anchor.shape[0]
                        global_step += n
                        
                        loss = self.criterion(emb1, emb2, emb3)
                        
                        if mode == 'train':
                            loss.backward()
                            self.optimizers.step()
                            self.optimizers.zero_grad()
                            
                        self.loss_meter.add(loss.item(), n)
                        
                        if mode == 'train':
                            writer.add_scalar('loss/train', self.loss_meter.value(), global_step = global_step)
                            
                        print("Batch: %d" % i + " ===> [Loss: %f]" % loss)
                        
                writer.add_scalar('loss/' + mode, self.loss_meter.value(), global_step = global_step)
                
            #torch.save(self.embedding_net.state_dict(), '%s-%d.pth' % (exp_name, e+1))
            
            print("Fine epoca: %d" % e)
'''
#RETE Pytorch lightning

class SiameseNetworkTask(pl.LightningModule):
    def __init__(self,
                 embedding_network,
                 lr = 0.01,
                 momentum = 0.99,
                 margin = 2):
        
        super(SiameseNetworkTask, self).__init__()
        self.save_hyperparameters()
        self.embedding_net = embedding_network
        self.criterion = ContrastiveLoss(margin)
        
        print("Rete inizializzata")
        
        
    def forward(self, x):
        print("Forward")
        return self.model(x)
    
    def configure_optimizers(self):
        print("Optimizers configurati")
        return SGD(self.embedding_net.parameters(), self.hparams.lr, momentum=self.hparams.momentum)
    
    def training_step(self, batch, batch_idx):
        print("Inizio del training:")
        # preleviamo gli elementi I_i e I_j e l'etichetta l_ij
        I_i, I_j, l_ij = batch
        
        
        #l'implementazione della rete siamese è banale:
        #eseguiamo la embedding net sui due input
        phi_i = self.embedding_net(I_i)
        phi_j = self.embedding_net(I_j)

        #calcoliamo la loss
        l = self.criterion(phi_i, phi_j, l_ij)
        
        self.log('train/loss', l)
        
        print("Loss:")
        print(l)
        print("\n")
        
        return l
        
    def validation_step(self, batch, batch_idx):
        I_i, I_j, l_ij = batch
        phi_i = self.embedding_net(I_i)
        phi_j = self.embedding_net(I_j)
        l = self.criterion(phi_i, phi_j, l_ij)
        self.log('valid/loss', l)
'''