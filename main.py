# -*- coding: utf-8 -*-
"""
Created on Thu Feb  1 17:02:02 2024

@author: simon
"""

import os
from MyModel import SiameseNetworkTask, EmbeddingNet
from MyDatasets import PairsDataset
from PIL import Image
from matplotlib import pyplot as plt
from torchvision import transforms
import torch
import numpy as np
from sklearn.manifold import TSNE
from sklearn.neighbors import KNeighborsClassifier

means = (0.4332, 0.3757, 0.3340)
stds = (0.2983, 0.2732, 0.2665)
BATCH_SIZE = 16

def stampa_demo(loader):
    plt.figure(figsize = (12, 4))
               
    for i, batch in enumerate(loader):
        print("\nBatch: %d" % i)
        print("Label: %f" % batch[2][0])
        plt.subplot(1, 2, 1)
        plt.imshow(batch[0][0].permute(1, 2, 0))
        plt.subplot(1, 2, 2)
        plt.imshow(batch[1][0].permute(1, 2, 0))
        plt.show()

def main():
    
    transform = transforms.Compose([
            transforms.Resize(100),
            transforms.ToTensor(),
            transforms.Normalize(means, stds)
        ])

    siamese_dataset = PairsDataset(transform = transform)
    
    siamese_net_task = SiameseNetworkTask(EmbeddingNet(), lr = 0.001)
    
    menu_choice = input("Face Recognition:\n1 - Allena il modello\n2 - Testa il modello gia allenato\nScelta: ")
    
    if(int(menu_choice) == 1):
        
        train_loader = siamese_dataset.get_loaders(batch_size=BATCH_SIZE)['train']
        validation_loader = siamese_dataset.get_loaders(batch_size=BATCH_SIZE)['valid']
        
        epochs = input("Inserisci il numero di epoche: ")
        exp_name = input("Inserisci il nome del log: ")
        
        print("Inizio il training...")
        siamese_net_task.embedding_net = siamese_net_task.training_step(train_loader, validation_loader, exp_name= exp_name, epochs = int(epochs))
    
        choice = input("Vuoi salvare il modello?: ")
        if(choice == 's'):
            torch.save(siamese_net_task.embedding_net.state_dict(), 'my_model.pth')
            print("Modello salvato")
            
    else:
        
        test_accuracy_loader = siamese_dataset.get_loaders(batch_size=BATCH_SIZE)['test_people']
        
        siamese_net_task.embedding_net.load_state_dict(torch.load('my_model.pth'))
        
        print("Rete caricata")
        
        acc = siamese_net_task.test_accuracy(test_accuracy_loader, neighbors=3)
        
        print("{:d}%".format(int(acc)))

    
if __name__ == "__main__":
    main()
