# -*- coding: utf-8 -*-
"""
Created on Thu Feb  1 17:02:02 2024

@author: simon
"""

import os
from MyModel import SiameseNetworkTask, EmbeddingNet
from MyDatasets import SiameseDataset
from PIL import Image
from matplotlib import pyplot as plt
from torchvision import transforms
import torch
import numpy as np
from sklearn.manifold import TSNE
from sklearn.neighbors import KNeighborsClassifier

means = (0.4332, 0.3757, 0.3340)
stds = (0.2983, 0.2732, 0.2665)

def extract_representations(model, loader):
    print("Calcolo")
    device = "cuda" if torch.cuda.is_available() else "cpu"
    model.eval()
    model.to(device)
    
    representations, labels = [], []
    
    for batch in loader:
        img = batch[0].to(device)
        emb = model(img)
        emb = emb.detach().to('cpu').numpy()
        labels.append(batch[1])
        representations.append(emb)
        
    return np.concatenate(representations), np.concatenate(labels)

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

    siamese_train_dataset = SiameseDataset(transform = transform)
    siamese_test_dataset = SiameseDataset(split = 'test', transform=transform)
    
    siamese_net_task = SiameseNetworkTask(EmbeddingNet())
    
    menu_choice = input("Face Recognition:\n1 - Allena il modello\n2 - Testa il modello gia allenato\nScelta: ")
    
    if(int(menu_choice) == 1):
        
        train_loader = siamese_train_dataset.get_loader()['pairsTrain']
        test_loader = siamese_test_dataset.get_loader()['pairsTest']
        
        epochs = input("Inserisci il numero di epoche: ")
        exp_name = input("Inserisci il nome del log: ")
        
        print("Inizio il training...")
        siamese_net_task.embedding_net = siamese_net_task.training_step(train_loader, test_loader, exp_name= exp_name, epochs = int(epochs))
    
        choice = input("Vuoi salvare il modello?: ")
        if(choice == 's'):
            torch.save(siamese_net_task.embedding_net.state_dict(), 'my_model.pth')
            print("Modello salvato")
    else:
        
        test_accuracy_loader = siamese_test_dataset.get_loader()['test']
        
        siamese_net_task.embedding_net.load_state_dict(torch.load('my_model.pth'))
        
        print("Rete caricata")
        
        test_representations, test_labels = extract_representations(siamese_net_task.embedding_net, test_accuracy_loader)
    
        print("Rappresentazioni calcolate")
        knn_classifier = KNeighborsClassifier(n_neighbors=3)
        knn_classifier.fit(test_representations, test_labels)
    
        e = 0
        positives = 0
        
        print(len(test_accuracy_loader))
        
        for batch in test_accuracy_loader:
            img = batch[0].to('cuda')
            
            y_true = batch[1]
            
            person_representations = siamese_net_task.embedding_net(img)
            predicted_labels = knn_classifier.predict(person_representations.detach().to('cpu').numpy())
            
            for i, label in enumerate(predicted_labels):
                if label == y_true[i]:
                    positives += 1
            
            e += len(y_true)
            
            print("Successi ==> ", positives)
            print("Esperimenti totali ==> ", e)
            
        print("Successi: ", positives)
        print("Totali: ", e)

    
if __name__ == "__main__":
    main()
