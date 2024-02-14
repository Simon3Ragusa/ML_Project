# -*- coding: utf-8 -*-
"""
Created on Sat Feb 10 19:15:15 2024

@author: simon
"""

from MyModel import SiameseNetworkTask, EmbeddingNet
from MyDatasets import PairsDataset
import torch
import numpy as np
from torchvision import transforms
from sklearn.neighbors import KNeighborsClassifier
import random
from matplotlib import pyplot as plt

means = (0.4332, 0.3757, 0.3340)
stds = (0.2983, 0.2732, 0.2665)
BATCH_SIZE = 32

transform = transforms.Compose([
        transforms.Resize(100),
        transforms.ToTensor(),
        transforms.Normalize(means, stds)
    ])

def extract_representations(model, loader):
    print("Calcolo rappresentazioni delle immagini")
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

def demo(model, dataset, knn_classifier):
    
    plt.figure(figsize = (12, 5))
    model.eval()
    print("Benvenuto nella demo del nostro sistema di face recognition")
    while True:
        
        choice = input("Cosa vuoi fare?\n1 - Esegui un test randomico\n2 - Esci\nScelta: ")
        
        if int(choice) == 2:
            break
        
        test = random.choice(dataset)
        
        pil_img = test[0]
        transformed_img = transform(pil_img)
        transformed_img = transformed_img.to('cuda')
        y_true = test[1]
        
        plt.title("Etichetta reale: %d" % y_true)
        plt.imshow(pil_img)
        plt.show()
        
        pred = knn_classifier.predict(model(transformed_img.unsqueeze(0)).detach().to('cpu').numpy())
        
        print("Etichetta predetta: ", pred)
        
        print("\n")
        

#Carico la rete e il modello allenato
siamese_network = SiameseNetworkTask(EmbeddingNet())
siamese_network.embedding_net.load_state_dict(torch.load('my_model.pth'))

#Carico il dataset
dataset = PairsDataset(transform=transform)
test_loader = dataset.get_loaders(batch_size = BATCH_SIZE)['test_people']

#estraggo le rappresentazioni
test_representations, test_labels = extract_representations(siamese_network.embedding_net, test_loader)

#Instanzio e alleno il knn classifier
knn_classifier = KNeighborsClassifier(n_neighbors=3)
knn_classifier.fit(test_representations, test_labels)

dataset = PairsDataset()
demo(siamese_network.embedding_net, dataset.get_test_people_labels(), knn_classifier)