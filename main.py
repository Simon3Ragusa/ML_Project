# -*- coding: utf-8 -*-
"""
Created on Thu Feb  1 17:02:02 2024

@author: simon
"""

from MyModel import SiameseNetworkTask, EmbeddingNet
from MyDatasets import PairsDataset
from torchvision import transforms
import torch

means = (0.4332, 0.3757, 0.3340)
stds = (0.2983, 0.2732, 0.2665)
BATCH_SIZE = 32


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
        test_loader = siamese_dataset.get_loaders(batch_size=BATCH_SIZE)['test']
        
        siamese_net_task.embedding_net.load_state_dict(torch.load('my_model.pth'))
        
        print("Rete caricata")
        
        acc = siamese_net_task.test_accuracy(test_loader)
        
        print("Accuracy di test: %2f" % acc)
        
    
if __name__ == "__main__":
    main()
