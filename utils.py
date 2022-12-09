import os
import numpy as np
import shutil

def split_data():
    '''
    split struktury folderu z obrazami by miec podzial na zbior uczacy i testowy
    usage -> utils.split_data()
    '''
    root_dir = 'raw-img/'
    new_dir = 'dataset/'
    classes = os.listdir('raw-img/')

    for cl_ in classes:
        os.makedirs(new_dir + 'train/' + cl_)
        os.makedirs(new_dir + 'test/' + cl_)

    for cl_ in classes:
        src = root_dir + cl_ # folder to copy images from
        allFileNames = os.listdir(src)
        np.random.shuffle(allFileNames)

        ## here 0.75 = training ratio
        train_FileNames, test_FileNames = np.split(
            np.array(allFileNames), [int(len(allFileNames)*0.75)])

        # Converting file names from array to list
        train_FileNames = [src+'/'+ name for name in train_FileNames]
        test_FileNames = [src+'/' + name for name in test_FileNames]

        print('Total images  : '+ cl_ + ' ' +str(len(allFileNames)))
        print('Training : '+ cl_ + ' '+str(len(train_FileNames)))
        print('Testing : '+ cl_ + ' '+str(len(test_FileNames)))
        
        ## Copy pasting images to target directory

        for name in train_FileNames:
            shutil.copy(name, new_dir + 'train/' + cl_)

        for name in test_FileNames:
            shutil.copy(name, new_dir + 'test/' + cl_)

def data_normalize_values(train_loader, device):
    avg_mean = 0.0
    avg_std = 0.0

    for i, (data, _) in enumerate(train_loader):
        data = data.to(device)
        avg_mean += data.mean(axis=(0, 2, 3))
        avg_std += data.std(axis=(0, 2, 3))

    avg_mean /= len(train_loader)
    avg_std /= len(train_loader)

    return avg_mean, avg_std


import torch
from torch import nn
from datetime import datetime
import csv

def train_fine_tuning(model, learning_rate, train_loader, test_loader, device, num_epochs=5, param_group=True):
    '''
    param_group = true - transfer learning
    '''
    session_id = datetime.now().strftime("%Y%m%d_%H%M%S")
    if not os.path.exists(f'models/{session_id}'):
        os.makedirs(f'models/{session_id}')
    train_size = len(train_loader.dataset.imgs)
    test_size = len(test_loader.dataset.imgs)
    # funkcja celu
    loss = nn.CrossEntropyLoss(reduction="none")

    if model._get_name() in ['Resnet', 'LeNet'] :
        excluded_weights = ["fc.weight", "fc.bias"]
        out_weights = model.fc.parameters()
    elif model._get_name() == 'AlexNet':
        excluded_weights = ["classifier.6.weight", "classifier.6.bias"]
        out_weights = model.classifier[6].parameters()

    if param_group:
        # wszystkie wagi poza warstwa wyjsciową
        params_1x = [param for name, param in model.named_parameters() if name not in excluded_weights]
        # wyjscie uczone silniejszymi wagami niz reszta sieci
        optimizer = torch.optim.SGD([{'params': params_1x},
                                     {'params': out_weights, 'lr': learning_rate * 10}],
                                lr=learning_rate, weight_decay=0.001)
    else:
        optimizer = torch.optim.SGD(model.parameters(), lr=learning_rate, weight_decay=0.001)
    
    # METRYKI Z UCZENIA SIECI
    metrics = {'train_loss':[], 'train_acc': [], 'test_acc': []}
    # TRAINING LOOP
    for epoch in range(num_epochs):
        print(f'Progress: {epoch+1}/{num_epochs} epochs')
        # loss i dokladnosc per epoch
        running_loss = 0.0
        running_accuracy = 0.0

        for (data, labels) in train_loader:
            # wczytanie na gpu danych
            data, labels = data.to(device), labels.to(device)
            # ustawienie sieci w trybie uczenia (weight decay)
            model.train()
            # wyzrowanie gradientu
            optimizer.zero_grad()
            # predykcja
            labels_pred = model(data)
            # obleczenie bledzu
            l = loss(labels_pred, labels)
            l.sum().backward()
            # optymalizacja parametrów
            optimizer.step() 
            # per batch statistics
            running_loss += l.sum()
            running_accuracy += torch.sum(torch.argmax(labels_pred, axis=1) == labels)

        # per epoch statistics
        epoch_loss = (running_loss / train_size).item()
        epoch_acc = (running_accuracy.double() / train_size).item()
        metrics['train_loss'].append(epoch_loss)
        metrics['train_acc'].append(epoch_acc)

        # EVAL TRAIN ACCURACY
        with torch.no_grad():
            model.eval() # disables dropout
            test_acc = 0.0
            for (data, labels) in test_loader:
                data, labels = data.to(device), labels.to(device)
                labels_pred = model(data)
                labels_pred = torch.argmax(labels_pred, axis=1)
                test_acc += torch.sum(labels_pred == labels)
            
            epoch_test_acc = (test_acc / test_size).item()
            metrics['test_acc'].append(epoch_test_acc)
        
        # save model parameters and per epoch informations
        epoch_metrics = [session_id, epoch+1, round(epoch_loss, 3), round(epoch_acc, 3), round(epoch_test_acc, 3)]
        torch.save(model.state_dict(), f'models/{session_id}/{session_id}_{epoch+1}')
        with open('models/results.csv', 'a') as f:
            writer = csv.writer(f)
            writer.writerow(epoch_metrics)
        print(f'Epoch: {epoch_metrics[1]}, Loss {epoch_metrics[2]}, Train acc: {epoch_metrics[3]}, Test acc: {epoch_metrics[4]}')

    return metrics