import os
import numpy as np
import shutil
import torch
from torch import nn
from datetime import datetime
import csv
from torch.optim.lr_scheduler import ExponentialLR

def split_data():
    '''
    Funkcja do podzielenia danych na zbior treningowy i testowy obrazow pobranych z kaggla
    '''
    root_dir = 'raw-img/'               # lokalizacja oryginalnych obrazow
    new_dir = 'dataset/'                # lokalizacja folderu docelowego
    classes = os.listdir(root_dir)      # obrazy sa podzielone na foldery z nazwa klasy

    # przygotowanie w docelowym folderze folderow na dane treningowe i testowe z klasami
    for cl_ in classes:
        os.makedirs(new_dir + 'train/' + cl_)
        os.makedirs(new_dir + 'test/' + cl_)

    # dla kazdej klasy
    for cl_ in classes:
        src = root_dir + cl_                # folder to copy images from
        allFileNames = os.listdir(src)      # lista wszystki obrazow
        np.random.shuffle(allFileNames)     # losowe przemieszanie

        # split 0.75 = .75 training ratio i .25 test ratio
        train_FileNames, test_FileNames = np.split(np.array(allFileNames), [int(len(allFileNames)*0.75)])

        # przypisanie pelnej zrodlowej sciezki dla pliku
        train_FileNames = [src + '/' + name for name in train_FileNames]
        test_FileNames =  [src + '/' + name for name in test_FileNames]
        
        # informacja ile plikow dla danej klasy i jak split wyglada
        print('Total images  : '+ cl_ + ' ' +str(len(allFileNames)))
        print('Training : '+ cl_ + ' '+str(len(train_FileNames)))
        print('Testing : '+ cl_ + ' '+str(len(test_FileNames)))
        
        # kopiowanie obrazow do target folderu
        for name in train_FileNames:
            shutil.copy(name, new_dir + 'train/' + cl_)

        for name in test_FileNames:
            shutil.copy(name, new_dir + 'test/' + cl_)


def data_normalize_values(train_loader, device):
    '''
    Funkcja do obliczenia sredniej i odchylen standardowy na kanalach obrazow całego zbioru treningowego
    '''
    avg_mean = 0.0
    avg_std = 0.0

    for i, (data, _) in enumerate(train_loader):
        data = data.to(device)
        avg_mean += data.mean(axis=(0, 2, 3))
        avg_std += data.std(axis=(0, 2, 3))

    avg_mean /= len(train_loader)
    avg_std /= len(train_loader)

    return avg_mean, avg_std



def train_fine_tuning(model, learning_rate, train_loader, test_loader, device, num_epochs=5, param_group=True):
    '''
    Funkcja do uczenia modeli
    param_group = true - transfer learning
    '''
    session_id = datetime.now().strftime("%Y%m%d_%H%M%S")   # uniq id do rozroznienia sesji 
    # utworzenie folderu gdzie gromadzone beda wagi modeli w poszczegolnych epokach
    if not os.path.exists(f'models/{session_id}'):
        os.makedirs(f'models/{session_id}')
    #liczba obrazow w zbiorach
    train_size = len(train_loader.dataset.imgs)
    test_size = len(test_loader.dataset.imgs)

    # wagi warstwy wyjsciowej modelu
    if model._get_name() in ['Resnet', 'LeNet'] :
        excluded_weights = ["fc.weight", "fc.bias"]
        out_weights = model.fc.parameters()
    elif model._get_name() == 'AlexNet':
        excluded_weights = ["classifier.6.weight", "classifier.6.bias"]
        out_weights = model.classifier[6].parameters()

    if param_group: # transfer learning
        # zebranie wszystkie wagi poza warstwa wyjsciową
        params_1x = [param for name, param in model.named_parameters() if name not in excluded_weights]
        # wyjscie uczone wagami 10x wiekszymi niz reszta sieci
        optimizer = torch.optim.SGD([{'params': params_1x},
                                     {'params': out_weights, 'lr': learning_rate * 10}],
                                lr=learning_rate, weight_decay=0.001)
    else:
        optimizer = torch.optim.SGD(model.parameters(), lr=learning_rate, weight_decay=0.001)
    
    loss = nn.CrossEntropyLoss(reduction="none")    # funkcja celu
    
    # METRYKI Z UCZENIA SIECI
    metrics = {'train_loss':[], 'train_acc': [], 'test_acc': []}
    # TRAINING LOOP
    for epoch in range(num_epochs):
        print(f'Progress: {epoch+1}/{num_epochs} epochs')
        running_loss, running_accuracy = (0.0, 0.0)     # statystyki liczone na epoke

        for (data, labels) in train_loader:
            data, labels = data.to(device), labels.to(device)   # zaladowanie danych na gpu
            model.train()   # przestawienie sieci w tryb uczenia
            optimizer.zero_grad()   # zerowanie wektora gradientu
            labels_pred = model(data)   # predykcja z modelu
            l = loss(labels_pred, labels)   # obleczenie bledu sieci
            l.sum().backward()  # backpropagacja bledu
            optimizer.step()    # optymalizacja parametrów
            
            # obliczenie statystyk batcha
            running_loss += l.sum()
            running_accuracy += torch.sum(torch.argmax(labels_pred, axis=1) == labels)

        # obliczenie statystyk epoki
        epoch_loss = (running_loss / train_size).item()
        epoch_acc = (running_accuracy.double() / train_size).item()
        metrics['train_loss'].append(epoch_loss)
        metrics['train_acc'].append(epoch_acc)

        # ewaluacja co epoke bledu na zb testowym
        with torch.no_grad():
            model.eval()    # przestawienie w tryb ewaluacji
            test_acc = 0.0  # dokladnosc predykcji
            for (data, labels) in test_loader:
                data, labels = data.to(device), labels.to(device)
                labels_pred = model(data)
                labels_pred = torch.argmax(labels_pred, axis=1)
                test_acc += torch.sum(labels_pred == labels)
            
            epoch_test_acc = (test_acc / test_size).item()
            metrics['test_acc'].append(epoch_test_acc)
        
        # zapis modelu oraz informacji o uczeniu
        epoch_metrics = [session_id, epoch+1, round(epoch_loss, 3), round(epoch_acc, 3), round(epoch_test_acc, 3)]
        torch.save(model.state_dict(), f'models/{session_id}/{session_id}_{epoch+1}')
        with open('models/results.csv', 'a') as f:
            writer = csv.writer(f)
            writer.writerow(epoch_metrics)
        print(f'Epoch: {epoch_metrics[1]}, Loss {epoch_metrics[2]}, Train acc: {epoch_metrics[3]}, Test acc: {epoch_metrics[4]}')

    return metrics





def train_fine_tuning_long_train(model, learning_rate, train_loader, test_loader, device, num_epochs=10, param_group=True):
    '''
    Funkcja do uczenia modeli
    param_group = true - transfer learning
    '''
    session_id = datetime.now().strftime("%Y%m%d_%H%M%S")   # uniq id do rozroznienia sesji 
    # utworzenie folderu gdzie gromadzone beda wagi modeli w poszczegolnych epokach
    if not os.path.exists(f'models/{session_id}'):
        os.makedirs(f'models/{session_id}')
    #liczba obrazow w zbiorach
    train_size = len(train_loader.dataset.imgs)
    test_size = len(test_loader.dataset.imgs)

    # wagi warstwy wyjsciowej modelu
    excluded_weights = ["fc.weight", "fc.bias"]
    out_weights = model.fc.parameters()

    if param_group: # transfer learning
        # zebranie wszystkie wagi poza warstwa wyjsciową
        params_1x = [param for name, param in model.named_parameters() if name not in excluded_weights]
        # wyjscie uczone wagami 10x wiekszymi niz reszta sieci
        optimizer = torch.optim.SGD([{'params': params_1x},
                                     {'params': out_weights, 'lr': learning_rate * 10}],
                                lr=learning_rate, weight_decay=0.001)
    else:
        optimizer = torch.optim.SGD(model.parameters(), lr=learning_rate, weight_decay=0.001)
    
    loss = nn.CrossEntropyLoss(reduction="none")    # funkcja celu
    scheduler1 = ExponentialLR(optimizer, gamma=0.95)
    best_test = 0.0
    # METRYKI Z UCZENIA SIECI
    metrics = {'train_loss':[], 'train_acc': [], 'test_acc': []}
    # TRAINING LOOP
    for epoch in range(num_epochs):
        print(f'Progress: {epoch+1}/{num_epochs} epochs')
        running_loss, running_accuracy = (0.0, 0.0)     # statystyki liczone na epoke

        for (data, labels) in train_loader:
            data, labels = data.to(device), labels.to(device)   # zaladowanie danych na gpu
            model.train()   # przestawienie sieci w tryb uczenia
            optimizer.zero_grad()   # zerowanie wektora gradientu
            labels_pred = model(data)   # predykcja z modelu
            l = loss(labels_pred, labels)   # obleczenie bledu sieci
            l.sum().backward()  # backpropagacja bledu
            optimizer.step()    # optymalizacja parametrów
            
            # obliczenie statystyk batcha
            running_loss += l.sum()
            running_accuracy += torch.sum(torch.argmax(labels_pred, axis=1) == labels)

        scheduler1.step()
        # obliczenie statystyk epoki
        epoch_loss = (running_loss / train_size).item()
        epoch_acc = (running_accuracy.double() / train_size).item()
        metrics['train_loss'].append(epoch_loss)
        metrics['train_acc'].append(epoch_acc)

        # ewaluacja co epoke bledu na zb testowym
        with torch.no_grad():
            model.eval()    # przestawienie w tryb ewaluacji
            test_acc = 0.0  # dokladnosc predykcji
            for (data, labels) in test_loader:
                data, labels = data.to(device), labels.to(device)
                labels_pred = model(data)
                labels_pred = torch.argmax(labels_pred, axis=1)
                test_acc += torch.sum(labels_pred == labels)
            
            epoch_test_acc = (test_acc / test_size).item()
            metrics['test_acc'].append(epoch_test_acc)
        
        # zapis modelu oraz informacji o uczeniu
        epoch_metrics = [session_id, epoch+1, round(epoch_loss, 3), round(epoch_acc, 3), round(epoch_test_acc, 3)]
        if epoch_test_acc> best_test:
            torch.save(model.state_dict(), f'models/{session_id}/{session_id}')
            best_test = epoch_test_acc
        with open('models/results.csv', 'a') as f:
            writer = csv.writer(f)
            writer.writerow(epoch_metrics)
        print(f'Epoch: {epoch_metrics[1]}, Loss {epoch_metrics[2]}, Train acc: {epoch_metrics[3]}, Test acc: {epoch_metrics[4]}')

    return metrics
