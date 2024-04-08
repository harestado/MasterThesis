from datetime import datetime
from copy import deepcopy
import torch
import torch.optim as optim
import torch.nn as nn
import numpy as np
from sklearn.utils.class_weight import compute_class_weight
from sklearn.metrics import balanced_accuracy_score
from math import floor
import os

seed = 42
torch.manual_seed(1337)

def train_model(unet, device, trainloader, valloader, N_EPOCHS, learning_rate, dropout_rate, layer_size):
    
    # y = []
    # for data in trainloader:
    #     # Extract labels from the batch
    #     inputs, labels = data[0], data[1]
    #     # inputs, labels = inputs.to(device), labels.to(device)
    #     ls = deepcopy(labels)
    #     ls = ls.to('cpu')
    #     y.append(ls)
    # y = torch.cat(y)
    # y = y.to('cpu')
    # y = y.numpy()
    # y_unique, y_counts = np.unique(y, return_counts=True)
    # print(y_unique, y_counts)
    # class_weights = compute_class_weight(class_weight = 'balanced', classes = y_unique, y = y)
    # print(class_weights)
    # ls, y = None, None
    
    class_weights = [0.65833333, 0.75238095, 6.58333333]
    class_weights = torch.tensor(class_weights, dtype=torch.float)
    class_weights = class_weights.to(device)

    unet = unet.to(device)
    criterion = nn.CrossEntropyLoss(weight=class_weights)
    optimizer = optim.Adam(unet.parameters(), lr=torch.tensor(learning_rate, device=device), weight_decay=torch.tensor(0.025, device=device), fused=True)

    date = datetime.now().strftime('%d%m')
    data_dir = 'C:/Users/oscar/OneDrive - University of Bergen/Documents/Master/vsc/COBRE_learning/multilabel_regrouped/models/' + date + '_' + str(learning_rate) + '_' + str(dropout_rate) + '_' +  str(layer_size)
    if not os.path.isdir(data_dir):
        os.mkdir(data_dir)

    losses_train = []
    losses_val = []
    accuracies_train = []
    accuracies_val = []
    balanced_accuracies = []
    balanced_accuracies_train = []
    best_acc_val = 0
    best_bal_acc = 0
    best_acc_train = 0
    best_val_loss = 0
    n_batch_train = len(trainloader)
    n_batch_val = len(valloader)

    for epoch in range(1,N_EPOCHS+1,1):  # loop over the dataset multiple times
        ### TRAINING PHASE
        correct = 0
        total = 0
        # correct_bal = 0
        loss_train = 0
        y_pred_train = []
        y_true_train = []
        unet.train()
        for data in trainloader:
            # get the inputs; data is a list of [inputs, labels]
            inputs, labels = data[0], data[1]
            inputs, labels = inputs.to(device), labels.to(device)
            # zero the parameter gradients
            optimizer.zero_grad()

            # forward + backward + optimize
            outputs = unet(inputs)
            _, predicted = torch.max(outputs.data, 1)
            correct += (predicted == labels).sum().item()
            total += labels.size(0)
            for p, l in zip(predicted, labels):
                # correct_bal += ((p == l).sum().item() * class_weights[p])
                y_pred_train.append(p.sum().item())
                y_true_train.append(l.sum().item())

            loss = criterion(outputs, labels)
            loss.backward()
            optimizer.step()

            loss_train += loss.item()
        losses_train.append(loss_train/n_batch_train)

        current_acc_train = correct / total
        accuracies_train.append(current_acc_train)

        # current_bal_acc_train = correct_bal / total
        current_bal_acc_train = balanced_accuracy_score(y_true_train, y_pred_train)
        balanced_accuracies_train.append(current_bal_acc_train)

        ### VALIDATION PHASE
        correct = 0
        total = 0
        # correct_bal = 0
        loss_val = 0
        y_pred_val = []
        y_true_val = []
        unet.eval()
        with torch.no_grad():
            for data in valloader:
                inputs, labels = data[0], data[1]
                inputs, labels = inputs.to(device), labels.to(device)

                outputs = unet(inputs)
                _, predicted = torch.max(outputs.data, 1)
                correct += (predicted == labels).sum().item()
                total += labels.size(0)
                # correct_bal += ((predicted == labels).sum().item() * class_weights[predicted])
                y_pred_val.append(predicted.sum().item())
                y_true_val.append(labels.sum().item())

                loss = criterion(outputs, labels)
                loss_val += loss.item()
        current_val_loss = loss_val/n_batch_val
        losses_val.append(current_val_loss)

        current_acc_val = correct / total
        accuracies_val.append(current_acc_val)

        current_bal_acc = balanced_accuracy_score(y_true_val, y_pred_val)
        balanced_accuracies.append(current_bal_acc)

        print('{} | Epoch {}/{} | Train loss {:.7f} | Val loss {:.7f} | Train acc {:.3f} | Val acc {:.3f} | Train acc (bal) {:.3f} | Val acc (bal) {:.3f}'.format(
                    datetime.now().time().strftime("%H:%M:%S"), epoch, N_EPOCHS, loss_train / n_batch_train, current_val_loss, current_acc_train, current_acc_val, current_bal_acc_train, current_bal_acc))
        
        if(current_bal_acc > best_bal_acc):
            best_params = deepcopy(unet.state_dict())
            model_name = data_dir + '/best_model_0' + str(floor(current_bal_acc*100000)) + '_bal.pt'
            torch.save(unet, model_name)
            print('New best bal acc:', current_bal_acc, end='  -----------  ')
            print('Old bal acc:' , best_bal_acc)
            best_bal_acc = current_bal_acc
            best_val_loss = current_val_loss
            best_epoch = epoch
        elif(current_bal_acc == best_bal_acc) and (current_val_loss < best_val_loss):
            best_params = deepcopy(unet.state_dict())
            model_name = data_dir + '/best_model_0' + str(floor(current_bal_acc*100000)) + '_bal.pt'
            torch.save(unet, model_name)
            print('Saving model with lower loss:', current_val_loss)
            best_val_loss = current_val_loss
            best_epoch = epoch

    print('Finished Training')
    print('---------------------')
    data_dir = 'C:/Users/oscar/OneDrive - University of Bergen/Documents/Master/vsc/COBRE_learning/multilabel_regrouped/model_acc&loss/' + date + '_' + str(learning_rate) + '_' + str(dropout_rate) + '_' +  str(layer_size)
    if not os.path.isdir(data_dir):
        os.mkdir(data_dir)
    torch.save(losses_val, data_dir + '/losses_val')
    torch.save(losses_train, data_dir + '/losses_train')
    torch.save(accuracies_val, data_dir + '/acc_val')
    torch.save(accuracies_train, data_dir + '/acc_train')
    torch.save(balanced_accuracies, data_dir + '/bal_acc_val')
    torch.save(balanced_accuracies_train, data_dir + '/bal_acc_train')
    print('Saved losses and accuracies')
    return best_params, losses_val, losses_train, accuracies_val, accuracies_train, balanced_accuracies, balanced_accuracies_train, best_epoch
