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

# setting seeds for reproducibility
seed = 42
torch.manual_seed(1337)

def train_model(net, device, trainloader, valloader, N_EPOCHS, learning_rate, dropout_rate, layer_size, mode, class_weights):
    
    # print(class_weights)
    class_weights = torch.tensor(class_weights, dtype=torch.float) # class weights for the cross entropy loss function
    class_weights = class_weights.to(device)

    net = net.to(device) # to cuda or cpu
    criterion = nn.CrossEntropyLoss(weight=class_weights) # loss func
    optimizer = optim.Adam(net.parameters(), lr=torch.tensor(learning_rate, device=device), weight_decay=torch.tensor(0.025, device=device)) # optimizer

    date = datetime.now().strftime('%d%m')
    data_dir = '/models/' + date + '_' + str(learning_rate) + '_' + str(dropout_rate) + '_' +  str(layer_size) # for storing models and results
    if not os.path.isdir(data_dir):
        os.mkdir(data_dir)

    print('Training start at {}'.format(datetime.now().time().strftime("%H:%M:%S")))
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

    for epoch in range(1,N_EPOCHS+1,1):
        # TRAINING PHASE
        correct = 0
        total = 0
        loss_train = 0
        y_pred_train = []
        y_true_train = []
        net.train() # in train mode, calculate grads
        for data in trainloader:
            inputs, labels = data[0], data[1]
            inputs, labels = inputs.to(device), labels.to(device) # to cuda or cpu
            optimizer.zero_grad()

            outputs = net(inputs)
            _, predicted = torch.max(outputs.data, 1)
            correct += (predicted == labels).sum().item()  # sum all correct predictions
            total += labels.size(0)
            for p, l in zip(predicted, labels):
                y_pred_train.append(p.sum().item())
                y_true_train.append(l.sum().item())

            loss = criterion(outputs, labels)
            loss.backward() # backprop
            optimizer.step() # parameter update

            loss_train += loss.item()
        losses_train.append(loss_train/n_batch_train)

        current_acc_train = correct / total
        accuracies_train.append(current_acc_train)

        current_bal_acc_train = balanced_accuracy_score(y_true_train, y_pred_train)
        balanced_accuracies_train.append(current_bal_acc_train)

        # VALIDATION PHASE
        correct = 0
        total = 0
        loss_val = 0
        y_pred_val = []
        y_true_val = []
        net.eval() # no grad calc
        with torch.no_grad():
            for data in valloader:
                inputs, labels = data[0], data[1]
                inputs, labels = inputs.to(device), labels.to(device)

                outputs = net(inputs)
                _, predicted = torch.max(outputs.data, 1)
                correct += (predicted == labels).sum().item()
                total += labels.size(0)
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


        # SAVE BEST MODEL
        if(mode=='bal'):
            if(current_bal_acc > best_bal_acc):
                best_params = deepcopy(net.state_dict())
                model_name = data_dir + '/best_model_0' + str(floor(current_bal_acc*100000)) + '_bal.pt'
                torch.save(net, model_name)
                print('New best bal acc:', current_bal_acc, end='  -----------  ')
                print('Old bal acc:' , best_bal_acc)
                best_bal_acc = current_bal_acc
                best_val_loss = current_val_loss
                best_epoch = epoch
            elif(current_bal_acc == best_bal_acc) and (current_val_loss < best_val_loss):
                best_params = deepcopy(net.state_dict())
                model_name = data_dir + '/best_model_0' + str(floor(current_bal_acc*100000)) + '_bal.pt'
                torch.save(net, model_name)
                print('Saving model with lower loss:', current_val_loss)
                best_val_loss = current_val_loss
                best_epoch = epoch
        elif(mode=='acc'):
            if(current_acc_val > best_acc_val):
                best_params = deepcopy(net.state_dict())
                model_name = data_dir + '/best_model_0' + str(floor(current_acc_val*100000)) + '.pt'
                torch.save(net, model_name)
                print('New best acc:', current_acc_val, end='  -----------  ')
                print('Old acc:' , best_acc_val)
                best_acc_val = current_acc_val
                best_val_loss = current_val_loss
                best_epoch = epoch
            elif(current_acc_val == best_acc_val) and (current_val_loss < best_val_loss):
                best_params = deepcopy(net.state_dict())
                model_name = data_dir + '/best_model_0' + str(floor(current_acc_val*100000)) + '.pt'
                torch.save(net, model_name)
                print('Saving model with lower loss with val acc:', current_acc_val, 'and train acc:', current_acc_train)
                best_val_loss = current_val_loss
                best_epoch = epoch
        else:
            print('Parameter "mode" is', str(mode), '-- needs to be "acc" or "bal".')
            return None

    print('Finished Training')
    print('---------------------')

    # SAVE LOSSES AND ACCS
    data_dir = '/model_acc&loss/' + date + '_' + str(learning_rate) + '_' + str(dropout_rate) + '_' +  str(layer_size)
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
