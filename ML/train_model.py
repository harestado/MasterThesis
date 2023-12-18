from datetime import datetime
from copy import deepcopy
import torch
import torch.optim as optim
import torch.nn as nn

seed = 42
torch.manual_seed(1337)

def train_model(unet, device, trainloader, valloader, N_EPOCHS, learning_rate):
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.Adam(unet.parameters(), lr=learning_rate) #endre på lr for å få ned loss?
    unet = unet.to(device)

    losses_train = []
    losses_val = []
    correct = 0
    total = 0
    best_acc_val = 0
    n_batch_train = len(trainloader)
    n_batch_val = len(valloader)

    for epoch in range(1,N_EPOCHS+1,1):  # loop over the dataset multiple times
        ### TRAINING PHASE
        loss_train = 0
        unet.train()
        for data in trainloader:
            # get the inputs; data is a list of [inputs, labels]
            inputs, labels = data[0], data[1]
            inputs, labels = inputs.to(device), labels.to(device)
            # zero the parameter gradients
            optimizer.zero_grad()

            # forward + backward + optimize
            outputs = unet(inputs)
            loss = criterion(outputs, labels)
            loss.backward()
            optimizer.step()

            loss_train += loss.item()
        losses_train.append(loss_train/n_batch_train)

        ### VALIDATION PHASE
        correct = 0
        total = 0
        loss_val = 0
        unet.eval()
        with torch.no_grad():
            for data in valloader:
                inputs, labels = data[0], data[1]
                inputs, labels = inputs.to(device), labels.to(device)

                outputs = unet(inputs)
                _, predicted = torch.max(outputs.data, 1)
                correct += (predicted == labels).sum().item()
                total += labels.size(0)

                loss = criterion(outputs, labels)
                loss_val += loss.item()
        losses_val.append(loss_val/n_batch_val)

        current_acc_val = 100 * correct / total

        print('{}  |  Epoch {}/{}  |  Training loss {:.9f} |  Validation loss {:.9f} | Validation accuracy {:.3f}'.format(
                    datetime.now().time(), epoch, N_EPOCHS, loss_train / n_batch_train, loss_val/n_batch_val, current_acc_val))
        
        if(current_acc_val > best_acc_val):
            # torch.save(unet.state_dict().copy(), 'models/best_model_params_nou.pt')
            best_params = deepcopy(unet.state_dict())
            torch.save(best_params, 'models/best_model_params_nou.pt')
            # torch.save(unet, 'models/best_model_nou.pt')
            print('New best acc:', current_acc_val, end='  ------  ')
            print('Old acc:' , best_acc_val)
            best_acc_val = current_acc_val

    print('Finished Training')
    return best_params, losses_val, losses_train