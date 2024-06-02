import torch
from sklearn.metrics import accuracy_score, f1_score, balanced_accuracy_score, recall_score, precision_score

# setting seeds for reproducibility
seed = 42
torch.manual_seed(1337)

def validate(best_model, device, dataloader, class_weights):
    class_weights = torch.tensor(class_weights, dtype=torch.float) # class weights for the cross entropy loss function
    class_weights = class_weights.to(device)
    criterion = torch.nn.CrossEntropyLoss(weight=class_weights)
    n_batch_val = len(dataloader)

    correct = 0
    total = 0
    y_true = []
    y_pred = []
    loss_val = 0
    best_model.eval() # no gradient calculation when validating/testing
    with torch.no_grad(): # same here
        for data in dataloader:
            inputs, labels = data[0], data[1]
            inputs, labels = inputs.to(device), labels.to(device) # to cuda or cpu
            outputs = best_model(inputs)
            _, predicted = torch.max(outputs.data, 1)
            total += labels.size(0)
            correct += (predicted == labels).sum().item() # sum all correct predictions
            for i in range(len(predicted)):
                y_pred.append(predicted[i].item())
                y_true.append(labels[i].item())
            
            loss = criterion(outputs, labels) # calculate loss
            loss_val += loss.item() # add to loss
    loss_val = loss_val/n_batch_val # average loss

    print('Loss: ', loss_val)
    print('Accuracy:', accuracy_score(y_true, y_pred))
    print('F1-score:', f1_score(y_true, y_pred, average='weighted'))
    print('Balanced accuracy score:', balanced_accuracy_score(y_true, y_pred))
    print('Recall:', recall_score(y_true, y_pred, average='weighted'))
    print('Precision:', precision_score(y_true, y_pred, average='weighted'))
    return y_true, y_pred
