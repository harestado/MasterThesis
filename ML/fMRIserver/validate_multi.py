import torch
from sklearn.metrics import accuracy_score, f1_score, balanced_accuracy_score, recall_score, precision_score

seed = 42
torch.manual_seed(1337)

def validate(best_model, device, dataloader):
    # class_weights = [0.65833333, 0.75238095, 6.58333333]
    class_weights = torch.tensor(class_weights, dtype=torch.float)
    class_weights = class_weights.to(device)
    criterion = torch.nn.CrossEntropyLoss(weight=class_weights)
    n_batch_val = len(dataloader)

    correct = 0
    total = 0
    y_true = []
    y_pred = []
    loss_val = 0
    # since we're not training, we don't need to calculate the gradients for our outputs
    best_model.eval()
    with torch.no_grad():
        for data in dataloader:
            inputs, labels = data[0], data[1]
            # labels = labels.type(torch.LongTensor)   # casting to long
            inputs, labels = inputs.to(device), labels.to(device)
            # calculate outputs by running images through the network
            outputs = best_model(inputs)
            # the class with the highest energy is what we choose as prediction
            _, predicted = torch.max(outputs.data, 1)
            total += labels.size(0)
            correct += (predicted == labels).sum().item()
            for i in range(len(predicted)):
                y_pred.append(predicted[i].item())
                y_true.append(labels[i].item())
            
            loss = criterion(outputs, labels)
            loss_val += loss.item()
    loss_val = loss_val/n_batch_val

    print('Loss: ', loss_val)
    print('Accuracy:', accuracy_score(y_true, y_pred))
    print('F1-score:', f1_score(y_true, y_pred, average='weighted'))
    print('Balanced accuracy score:', balanced_accuracy_score(y_true, y_pred))
    print('Recall:', recall_score(y_true, y_pred, average='weighted'))
    print('Precision:', precision_score(y_true, y_pred, average='weighted'))
    return y_true, y_pred
