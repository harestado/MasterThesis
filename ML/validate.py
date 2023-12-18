import torch

seed = 42
torch.manual_seed(1337)

def validate(best_model, device, dataloader, name):
    correct = 0
    total = 0
    y_true = []
    y_pred = []
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

    print('Accuracy of the network on the {} images: {:.3f}'.format(name, (100 * correct / total)))
    return y_true, y_pred