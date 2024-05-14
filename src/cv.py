import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import ConcatDataset, SubsetRandomSampler, DataLoader

from sklearn.model_selection import KFold
from sklearn.metrics import accuracy_score, f1_score, roc_curve, confusion_matrix, recall_score, precision_score

from data import DataHandler
from model import CNN
from hyperparameters import Hyperparameters as hp

def print_test_metrics(y_true, y_pred):
    test_acc = accuracy_score(y_true, y_pred)
    test_f1 = f1_score(y_true, y_pred, average='weighted')
    test_recall = recall_score(y_true, y_pred, average='weighted')
    precision = precision_score(y_true, y_pred, average='weighted')
    cm = confusion_matrix(y_true, y_pred)
    tag = "Test"
    print(
        f"{tag} Acc: {test_acc:.4f}, \
        {tag} F1: {test_f1:.4f}, \
        {tag} Recall: {test_recall:.4f}, \
        {tag} Precision: {precision:.4f}, \
        {tag} CM:\n{cm}"
    )

def reset_weights(m):
  '''
    Try resetting model weights to avoid
    weight leakage.
  '''
  for layer in m.children():
   if hasattr(layer, 'reset_parameters'):
    print(f'Reset trainable parameters of layer = {layer}')
    layer.reset_parameters()

data_handler = DataHandler()

train_loader, valid_loader, test_loader = data_handler.get_dataset()

dataset = ConcatDataset([train_loader.dataset, valid_loader.dataset, test_loader.dataset])


k_folds = 3
kfold = KFold(n_splits=k_folds, shuffle=True, random_state=42)

for fold, (train_ids, test_ids) in enumerate(kfold.split(dataset)):
    print(f'FOLD {fold}')
    print('--------------------------------')
    print(len(train_ids))
    print(len(test_ids))

    train_subsampler = SubsetRandomSampler(train_ids)
    test_subsampler = SubsetRandomSampler(test_ids)

    trainloader = DataLoader(
                      dataset, 
                      batch_size=hp.batch_size, sampler=train_subsampler)
    testloader = DataLoader(
                      dataset,
                      batch_size=hp.batch_size, sampler=test_subsampler)
    
    model = CNN()
    model.apply(reset_weights)
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.Adam(model.parameters(), lr=hp.learning_rate)

    for epoch in range(hp.num_epochs):
        model.train()
        running_loss = 0.0
        for inputs, labels in trainloader:
            optimizer.zero_grad()
            outputs = model(inputs)
            loss = criterion(outputs, labels)
            loss.backward()
            optimizer.step()
            running_loss += loss.item()
        
        train_loss = running_loss/len(trainloader)
        print(
            f"Epoch {epoch+1}/{hp.num_epochs}, Train Loss: {train_loss:.4f}"
        )
        
    print('Training process has finished. Saving trained model.')
    torch.save(model.state_dict(), f"cv_models/model_{fold}.pth")

    # Test the model
    model.eval()
    test_predictions = []
    test_true_labels = []
    class_probs = []
    with torch.no_grad():
        for inputs, labels in testloader:
            outputs = model(inputs)
            _, predicted = torch.max(outputs.data, 1)

            test_predictions.extend(predicted.cpu().numpy())
            test_true_labels.extend(labels.cpu().numpy())
            class_probs.extend(torch.nn.functional.softmax(outputs, dim=1).cpu().numpy())
    print_test_metrics(test_true_labels, test_predictions)