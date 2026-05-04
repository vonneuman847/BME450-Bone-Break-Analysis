# -*- coding: utf-8 -*-
"""
Created on Fri Mar  6 18:16:15 2026

@author: ThadD
"""
import numpy as np
import pandas as pd
import torch
from torch import nn
import torch.optim as optim
import torch.nn.functional as F
from torch.utils.data import DataLoader
from torch.utils.data import Dataset
import time
import matplotlib.pyplot as plt

total_time = time.time()

class ECGDataset(Dataset):
    def __init__(self, file_path):
        df = pd.read_csv(file_path)
        
        self.X = df.iloc[:, :-1].values.astype('float32')
        self.y = df.iloc[:, -1].values.astype('int64') #classes represented as numbers 0-4

        self.X = torch.tensor(self.X)
        self.y = torch.tensor(self.y)

        self.X = (self.X - self.X.mean(dim=1, keepdim=True)) / (self.X.std(dim=1, keepdim=True) + 1e-8)

    def __len__(self):
        return len(self.X)

    def __getitem__(self, idx):
        return self.X[idx], self.y[idx]

train_data = ECGDataset("mitbih_train.csv")

#Calculating weights of bins to improve avg loss
labels = train_data.y.numpy()
class_counts = np.bincount(labels)

print("Class counts:", class_counts)

class_weights = 1.0 / class_counts
class_weights = class_weights / class_weights.sum() * len(class_counts)

class_weights = torch.tensor(class_weights, dtype=torch.float32)

print("Class weights:", class_weights)

test_data = ECGDataset("mitbih_test.csv")

categories = ['N', 'S', 'V', 'F', 'Q']
class_accuracy_history = {i: [] for i in range(5)}

def plot_ecg_sample(dataset, idx):
    x, y = dataset[idx]
    plt.figure()
    plt.plot(x.numpy())
    plt.title(f"ECG Sample - Class {categories[y]}")
    plt.xlabel("Time Step")
    plt.ylabel("Normalized Amplitude")
    plt.show()

plot_ecg_sample(train_data, idx=82761)



train_dataloader = DataLoader(train_data, batch_size=64, shuffle=True)
test_dataloader = DataLoader(test_data, batch_size=64)

# class Net(nn.Module):
    # def __init__(self):
        # super(Net, self).__init__()
        # Old Structure
        # self.conv1 = nn.Conv1d(1, 16, kernel_size=5)
        # self.conv2 = nn.Conv1d(16, 32, kernel_size=5)
        # self.pool = nn.MaxPool1d(2)

        # self.fc1 = None
        # self.fc2 = nn.Linear(64, 5)  # 5 classes: N,S,V,F,Q

    # def forward(self, x):
        # Old Structure
        # x = x.unsqueeze(1)  # (batch, 1, 188)

        # x = self.pool(F.relu(self.conv1(x)))
        # x = self.pool(F.relu(self.conv2(x)))

        # x = torch.flatten(x, 1)

        # if self.fc1 is None: #Set fc1 based on flattened shape
        #     self.fc1 = nn.Linear(x.shape[1], 64).to(x.device)

        # x = F.relu(self.fc1(x))
        # return self.fc2(x)

class BasicBlock1D(nn.Module):
    def __init__(self, in_channels, out_channels, stride=1):
        super().__init__()

        self.conv1 = nn.Conv1d(in_channels, out_channels, kernel_size=3, stride=stride, padding=1)
        self.bn1 = nn.BatchNorm1d(out_channels)

        self.conv2 = nn.Conv1d(out_channels, out_channels, kernel_size=3, padding=1)
        self.bn2 = nn.BatchNorm1d(out_channels)

        self.shortcut = nn.Sequential()

        # If dimensions change, adjust shortcut
        if stride != 1 or in_channels != out_channels:
            self.shortcut = nn.Sequential(
                nn.Conv1d(in_channels, out_channels, kernel_size=1, stride=stride),
                nn.BatchNorm1d(out_channels)
            )

    def forward(self, x):
        out = F.relu(self.bn1(self.conv1(x)))
        out = self.bn2(self.conv2(out))
        out += self.shortcut(x)
        return F.relu(out)
    
def train_loop(dataloader, model, loss_fn, optimizer):
    size = len(dataloader.dataset)
    for batch, (X, y) in enumerate(dataloader):
        # Compute prediction and loss
        pred = model(X)
        loss = loss_fn(pred, y)

        # Backpropagation
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        if batch % 100 == 0:
            loss, current = loss.item(), (batch + 1) * len(X)
            print(f"loss: {loss:>7f}  [{current:>5d}/{size:>5d}]")

class ResNet1D(nn.Module):
    def __init__(self, num_classes=5):
        super().__init__()

        self.in_channels = 64

        self.conv1 = nn.Conv1d(1, 64, kernel_size=5, stride=1, padding=2)
        self.bn1 = nn.BatchNorm1d(64)
        self.pool = nn.MaxPool1d(2)

        self.layer1 = self._make_layer(64, 2, stride=1)
        self.layer2 = self._make_layer(128, 2, stride=2)
        self.layer3 = self._make_layer(64, 2, stride=2)
        # self.layer4 = self._make_layer(64, 2, stride=2)

        self.global_pool = nn.AdaptiveAvgPool1d(1)
        self.fc = nn.Linear(64, num_classes)
        self.dropout = nn.Dropout(0.4)


    def _make_layer(self, out_channels, blocks, stride):
        layers = []
        layers.append(BasicBlock1D(self.in_channels, out_channels, stride))
        self.in_channels = out_channels

        for _ in range(1, blocks):
            layers.append(BasicBlock1D(out_channels, out_channels))

        return nn.Sequential(*layers)

    def forward(self, x):
        x = x.unsqueeze(1)  # (batch, 1, 188)

        x = F.relu(self.bn1(self.conv1(x)))
        x = self.pool(x)

        x = self.layer1(x)
        x = self.layer2(x)
        x = self.layer3(x)
        # x = self.layer4(x)

        x = self.global_pool(x)   
        x = torch.flatten(x, 1)   
        x = self.dropout(x)

        return self.fc(x)

def test_loop(dataloader, model, loss_fn):
    
    model.eval()
    size = len(dataloader.dataset)
    num_batches = len(dataloader)
    test_loss, correct = 0, 0

    num_classes = 5
    correct_per_class = torch.zeros(num_classes)
    total_per_class = torch.zeros(num_classes)
    
    with torch.no_grad():
        for X, y in dataloader:
            # Total Accuracy Check
            # pred = model(X)
            # test_loss += loss_fn(pred, y).item()
            # correct += (pred.argmax(1) == y).type(torch.float).sum().item()

            pred = model(X)
            test_loss += loss_fn(pred, y).item()
            preds = pred.argmax(1)
            correct += (preds == y).sum().item()

            # Per-class Accuracy
            for i in range(len(y)):
                label = y[i]
                total_per_class[label] += 1
                if preds[i] == label:
                    correct_per_class[label] += 1

    test_loss /= num_batches
    overall_correct = correct / size
    print(f"Test Error: \n Accuracy: {(100*overall_correct):0.2f}%, Avg loss: {test_loss:.8f} \n")

    #Per-class Accuracy Print
    for i in range(num_classes):
        if total_per_class[i] > 0:
            acc = correct_per_class[i] / total_per_class[i]
            print(f"Class {categories[i]} Accuracy: {100*acc:.2f}%")
        else:
            print(f"Class {categories[i]} Accuracy: N/A")

    print("\n-------------------------------\n")

    for i in range(num_classes):
        if total_per_class[i] > 0:
            acc = correct_per_class[i] / total_per_class[i]
            class_accuracy_history[i].append(acc.item())
        else:
            class_accuracy_history[i].append(0)
    
    
def plot_class_accuracy():
        plt.figure()

        for i in range(5):
            plt.plot(class_accuracy_history[i], label=categories[i])

        plt.xlabel("Epoch")
        plt.ylabel("Accuracy")
        plt.title("Per-Class Accuracy Over Epochs")
        plt.legend()
        plt.grid()
        plt.show()

feature_maps = {}

def get_activation(name):
    def hook(model, input, output):
        feature_maps[name] = output.detach()
    return hook



model = ResNet1D(num_classes=5)

batch_size = 64
train_dataloader = DataLoader(train_data, batch_size=batch_size, shuffle=True)
test_dataloader = DataLoader(test_data, batch_size=batch_size)

loss_fn = nn.CrossEntropyLoss(weight=class_weights) # used for categorization (updated to add weights)
learning_rate = 5e-6
# note: optimizer is Adam: one of the best optimizers to date
# it can infer learning rate and all hyper-parameters automatically
optimizer = optim.Adam(model.parameters(), lr=learning_rate)

epochs = 30
for t in range(epochs):
    epoch_timer = time.time()

    print(f"Epoch {t+1}\n-------------------------------")
    train_loop(train_dataloader, model, loss_fn, optimizer)
    test_loop(test_dataloader, model, loss_fn)

    elapsed = time.time() - epoch_timer
    print(f"Epoch Time: {elapsed:.2f} seconds\n")

print("Done!")

runtime = time.time() - total_time
print(f"Total runtime: {runtime:.2f} seconds\n")

plot_class_accuracy()