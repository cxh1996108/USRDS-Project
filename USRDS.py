#!/usr/bin/env python
# coding: utf-8

# Setting Up the Enivronment
import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
import matplotlib.pyplot as plt
from sklearn import preprocessing
import torch.utils.data as Data
from sklearn import metrics
import seaborn as sns

# Whether GPU is available
print('\nCUDA availability:  %s\n' % (torch.cuda.is_available()))
device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')

# Data Preprocess
data1 = pd.read_csv('usrdscohort_20210520.csv', nrows = 1000000)
# data1 = pd.read_csv('usrdscohort_20210520.csv', nrows = 100, index_col = 'USRDS patient ID')

# Yes to 1, No to 0
data1['Medicare Indicator (Y/N)'] = pd.Series(np.where(data1['Medicare Indicator (Y/N)'].values == 'Yes', 1, 0))
data1['Medicare/Medicaid Dual Eligibility (Y/N)'] = pd.Series(np.where(data1['Medicare/Medicaid Dual Eligibility (Y/N)'].values == 'Yes', 1, 0))

# 1 means Alive, 0 means Death
data1['Alive'] = pd.Series(np.where(data1['Date of Death'].isnull(), 1, 0))

# OneHot encodes Treatment modality (training recoded)
'''
treatment = data1['Treatment modality (training recoded)']
le = preprocessing.LabelEncoder()
labels = le.fit_transform(treatment)
labels.reshape(-1, 1)
enc = preprocessing.OneHotEncoder()
treatrment_onehot = enc.fit_transform(labels.reshape(-1, 1))  
# save OneHot as a tuple in one column
data1['treatment'] = tuple(treatrment_onehot.toarray())
'''
data1 = data1.join(pd.get_dummies(data1['Treatment modality (training recoded)']))
data1.to_csv('data1')
print("data1 success")

# If there is more than n = 12 entries for one patient, use the last n = 12 entries.
# If there is less than n = 12 entries for one patient, use none of them.
n = 12
data2 = pd.DataFrame()
for t in data1['USRDS patient ID'].unique():
    if data1[data1['USRDS patient ID'] == t].shape[0] >= n:
        temp = data1[data1['USRDS patient ID'] == t].tail(n)
        data2 = data2.append(temp)   
data2.to_csv('data2.csv')

num = len(data2['USRDS patient ID'].unique()) # num of patients

labels = [
    data2.Alive.values[i * n]
    for i in range(num)
]

data2 = data2.iloc[:, 9:].drop('Treatment modality (training recoded)', axis = 1).drop('IP_Encounters', axis = 1).drop('Alive', axis = 1).astype('float32')

features = [
    data2.iloc[i * n: i * n + n].values.tolist()
    for i in range(num)
]

print("Num of patients is %d" % len(features)) # Should be equal to num

# Create dataset
labels = torch.tensor(labels)
features = torch.tensor(features)
dataset = torch.utils.data.TensorDataset(features, labels)

features_num = features.shape[2]
print("Num of features is %d" % features_num) # Dynamic


# Split dateset into training set and test set
train_size = int(len(dataset) * 0.7)
test_size = len(dataset) - train_size
trainset, testset = torch.utils.data.random_split(dataset, [train_size, test_size])

# Hyperparameters
batch_size = 16
epochs = 10

# Load datasets
# Add drop_last=True to solve Expected hidden[0] size (a, c, b), got [a, d, b]
train_loader = torch.utils.data.DataLoader(dataset = trainset, batch_size = batch_size, shuffle = True)
test_loader = torch.utils.data.DataLoader(dataset = testset, batch_size = batch_size, shuffle = True)


'''''
# View train_loader
for i, (x, y) in enumerate(train_loader):
    # print(y)  

for i, (x, y) in enumerate(test_loader):
    # print(y.view(len(y), -1).shape)  
    # print(y)  
'''

class RNN_Model(nn.Module):
    def __init__(self, input_dim, hidden_dim, layer_dim, output_dim):
        super(RNN_Model, self).__init__()
        self.hidden_dim = hidden_dim
        self.layer_dim = layer_dim
        # self.rnn = nn.RNN(input_dim, hidden_dim, layer_dim, batch_first = True, nonlinearity = 'relu')
        # self.lstm = nn.LSTM(input_dim, hidden_dim, layer_dim, batch_first = True)
        self.lstm = nn.LSTM(input_dim, hidden_dim, layer_dim)
        self.gru = nn.GRU(input_dim, hidden_dim, layer_dim)
        self.classifier = nn.Linear(hidden_dim, output_dim)
    def forward(self, x):
        # h0 = torch.zeros(self.layer_dim, x.size(0), self.hidden_dim).requires_grad_().to(device)
        # print(h0.shape)
        # out, hn = self.lstm(x, h0.detach())
        out, hn = self.lstm(x, None) # Solve Expected hidden[0] size (2, 20, 4), got [20, 4]
        # out, hn = self.gru(x, None)
        out = self.classifier(out[:, -1, :])
        # out = F.softmax(out, dim = 1)

        return out

# Model Initialization
input_dim = features_num
hidden_dim = 4
layer_dim = 2
output_dim = 2

model = RNN_Model(input_dim, hidden_dim, layer_dim, output_dim).to(device)

# Parameters
learning_rate = 0.001
criterion = nn.CrossEntropyLoss()
optimizer = torch.optim.Adam(model.parameters(), lr = learning_rate)

length = len(list(model.parameters()))
'''
# Model Parameters
for i in range(length):
    print('parameter: %d' % (i+1))
    print(list(model.parameters())[i].size())
'''

# Train

iter = 0
for epoch in range(epochs):
    for i, (x, y) in enumerate(train_loader):
        model.train()
        # print(x.shape)
        # x = x.view(len(x), 1, -1).requires_grad_().to(device)
        x.requires_grad_().to(device)
        # print(x.shape)
        # print(y.shape)
        # y = y.view(len(y), -1).to(device) # Dimension out of range (expected to be in range of [-1, 0], but got 1)
        # print(y.shape)
        optimizer.zero_grad()
        y = y.long().to(device) # expected scalar type Long but found Float
        # print(x.shape)
        outputs = model(x)
        # print(outputs)
        loss = criterion(outputs, y)
        loss.backward()
        optimizer.step()
        iter += 1   
        if iter % 100 == 0: 
            print('Iteration: {}, Loss: {:.5f}'.format(iter, loss.item()))
            
# Test
def test(dataloader, model, criterion):
    Y_valid = []
    Y_pred = []
    Y_score = np.empty(shape=[0, 2])
    size = len(dataloader.dataset)
    model.eval()
    test_loss, correct = 0, 0
    with torch.no_grad():
        for x, y in dataloader:
            # print(x.shape)
            # x = x.view(len(x), 1, -1).to(device)
            # print(x.shape)
            y = y.long().to(device)
            # print(y.shape)
            outputs = model(x)
            logits = F.softmax(outputs, dim = 1)
            # print(np.array(logits).shape)
            # print(y)
            # print("###")
            
            Y_valid = np.append(Y_valid, y.numpy())
            Y_score = np.concatenate((Y_score, logits), axis = 0)
            Y_pred = np.append(Y_pred, np.amax(outputs.numpy(), 1))
            # Y_pred = np.append(Y_pred, outputs.argmax(dim = 1).numpy())
            # Y_pred = np.append(Y_pred, outputs.max(1, keepdim=True) [1])
            # Y_pred = torch.max(ouputs, dim = 1)
            # Y_pred = outputs.argmax(dim = 1)
            
            # print(np.amax(outputs.numpy(), 1))
            test_loss += criterion(outputs, y).item()
            correct += (outputs.argmax(1) == y).type(torch.float).sum().item()
    
    test_loss /= batch_size
    correct /= size
    print(f"Test Error: \n Accuracy: {(100 * correct): > 0.1f}%, Avg loss: {test_loss:>8f} \n")
    return Y_valid, Y_pred, Y_score

Y_valid, Y_pred, Y_score = test(test_loader, model, criterion)

# Save model
torch.save(model.state_dict(), "model.pth")
print("Saved PyTorch Model State to model.pth")



# Plot ROC curve
fpr, tpr, thresholds = metrics.roc_curve(Y_valid, Y_score[:, 1]) # , pos_label = 1 , drop_intermediate = False
# print("fpr:{},tpr:{},thresholds:{}".format(fpr,tpr,thresholds))
roc_auc = metrics.auc(fpr, tpr)
print("AUC = ", roc_auc)
plt.figure(1, figsize=(10,6))
plt.plot(fpr, tpr)

plt.xlim([0.00, 1.0])
plt.ylim([0.00, 1.0])
plt.xlabel("False Positive Rate")
plt.ylabel("True Positive Rate")
plt.title("ROC")
# plt.legend(loc="lower right")
plt.savefig(r"./ROC.png")



# Plot PR curve
precision, recall, thresholds = metrics.precision_recall_curve(Y_valid, Y_score[:, 1])
pr = metrics.auc(recall, precision)
print("AUPR = ", pr)
plt.figure(2, figsize=(10,6))
plt.plot(recall, precision)

plt.xlim([0.00, 1.0])
plt.ylim([0.00, 1.0])
plt.xlabel("Recall")
plt.ylabel("Precision")
plt.title("PR")
# plt.legend(loc="lower right")
plt.savefig(r"./PR.png")



# Boxplot
dfp = pd.DataFrame([Y_valid, Y_pred]).T
dfp.columns = ['Y_valid', 'Y_pred']
dfp1 = dfp[dfp['Y_valid'] == 1] # Alive
print("Num of Alive: %d" % dfp1.shape[0])
dfp2 = dfp[dfp['Y_valid'] == 0] # Dead
print("Num of Dead: %d" % dfp2.shape[0])
print("Total patient: %d" % (dfp1.shape[0] + dfp2.shape[0]))
ddf = pd.DataFrame([dfp1['Y_pred'].values, dfp2['Y_pred'].values])
ddf = ddf.T
ddf.columns = ['Alive', 'Dead']
plt.figure(figsize=(10,6))
sns.boxplot(data = ddf)


