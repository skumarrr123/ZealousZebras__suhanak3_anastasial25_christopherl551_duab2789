import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import sklearn

#Pt2
import torch
import torch.utils.data
import torch.nn as nn
import torch.nn.functional as F
from torch.autograd import Variable

from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_squared_error
from math import sqrt

df = pd.read_csv('data.csv') #Change
print(df.shape)
df.describe()

target_column = ['approval_status'] #Change
predictors = list(set(list(df.columns))-set(target_column))

print(target_column)
print(predictors)

X = df[predictors].values
y = df[target_column].values

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size = 0.3, random_state = 30)
print(X_train.shape); print(X_test.shape)

class ANN(nn.Module):
    def __init__(self, input_dim = 7, output_dim = 1):
        super(ANN, self).__init__()
        self.fc1 = nn.Linear(input_dim, 64)
        self.fc2 = nn.Linear(64, 64)
        self.fc3 = nn.Linear(64, 32)
        self.fc4 = nn.Linear(32, 32)
        self.output_layer = nn.Linear(32,1)
        self.dropout = nn.Dropout(0.15)

     def forward(self, x):
        x = F.relu(self.fc1(x))
        x = F.relu(self.fc2(x))
        x = self.dropout(x)
        x = F.relu(self.fc3(x))
        x = F.relu(self.fc4(x))
        x = self.output_layer(x)

        return nn.Sigmoid()(x)

model = ANN(input_dim = 7, output_dim = 1)

print(model)
