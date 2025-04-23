import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import sklearn
from sklearn.preprocessing import LabelEncoder, StandardScaler
import torch
import torch.utils.data
import torch.nn as nn
import torch.nn.functional as F
from torch.autograd import Variable
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_squared_error
from math import sqrt

df = pd.read_csv('cyberdata.csv')
target_column = 'Incident Resolution Time (in Hours)'
categorical_cols = df.select_dtypes(include=['object', 'category']).columns.tolist()
numerical_cols = df.select_dtypes(include=['int64', 'float64']).columns.tolist()
if target_column in numerical_cols:
    numerical_cols.remove(target_column)
encoders = {}
for col in categorical_cols:
    encoders[col] = LabelEncoder()
    df[col] = encoders[col].fit_transform(df[col].astype(str))
X = df.drop(columns=[target_column]).values
y = df[target_column].values.reshape(-1, 1)
X_scaler = StandardScaler()
X = X_scaler.fit_transform(X)
y_scaler = StandardScaler()
y = y_scaler.fit_transform(y)
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=30)

#Define model
class ANN(nn.Module):
    def __init__(self, input_dim):
        super(ANN, self).__init__()
        self.fc1 = nn.Linear(input_dim, 64)
        self.fc2 = nn.Linear(64, 64)
        self.fc3 = nn.Linear(64, 32)
        self.fc4 = nn.Linear(32, 32)
        self.output_layer = nn.Linear(32, 1)
        self.dropout = nn.Dropout(0.15)

    def forward(self, x):
        x = F.relu(self.fc1(x))
        x = F.relu(self.fc2(x))
        x = self.dropout(x)
        x = F.relu(self.fc3(x))
        x = F.relu(self.fc4(x))
        x = self.output_layer(x)
        return x

# Set up model
input_dim = X_train.shape[1]
model = ANN(input_dim=input_dim)

X_train = torch.from_numpy(X_train).float()
y_train = torch.from_numpy(y_train).float()
X_test = torch.from_numpy(X_test).float()
y_test = torch.from_numpy(y_test).float()

train = torch.utils.data.TensorDataset(X_train, y_train)
test = torch.utils.data.TensorDataset(X_test, y_test)
train_loader = torch.utils.data.DataLoader(train, batch_size=32, shuffle=True)
test_loader = torch.utils.data.DataLoader(test, batch_size=32, shuffle=False)

loss_fn = nn.MSELoss()
optimizer = torch.optim.Adam(model.parameters(), lr=0.001, weight_decay=1e-5)

# Training
epochs = 500 #change for accuracy
epoch_list = []
train_loss_list = []
val_loss_list = []
for epoch in range(epochs):
    model.train()
    train_loss = 0.0
    for data, target in train_loader:
        optimizer.zero_grad()
        output = model(data)
        loss = loss_fn(output, target)
        loss.backward()
        optimizer.step()
        train_loss += loss.item() * data.size(0)
    train_loss = train_loss / len(train_loader.dataset)
    train_loss_list.append(train_loss)
    epoch_list.append(epoch + 1)
    # Validation every 50 epochs
    if (epoch + 1) % 50 == 0 or epoch == 0:
        model.eval()
        val_loss = 0.0
        with torch.no_grad():
            for data, target in test_loader:
                output = model(data)
                loss = loss_fn(output, target)
                val_loss += loss.item() * data.size(0)
            val_loss = val_loss / len(test_loader.dataset)
            val_loss_list.append(val_loss)
            print(f'Epoch: {epoch+1}/{epochs} | Training Loss: {train_loss:.6f} | Validation Loss: {val_loss:.6f}')

# Final evaluation
model.eval()
predictions = []
actuals = []
with torch.no_grad():
    for data, target in test_loader:
        output = model(data)
        predictions.append(output.numpy())
        actuals.append(target.numpy())

predictions = np.vstack(predictions)
actuals = np.vstack(actuals)
predictions_orig = y_scaler.inverse_transform(predictions)
actuals_orig = y_scaler.inverse_transform(actuals)

mse = mean_squared_error(actuals_orig, predictions_orig)
rmse = sqrt(mse)
mae = np.mean(np.abs(actuals_orig - predictions_orig))
print("\nModel Performance:")
print(f"Mean Squared Error: {mse:.4f}")
print(f"Root Mean Squared Error: {rmse:.4f} hours")
print(f"Mean Absolute Error: {mae:.4f} hours")

#view accuracy
'''
plt.figure(figsize=(10, 6))
plt.scatter(actuals_orig, predictions_orig, alpha=0.5)
plt.plot([min(actuals_orig), max(actuals_orig)], [min(actuals_orig), max(actuals_orig)], 'r--')
plt.xlabel('Actual Resolution Time (hours)')
plt.ylabel('Predicted Resolution Time (hours)')
plt.title('Actual vs Predicted Incident Resolution Time')
plt.grid(True)
plt.savefig('predictions_vs_actual.png')
plt.show()
'''

print("\nFinished\n")
