import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score
from sklearn.preprocessing import StandardScaler
from openai import OpenAI
import torch
import torch.nn as nn
from torch.utils.data import DataLoader, TensorDataset

# 读取CSV数据
data = pd.read_csv('../data/FormedISE.csv')

# 将日期列转换为日期格式并设置为索引
data['date'] = pd.to_datetime(data['date'], format='%d-%b-%y')
data.set_index('date', inplace=True)

# 分割训练和测试数据
train_end_date = '31-Dec-09'
train_data = data[:train_end_date]
test_data = data[train_end_date:]

# 标准化数据
scaler = StandardScaler()
data_scaled = scaler.fit_transform(data)

# 创建序列数据集
def create_sequences(data, seq_length=5):
    xs, ys = [], []
    for i in range(len(data) - seq_length):
        x = data[i:i+seq_length]
        y = data[i+seq_length, 0]  # 目标变量是ISE
        xs.append(x)
        ys.append(y)
    return np.array(xs), np.array(ys)

seq_length = 5
X, y = create_sequences(data_scaled, seq_length)

# 分割训练和测试数据
X_train, y_train = X[:len(train_data)-seq_length], y[:len(train_data)-seq_length]
X_test, y_test = X[len(train_data)-seq_length:], y[len(train_data)-seq_length:]

# 创建数据加载器
train_dataset = TensorDataset(torch.from_numpy(X_train).float(), torch.from_numpy(y_train).float())
test_dataset = TensorDataset(torch.from_numpy(X_test).float(), torch.from_numpy(y_test).float())
train_loader = DataLoader(train_dataset, batch_size=16, shuffle=True)
test_loader = DataLoader(test_dataset, batch_size=16, shuffle=False)

# 定义模型
class StockPredictor(nn.Module):
    def __init__(self, input_dim, hidden_dim, output_dim, num_layers):
        super(StockPredictor, self).__init__()
        self.lstm = nn.LSTM(input_dim, hidden_dim, num_layers, batch_first=True)
        self.fc = nn.Linear(hidden_dim, output_dim)

    def forward(self, x):
        h_0 = torch.zeros(2, x.size(0), 50).to(x.device)
        c_0 = torch.zeros(2, x.size(0), 50).to(x.device)
        _, (h_n, _) = self.lstm(x, (h_0, c_0))
        out = self.fc(h_n[-1])
        return out

# 设置模型参数
input_dim = X_train.shape[2]
hidden_dim = 50
output_dim = 1
num_layers = 2
num_epochs = 100
learning_rate = 0.001

# 初始化模型、损失函数和优化器
model = StockPredictor(input_dim, hidden_dim, output_dim, num_layers)
criterion = nn.MSELoss()
optimizer = torch.optim.Adam(model.parameters(), lr=learning_rate)

# 训练模型
for epoch in range(num_epochs):
    model.train()
    for X_batch, y_batch in train_loader:
        optimizer.zero_grad()
        y_pred = model(X_batch)
        loss = criterion(y_pred, y_batch.unsqueeze(1))
        loss.backward()
        optimizer.step()
    print(f'Epoch {epoch+1}/{num_epochs}, Loss: {loss.item()}')

# 测试模型
model.eval()
predicted_ises = []
with torch.no_grad():
    for X_batch, _ in test_loader:
        y_pred = model(X_batch)
        predicted_ises.extend(y_pred.squeeze().cpu().numpy())

# 创建一个与原始数据形状相同的数组，并将预测的ISE列插入其中
full_predicted_data = np.zeros((len(predicted_ises), data_scaled.shape[1]))
full_predicted_data[:, 0] = predicted_ises

# 反标准化
predicted_ises = scaler.inverse_transform(full_predicted_data)[:, 0]
actual_ises = scaler.inverse_transform(np.concatenate([y_test.reshape(-1, 1), np.zeros((len(y_test), data_scaled.shape[1] - 1))], axis=1))[:, 0]

# 确保日期长度匹配
dates = test_data.index[seq_length:seq_length+len(actual_ises)]

# 如果实际值或预测值长度大于日期长度，进行切割
if len(actual_ises) > len(dates):
    actual_ises = actual_ises[:len(dates)]
if len(predicted_ises) > len(dates):
    predicted_ises = predicted_ises[:len(dates)]

# 评估函数
def evaluation_metric(y_test, y_hat):
    mse = mean_squared_error(y_test, y_hat)
    rmse = np.sqrt(mse)
    mae = mean_absolute_error(y_test, y_hat)
    r2 = r2_score(y_test, y_hat)
    return mse, rmse, mae, r2

# 评估
metrics = evaluation_metric(actual_ises, predicted_ises)
print(f'MSE: {metrics[0]}, RMSE: {metrics[1]}, MAE: {metrics[2]}, R2: {metrics[3]}')

# 可视化
plt.figure(figsize=(10, 6))
plt.plot(dates, actual_ises, label='Actual ISE')
plt.plot(dates, predicted_ises, label='Predicted ISE')
plt.title('Stock Price Prediction')
plt.xlabel('Time')
plt.ylabel('ISE')
plt.legend()
plt.show()
