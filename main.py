import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score, roc_auc_score, f1_score
from sklearn.preprocessing import Binarizer
import torch
import torch.nn as nn
import torch.nn.functional as F
from mamba import Mamba, MambaConfig
import argparse

parser = argparse.ArgumentParser()
parser.add_argument('--use-cuda', default=False, help='CUDA training.')
parser.add_argument('--seed', type=int, default=1, help='Random seed.')
parser.add_argument('--epochs', type=int, default=100, help='Number of epochs to train.')
parser.add_argument('--lr', type=float, default=0.01, help='Learning rate.')
parser.add_argument('--wd', type=float, default=1e-5, help='Weight decay (L2 loss on parameters).')
parser.add_argument('--hidden', type=int, default=16, help='Dimension of representations')
parser.add_argument('--layer', type=int, default=2, help='Num of layers')
parser.add_argument('--n-test', type=int, default=300, help='Size of test set')
parser.add_argument('--data-file', type=str, default='data_akbilgic.csv', help='Data file')

args = parser.parse_args()
args.cuda = args.use_cuda and torch.cuda.is_available()

def evaluation_metric(y_test, y_hat):
    MSE = mean_squared_error(y_test, y_hat)
    RMSE = MSE**0.5
    MAE = mean_absolute_error(y_test, y_hat)
    R2 = r2_score(y_test, y_hat)
    binarizer = Binarizer(threshold=0.0)
    y_test_bin = binarizer.fit_transform(y_test.reshape(-1, 1)).ravel()
    y_hat_bin = binarizer.transform(np.array(y_hat).reshape(-1, 1)).ravel()
    AUC = roc_auc_score(y_test_bin, y_hat_bin)
    F1 = f1_score(y_test_bin, y_hat_bin)
    print('%.4f %.4f %.4f %.4f %.4f %.4f' % (MSE, RMSE, MAE, R2, AUC, F1))

def set_seed(seed, cuda):
    np.random.seed(seed)
    torch.manual_seed(seed)
    if cuda:
        torch.cuda.manual_seed(seed)

def dateinf(series, n_test):
    lt = len(series)
    print('Training start', series[0])
    print('Training end', series[lt-n_test-1])
    print('Testing start', series[lt-n_test])
    print('Testing end', series[lt-1])

set_seed(args.seed, args.cuda)

class Net(nn.Module):
    def __init__(self, in_dim, out_dim):
        super().__init__()
        self.config = MambaConfig(d_model=args.hidden, n_layers=args.layer)
        self.mamba = nn.Sequential(
            nn.Linear(in_dim, args.hidden),
            Mamba(self.config),
            nn.Linear(args.hidden, out_dim),
            nn.Tanh()
        )

    def forward(self, x):
        x = self.mamba(x)
        return x.flatten()

def PredictWithData(trainX, trainy, testX):
    clf = Net(len(trainX[0]), 1)
    opt = torch.optim.Adam(clf.parameters(), lr=args.lr, weight_decay=args.wd)
    xt = torch.from_numpy(trainX).float().unsqueeze(0)
    xv = torch.from_numpy(testX).float().unsqueeze(0)
    yt = torch.from_numpy(trainy).float()
    if args.cuda:
        clf = clf.cuda()
        xt = xt.cuda()
        xv = xv.cuda()
        yt = yt.cuda()

    for e in range(args.epochs):
        clf.train()
        z = clf(xt)
        loss = F.mse_loss(z, yt)
        opt.zero_grad()
        loss.backward()
        opt.step()
        if e % 10 == 0 and e != 0:
            print('Epoch %d | Loss: %.4f' % (e, loss.item()))

    clf.eval()
    mat = clf(xv)
    if args.cuda: mat = mat.cpu()
    yhat = mat.detach().numpy().flatten()
    return yhat

def build_markov_chain(data, n_states=10):
    data_min, data_max = data.min(), data.max()
    bins = np.linspace(data_min, data_max, n_states+1)
    states = np.digitize(data, bins) - 1
    transition_matrix = np.zeros((n_states, n_states))
    for i in range(len(states) - 1):
        if states[i] >= 0 and states[i] < n_states and states[i+1] >= 0 and states[i+1] < n_states:
            transition_matrix[states[i], states[i+1]] += 1
    transition_matrix /= transition_matrix.sum(axis=1, keepdims=True)
    transition_matrix = np.nan_to_num(transition_matrix)  # 将 NaN 转换为 0
    return transition_matrix, bins

def apply_markov_chain(predictions, bins, transition_matrix):
    n_states = transition_matrix.shape[0]
    states = np.digitize(predictions, bins) - 1
    adjusted_predictions = []
    for state in states:
        if state >= 0 and state < n_states:
            next_state = np.argmax(transition_matrix[state])
            adjusted_prediction = (bins[next_state] + bins[next_state+1]) / 2
        else:
            adjusted_prediction = predictions[state]  # 如果状态超出范围，保持原预测值
        adjusted_predictions.append(adjusted_prediction)
    return np.array(adjusted_predictions)

# 数据预处理
data = pd.read_csv(args.data_file, delimiter=',', skiprows=1)

# 打印列名以检查是否正确读取
print("Columns in the dataset:", data.columns)

# 确保列名中没有多余的空格
data.columns = data.columns.str.strip()

# 再次检查列名
print("Columns after stripping:", data.columns)

# 将日期列转换为日期格式
data['date'] = pd.to_datetime(data['date'], format='%d-%b-%y')
data.set_index('date', inplace=True)

# 选择要预测的目标变量，这里假设是 'ISE' 列
target_column = 'ISE'
target_data = data[target_column].values

# 取出除目标列外的其他特征
features_data = data.drop(columns=[target_column]).values

# 确认其他指数是否包含在特征数据中
print("Features data shape:", features_data.shape)

# 拆分训练集和测试集
trainX, testX = features_data[:-args.n_test, :], features_data[-args.n_test:, :]
trainy = target_data[:-args.n_test]

# 构建马尔可夫链
transition_matrix, bins = build_markov_chain(trainy)

predictions = PredictWithData(trainX, trainy, testX)

# 应用马尔可夫链调整预测结果
adjusted_predictions = apply_markov_chain(predictions, bins, transition_matrix)

# 预测未来的股票价格
time = data.index[-args.n_test:]
actual_values = target_data[-args.n_test:]
finalpredicted_stock_price = []
pred = target_data[-args.n_test-1]
for i in range(args.n_test):
    pred = target_data[-args.n_test-1+i] * (1 + adjusted_predictions[i])
    finalpredicted_stock_price.append(pred)

# 评估与可视化
dateinf(data.index, args.n_test)
print('MSE RMSE MAE R2 AUC F1')
evaluation_metric(np.array(actual_values), np.array(finalpredicted_stock_price))

plt.figure(figsize=(10, 6))
plt.plot(time, actual_values, label='Actual Stock Price')
plt.plot(time, finalpredicted_stock_price, label='Predicted Stock Price')
plt.title('Stock Price Prediction')
plt.xlabel('Time', fontsize=12, verticalalignment='top')
plt.ylabel('Close', fontsize=14, horizontalalignment='center')
plt.legend()
plt.show()
