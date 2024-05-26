import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score
from sklearn.preprocessing import MinMaxScaler
import torch
import torch.nn as nn
import torch.nn.functional as F
from mamba import Mamba, MambaConfig
import argparse

# 解析命令行参数
parser = argparse.ArgumentParser()
parser.add_argument('--use-cuda', default=False, help='CUDA training.')
parser.add_argument('--seed', type=int, default=1, help='Random seed.')
parser.add_argument('--epochs', type=int, default=100, help='Number of epochs to train.')
parser.add_argument('--lr', type=float, default=0.01, help='Learning rate.')
parser.add_argument('--wd', type=float, default=1e-5, help='Weight decay (L2 loss on parameters).')
parser.add_argument('--hidden', type=int, default=16, help='Dimension of representations')
parser.add_argument('--layer', type=int, default=2, help='Num of layers')
parser.add_argument('--n-test', type=int, default=300, help='Size of test set')
parser.add_argument('--data-file', type=str, default='../data/Tesla.csv', help='Data file')

args = parser.parse_args()
args.cuda = args.use_cuda and torch.cuda.is_available()
def evaluation_metric(y_test, y_hat):
    MSE = mean_squared_error(y_test, y_hat)
    RMSE = np.sqrt(MSE)  # RMSE 是 MSE 的平方根
    MAE = mean_absolute_error(y_test, y_hat)
    R2 = r2_score(y_test, y_hat)
    print(f'MSE: {MSE:.4f}, RMSE: {RMSE:.4f}, MAE: {MAE:.4f}, R2: {R2:.4f}')

# 设置随机种子以确保结果的可重现性
def set_seed(seed, cuda):
    np.random.seed(seed)
    torch.manual_seed(seed)
    if cuda:
        torch.cuda.manual_seed(seed)

set_seed(args.seed, args.cuda)

# 定义模型
class Net(nn.Module):
    def __init__(self, in_dim, out_dim):
        super().__init__()
        self.config = MambaConfig(d_model=args.hidden, n_layers=args.layer)
        self.mamba = nn.Sequential(
            nn.Linear(in_dim, args.hidden),
            Mamba(self.config),
            nn.Linear(args.hidden, out_dim),
            nn.Identity()  # 使用 Identity 以输出涨跌幅
        )

    def forward(self, x):
        x = self.mamba(x)
        return x.flatten()

# 数据预处理
data = pd.read_csv(args.data_file, delimiter=',')
data.columns = data.columns.str.strip()  # 清理列名
data['Date'] = pd.to_datetime(data['Date'], format='%m/%d/%Y')
data.set_index('Date', inplace=True)

# 计算涨跌幅
data['Change'] = data['Close'].pct_change() * 100
data.dropna(inplace=True)  # 移除NaN

target_column = 'Change'
target_data = data[target_column].values

features_data = data.drop(columns=[target_column, 'Adj Close', 'Close']).values
scaler = MinMaxScaler()
features_data = scaler.fit_transform(features_data)

# 拆分数据集
trainX, testX = features_data[:-args.n_test, :], features_data[-args.n_test:, :]
trainy = target_data[:-args.n_test]

# 训练模型
def train_and_predict(trainX, trainy, testX):
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
        if e % 10 == 0:
            print('Epoch %d | Loss: %.4f' % (e, loss.item()))

    clf.eval()
    mat = clf(xv)
    if args.cuda:
        mat = mat.cpu()
    yhat = mat.detach().numpy().flatten()
    return yhat

predictions = train_and_predict(trainX, trainy, testX)

# 评估模型
print('MSE RMSE MAE R2')
evaluation_metric(np.array(target_data[-args.n_test:]), predictions)

# 可视化结果
plt.figure(figsize=(10, 6))
plt.plot(data.index[-args.n_test:], target_data[-args.n_test:], label='Actual Change')
plt.plot(data.index[-args.n_test:], predictions, label='Predicted Change')
plt.title('Stock Price Change Prediction')
plt.xlabel('Time', fontsize=12)
plt.ylabel('Change', fontsize=14)
plt.legend()
plt.show()
