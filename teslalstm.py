import pandas as pd
import numpy as np
from sklearn.preprocessing import MinMaxScaler
from keras.models import Sequential
from keras.layers import LSTM, Dense, Dropout
import matplotlib.pyplot as plt

# 读取和清理数据
data = pd.read_csv('Tesla.csv')
data['Date'] = pd.to_datetime(data['Date'], format='%m/%d/%Y')
data.set_index('Date', inplace=True)

# 选择要预测的列（Adj Close）
dataset = data['Adj Close'].values.reshape(-1, 1)

# 数据规范化
scaler = MinMaxScaler(feature_range=(-1, 1))
scaled_data = scaler.fit_transform(dataset)

# 创建数据集
def create_dataset(data, look_back=1):
    X, Y = [], []
    for i in range(len(data)-look_back-1):
        a = data[i:(i+look_back), 0]
        X.append(a)
        Y.append(data[i + look_back, 0])
    return np.array(X), np.array(Y)

look_back = 10

# 查找训练和测试数据的分割点
train_end_date = '2016-12-31'
test_start_date = '2017-01-01'
test_end_date = '2017-03-17'

# 使用最接近的日期
train_end_index = data.index.get_indexer([pd.to_datetime(train_end_date)], method='nearest')[0]
test_start_index = data.index.get_indexer([pd.to_datetime(test_start_date)], method='nearest')[0]
test_end_index = data.index.get_indexer([pd.to_datetime(test_end_date)], method='nearest')[0]

# 分割数据集为训练集和测试集
train_data = scaled_data[:train_end_index + 1]
test_data = scaled_data[test_start_index:test_end_index + 1]

# 打印检查数据集的大小
print("Train data size:", train_data.shape)
print("Test data size:", test_data.shape)

X_train, Y_train = create_dataset(train_data, look_back)
X_test, Y_test = create_dataset(test_data, look_back)

# 打印检查创建的数据集的大小
print("X_train size:", X_train.shape)
print("Y_train size:", Y_train.shape)
print("X_test size:", X_test.shape)
print("Y_test size:", Y_test.shape)

# 确保测试集不为空
if X_test.shape[0] == 0 or X_test.shape[1] == 0:
    raise ValueError("Test dataset is empty. Please check the train/test split dates.")

# 重塑输入为LSTM模型的格式 [samples, time steps, features]
X_train = np.reshape(X_train, (X_train.shape[0], X_train.shape[1], 1))
X_test = np.reshape(X_test, (X_test.shape[0], X_test.shape[1], 1))

# 创建和训练LSTM模型
model = Sequential()
model.add(LSTM(100, return_sequences=True, input_shape=(look_back, 1)))
model.add(Dropout(0.2))
model.add(LSTM(100))
model.add(Dropout(0.2))
model.add(Dense(1))
model.compile(loss='mean_squared_error', optimizer='adam')
model.fit(X_train, Y_train, epochs=50, batch_size=1, verbose=2)

# 做预测
train_predict = model.predict(X_train)
test_predict = model.predict(X_test)

# 反规范化预测值
train_predict = scaler.inverse_transform(train_predict)
Y_train = scaler.inverse_transform([Y_train])
test_predict = scaler.inverse_transform(test_predict)
Y_test = scaler.inverse_transform([Y_test])

# 绘制结果
plt.figure(figsize=(12, 6))
plt.plot(data.index[:len(Y_train[0])], Y_train[0], label='Training Actual Data')
plt.plot(data.index[:len(Y_train[0])], train_predict[:,0], label='Training Predicted Data')
plt.plot(data.index[len(Y_train[0]):len(Y_train[0])+len(Y_test[0])], Y_test[0], label='Testing Actual Data')
plt.plot(data.index[len(Y_train[0]):len(Y_train[0])+len(Y_test[0])], test_predict[:,0], label='Testing Predicted Data')
plt.xlabel('Date')
plt.ylabel('Adj Close')
plt.legend()
plt.show()

# 马尔可夫链
# 定义状态：上涨（1），下跌（0）
states = [1 if val > 0 else 0 for val in dataset.flatten()]

# 构建状态转移矩阵
transition_matrix = np.zeros((2, 2))
for (i, j) in zip(states, states[1:]):
    transition_matrix[i, j] += 1
transition_matrix = transition_matrix / transition_matrix.sum(axis=1, keepdims=True)

print("Transition Matrix:")
print(transition_matrix)

# 使用马尔可夫链进行预测
def markov_predict(current_state, transition_matrix, n_steps):
    predictions = []
    state = current_state
    for _ in range(n_steps):
        next_state = np.random.choice([0, 1], p=transition_matrix[state])
        predictions.append(next_state)
        state = next_state
    return predictions

# 当前状态
current_state = states[-1]

# 预测未来10天的状态
future_states = markov_predict(current_state, transition_matrix, 10)
print("Predicted future states:", future_states)
