import pandas as pd
import numpy as np
from sklearn.preprocessing import MinMaxScaler
from keras.models import Sequential
from keras.layers import LSTM, Dense, Dropout
import matplotlib.pyplot as plt

# 读取和清理数据
column_names = ["date", "TL_BASED_ISE", "USD_BASED_ISE", "SP", "DAX", "FTSE", "NIKKEI", "BOVESPA", "EU", "EM"]
data = pd.read_csv('data_akbilgic.csv', names=column_names, header=1)
data = data.drop(columns=["SP", "DAX", "FTSE", "NIKKEI", "BOVESPA", "EU", "EM"])
data['date'] = pd.to_datetime(data['date'], format='%d-%b-%y')
data.set_index('date', inplace=True)

# 选择要预测的列（USD_BASED_ISE）
dataset = data['USD_BASED_ISE'].values.reshape(-1, 1)

# 数据规范化
scaler = MinMaxScaler(feature_range=(-1, 1))  # 使用不同的规范化范围
scaled_data = scaler.fit_transform(dataset)

# 创建数据集
def create_dataset(data, look_back=1):
    X, Y = [], []
    for i in range(len(data)-look_back-1):
        a = data[i:(i+look_back), 0]
        X.append(a)
        Y.append(data[i + look_back, 0])
    return np.array(X), np.array(Y)

look_back = 10  # 增加look_back的值

# 查找训练和测试数据的分割点
train_end_date = '2009-12-31'
test_start_date = '2010-01-01'
test_end_date = '2011-03-31'

train_end_index = data.index.get_loc(train_end_date)
test_start_index = data.index.get_indexer([test_start_date], method='nearest')[0]
test_end_index = data.index.get_indexer([test_end_date], method='nearest')[0]

# 分割数据集为训练集和测试集
train_data = scaled_data[:train_end_index + 1]
test_data = scaled_data[test_start_index:test_end_index + 1]

X_train, Y_train = create_dataset(train_data, look_back)
X_test, Y_test = create_dataset(test_data, look_back)

# 重塑输入为LSTM模型的格式 [samples, time steps, features]
X_train = np.reshape(X_train, (X_train.shape[0], X_train.shape[1], 1))
X_test = np.reshape(X_test, (X_test.shape[0], X_test.shape[1], 1))

# 创建和训练LSTM模型
model = Sequential()
model.add(LSTM(100, return_sequences=True, input_shape=(look_back, 1)))  # 增加LSTM层的神经元数量
model.add(Dropout(0.2))  # 增加Dropout层
model.add(LSTM(100))
model.add(Dropout(0.2))
model.add(Dense(1))
model.compile(loss='mean_squared_error', optimizer='adam')
model.fit(X_train, Y_train, epochs=200, batch_size=1, verbose=2)  # 增加训练的epoch数

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
plt.ylabel('USD_BASED_ISE')
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
