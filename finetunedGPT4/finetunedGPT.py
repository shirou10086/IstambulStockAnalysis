

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score
from openai import OpenAI

# 读取CSV数据
data = pd.read_csv('../data/FormedISE.csv')

# 将日期列转换为日期格式并设置为索引
data['date'] = pd.to_datetime(data['date'], format='%d-%b-%y')
data.set_index('date', inplace=True)

# 分割训练和测试数据
train_end_date = '31-Dec-09'
test_data = data[train_end_date:]

# 设置API密钥
client = OpenAI(
    # This is my api
    api_key="xxx",
)
# 示例测试数据
test_prompts = []
for i in range(len(test_data) - 1):
    test_prompt = f"Stock data on {test_data.index[i].strftime('%d-%b-%y')}: ISE: {test_data.iloc[i]['ISE']}, SP: {test_data.iloc[i]['SP']}, DAX: {test_data.iloc[i]['DAX']}, FTSE: {test_data.iloc[i]['FTSE']}, NIKKEI: {test_data.iloc[i]['NIKKEI']}, BOVESPA: {test_data.iloc[i]['BOVESPA']}, EU: {test_data.iloc[i]['EU']}, EM: {test_data.iloc[i]['EM']}. What is the next day's ISE?"
    test_prompts.append(test_prompt)

# 存储预测值
predicted_ises = []
for prompt in test_prompts:
    completion = client.chat.completions.create(
        model="ft:gpt-3.5-turbo-1106:personal::9SzrpfV2",
        messages=[
            {"role": "system", "content": "You are a Stock prediction assistant."},
            {"role": "user", "content": prompt}
        ]
    )
    predicted_ise = float(completion.choices[0].message.content.strip())
    predicted_ises.append(predicted_ise)

# 实际值
actual_ises = test_data['ISE'].values[1:]

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
dates = test_data.index[1:]
plt.plot(dates, actual_ises, label='Actual ISE')
plt.plot(dates, predicted_ises, label='Predicted ISE')
plt.title('Stock Price Prediction')
plt.xlabel('Time')
plt.ylabel('ISE')
plt.legend()
plt.show()
