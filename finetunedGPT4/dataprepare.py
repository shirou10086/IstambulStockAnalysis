import pandas as pd
import json

# 读取 CSV 数据
data = pd.read_csv('../data/FormedISE.csv')

data['date'] = pd.to_datetime(data['date'], format='%d-%b-%y')
data.set_index('date', inplace=True)

# 分割训练和测试数据
train_end_date = '31-Dec-09'
train_data = data[:train_end_date]
test_data = data[train_end_date:]

# 创建JSONL格式的数据
jsonl_data = []
for i in range(len(train_data) - 1):
    messages = [
        {"role": "system", "content": "Stock prediction assistant."},
        {"role": "user", "content": f"Stock data on {train_data.index[i].strftime('%d-%b-%y')}: ISE: {train_data.iloc[i]['ISE']}, SP: {train_data.iloc[i]['SP']}, DAX: {train_data.iloc[i]['DAX']}, FTSE: {train_data.iloc[i]['FTSE']}, NIKKEI: {train_data.iloc[i]['NIKKEI']}, BOVESPA: {train_data.iloc[i]['BOVESPA']}, EU: {train_data.iloc[i]['EU']}, EM: {train_data.iloc[i]['EM']}. What is the next day's ISE?"},
        {"role": "assistant", "content": f"{train_data.iloc[i+1]['ISE']}"}
    ]
    jsonl_data.append({'messages': messages})

# 保存为JSONL文件
with open('train_stock_data.jsonl', 'w') as outfile:
    for entry in jsonl_data:
        json.dump(entry, outfile)
        outfile.write('\n')
