
from openai import OpenAI
client = OpenAI(
    # This is the default and can be omitted
    api_key="xxx",
)

response = client.files.create(
  file=open("train_stock_data.jsonl", "rb"),
  purpose="fine-tune"
)
print(response)
