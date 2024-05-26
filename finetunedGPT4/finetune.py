import openai
from openai import OpenAI
client = OpenAI(
    # This is the default and can be omitted
    api_key="xxx",
)


response=client.fine_tuning.jobs.create(
  training_file="file-7exhrz8r1lWIap5CprJpTw1y",
  model="gpt-3.5-turbo"
)

print(response)
client.fine_tuning.jobs.list(limit=5)
