from openai import OpenAI

client = OpenAI(base_url='http://127.0.0.1:5000/v1', api_key='<key>')
model = 'manticoresearch'

test = client.chat.completions.create(
    model=model,
    messages=[{'role': 'user', 'content': 'You are a helpful assistant.'}]
)
print(test)
