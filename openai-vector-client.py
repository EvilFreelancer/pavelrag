from openai import OpenAI

client = OpenAI(base_url='http://127.0.0.1:5000/v1', api_key='<key>')
model = 'ManticoreSearchDocs:GraphRAG'

test = client.chat.completions.create(
    model=model,
    messages=[{'role': 'user', 'content': 'KNN vector search via HTTP API'}]
)
print(test)
