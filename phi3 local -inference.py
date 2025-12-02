from ollama import Client
client = Client()

prompt = """<|user|>
Top 5 products by total ordered quantity.
<|end|>
<|assistant|>
"""

response = client.chat(model="phi3-local", messages=[
    {"role": "user", "content": prompt}
])

print(response["message"]["content"])

