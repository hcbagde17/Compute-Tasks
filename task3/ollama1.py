import ollama

client = ollama.Client()
model = "tinyllama"
while True:
    prompt = input("Enter your prompt (or type 'q' to quit): ")
    if prompt.lower() == 'q':
        break
    response = client.generate(model=model, prompt=prompt)
    print(response.response)