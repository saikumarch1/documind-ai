from langchain_ollama import OllamaLLM

llm = OllamaLLM(model="llama3.2")

response = llm.invoke("What is Retrieval-Augmented Generation in AI? Answer in 3 sentences.")

print(response)