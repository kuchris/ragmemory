import os

from openai import OpenAI
from memory import MemoryStore

# MODEL = "meta/llama-3.1-8b-instruct"
# API_KEY_ENV = "NVIDIA_API_KEY"
# DB_PATH = os.environ.get("RAGMEMORY_DB_PATH", "./chroma_structured_test")

# api_key = os.environ.get(API_KEY_ENV)
# if not api_key:
#     raise RuntimeError(f"Set {API_KEY_ENV} before running chat.py.")

# client = OpenAI(
#     base_url="https://integrate.api.nvidia.com/v1",
#     api_key=api_key,
# )

# MODEL = "qwopus3.6-35b-a3b-v1"
MODEL = "gemma-4-e4b-uncensored-hauhaucs-aggressive"
API_KEY_ENV = "LMSTUDIO_API_KEY"
DB_PATH = os.environ.get("RAGMEMORY_DB_PATH", "./chroma_structured_test")

api_key = os.environ.get(API_KEY_ENV, "lm-studio")

client = OpenAI(
    base_url="http://localhost:1234/v1",
    api_key=api_key,
)

memory = MemoryStore(db_path=DB_PATH)


def call_model(context: str, user_message: str) -> str:
    system = "You are a helpful assistant with memory of past conversations."
    if context:
        system += f"\n\n{context}"

    response = client.chat.completions.create(
        model=MODEL,
        messages=[
            {"role": "system", "content": system},
            {"role": "user", "content": user_message},
        ],
        max_tokens=512,
    )
    return response.choices[0].message.content


def chat():
    print(f"RAG Memory Chat  |  model: {MODEL}  |  db: {DB_PATH}  |  type 'quit' to exit\n")
    while True:
        user_input = input("You: ").strip()
        if not user_input or user_input.lower() == "quit":
            break

        context = memory.build_context(user_input)
        response = call_model(context, user_input)

        memory.add_message("user", user_input, extract_structured=False)
        memory.add_message("assistant", response, extract_structured=False)

        print(f"Assistant: {response}\n")


if __name__ == "__main__":
    chat()
