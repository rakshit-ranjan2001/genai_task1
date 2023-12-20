import dotenv
import openai
import pinecone
import os

# Activating enviroment and setting keys
_ = dotenv.load_dotenv()
pinecone_key = os.getenv("Pinecone_key")
pinecone_env = "gcp-starter"
pinecone.init(api_key=pinecone_key, environment=pinecone_env)
index = pinecone.Index("practice-index")


def chat_completion(messages):
    response = openai.chat.completions.create(model="gpt-3.5-turbo", messages=messages)
    return response.choices[0].message.content


def collect_messages(prompt: str):
    prompt_em = openai.embeddings.create(model="text-embedding-ada-002", input=prompt)
    prompt_em = prompt_em.data[0].embedding
    top_k = 3
    result = index.query(prompt_em, top_k=top_k, includeMetadata=True)
    context = ""
    for i in range(top_k):
        context += result["matches"][i]["metadata"]["content"]
    messages = [
        {
            "role": "system",
            "content": f"You are a gardening assistant bot, an automated service that answers the \
        queries of the people interested in gardening. Greet the user first in a friendly manner \
        and then answer the question as truthfully as possible using the provided context, and if \
        the answer is not contained within the text and requires some latest information to be \
        updated, print 'Sorry not sufficient context to answer query' \
        The context is as follows:\
        {context}",
        }
    ]
    messages.append({"role": "user", "content": prompt + "\n"})
    response = chat_completion(messages)
    messages.append({"role": "assistant", "content": f"{response}"})

    return response


if __name__ == "__main__":
    while True:
        messages = []
        prompt = input("User:")
        if prompt == "exit":
            print("Assistant: Thank you for gardening with us.")
            break
        res = collect_messages(prompt)
        print("Assistant: ", res, "\n\n")
