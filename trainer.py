import os
import dotenv
import pandas as pd
import openai
import pinecone

# Activating DBs and getting enviroment keys
_ = dotenv.load_dotenv()
pinecone_key = os.getenv("Pinecone_key")
pinecone_env = "gcp-starter"
pinecone.init(api_key=pinecone_key, environment=pinecone_env)
index = pinecone.Index("practice-index")

# Reading document
df = pd.read_csv("gardening_dataset.csv")
df = df.drop_duplicates(subset="text", keep="first")
print(f"Dataframe created, size: {len(df)}")

# Upserting Data
upserted_data = []
for i, item in enumerate(df["text"].to_list()):
    vector = openai.embeddings.create(model="text-embedding-ada-002", input=item)
    upserted_data.append((str(i), vector.data[0].embedding, {"content": item}))
    print(f"Created vector {i}")
    if i % 100 == 0:
        index.upsert(vectors=upserted_data)
        print(f"Upserted vectors {i-100} to {i}")
        upserted_data = []
index.upsert(vectors=upserted_data)
print("Upserted Remaining Vectors")
