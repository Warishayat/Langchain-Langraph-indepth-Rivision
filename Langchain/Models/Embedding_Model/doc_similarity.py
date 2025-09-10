from langchain_huggingface import ChatHuggingFace,HuggingFaceEndpoint,HuggingFaceEmbeddings
from sklearn.metrics.pairwise import cosine_similarity
import numpy as np

text = [
    "hello my name is waris hayat abbasi",
    "software enginner work at goolge",
    "dairy forms people are creative minds",
]

query = "i have dairy forms"


embeddings = HuggingFaceEmbeddings(model="sentence-transformers/all-MiniLM-L6-v2",)


docs_em = embeddings.embed_documents(text)
query = embeddings.embed_query(query)

score = cosine_similarity([query],docs_em)[0]

index,score=sorted(list(enumerate(score)),key=lambda x:x[1])[-1]

print(text[index])
print("Similarity score is:",score)