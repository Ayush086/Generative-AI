from langchain_openai import OpenAIEmbeddings
from dotenv import load_dotenv
from sklearn.metrics.pairwise import cosine_similarity
import numpy as numpy

embedding = OpenAIEmbeddings(model='text-embedding-3-large', dimensions=200)

documents = [
    "virat kohli is an indian cricketer known for his aggressive batting and leadership.",
    "MS Dhoni is a former Indian captain famous for his calm demeanor and finishing skills.",
    "Sachin Tendulkar is a retired Indian cricketer who is considered one of the greatest batsmen of all time.",
    "Rohit Sharma is known for his elegant batting and record-breaking double centuries.",
    "Jasprit Bumrah is an indian cricketer known for his unorthodox action and yorkers."
]

query = "tell me about rohit sharma ?"

doc_embeddings = embedding.embed_documents(documents)
query_embedding = embedding.embed_query(query)

similarity_scores = cosine_similarity([query_embedding], doc_embeddings)
print(similarity_scores[0])
