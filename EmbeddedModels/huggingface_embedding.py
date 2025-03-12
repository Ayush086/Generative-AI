from langchain_huggingface import HuggingFaceEmbeddings
from dotenv import load_dotenv

load_dotenv()

embedding = HuggingFaceEmbeddings(model_name='sentence-transformers/all-MiniLLM-L6-v2')
text = "Who is the first president of india ?"
vector = embedding.embed_query(text=text)

print(vector)
print(str(vector))