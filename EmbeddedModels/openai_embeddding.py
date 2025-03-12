from langchain_openai import OpenAIEmbeddings
from dotenv import load_dotenv

load_dotenv()

embedding = OpenAIEmbeddings(model='text-embedding-3-large', dimensions=32)

documents = [
    "MS Dhoni is called as thala in cricket game.",
    "virat kohli has nickname in cricket which is run machine",
    "Rohit Sharma is my favorite cricketer.",
    "Virat Kohli is greatest cricketer till date."
]
# result = embedding.embed_query("Who is the god of cricket ?")
result = embedding.embed_documents(documents)

print(result)
print(str(result))