from langchain_openai import OpenAI
from dotenv import load_dotenv

load_dotenv()

# api_key = load_dotenv.get("OPENAI_API_KEY")

llm = OpenAI(model='gpt-3.5-turbo-instruct')

res = llm.invoke("What is the name of the first president of the United States?")
print(res)