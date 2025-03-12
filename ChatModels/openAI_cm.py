from langchain_openai import ChatOpenAI
from dotenv import load_dotenv

load_dotenv()

# model = ChatOpenAI(model='gpt-4')
model = ChatOpenAI(model='gpt-4', temperature='1.0')

'''
1. temperature - defines the randomness in response generation
    0.0 - 0.3 : factual answer(math, code)
    0.5 - 0.7: general responses
    0.9 - 1.2: creative responses(story, poems)
    1.5+ : random responses(brainsortming, wild ideas)
    
2. max_tokens/max_completion_tokens - defines the maximum number of tokens(words) in the response
'''

res = model.invoke("What is the name of the first president of the United States?")
print(res)
print(res.content)