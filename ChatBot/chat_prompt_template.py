# Dynamic response generation based on previous chat history
from langchain_core.prompts import ChatPromptTemplate

chat_template = ChatPromptTemplate([
    ('system', 'You are a personal {domain} expert, who is always ready to help is master'),
    ('human', 'Tell me about {topic}')
])

prompt = chat_template.invoke({'domain': 'cricket', 'topic': "carrom-ball"})

print(prompt)