# to dyanamically use the chat history to get references from the query
from langchain_core.prompts import ChatPromptTemplate, MessagesPlaceholder
from langchain_core.messages import HumanMessage
import os

# chat template
chat_template = ChatPromptTemplate([
    ('system', 'You are a virtual customer support agent'),
    MessagesPlaceholder(variable_name='chat_history'),
    ('human', '{query}')
])

# load chat history
chat_history = []
with open('chat_history_file.txt', 'r') as f:
    chat_history.extend(f.readlines())
    
print(chat_history)

# create prompt
prompt = chat_template.invoke({'chat_history': chat_history, 'query': HumanMessage(content='What is my refund status ?')})

print(prompt)