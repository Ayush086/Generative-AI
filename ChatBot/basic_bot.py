from langchain_cohere import ChatCohere
from langchain_core.prompts import PromptTemplate, load_prompt
from langchain_core.messages import SystemMessage, HumanMessage, AIMessage
from dotenv import load_dotenv
import streamlit as st
import os

load_dotenv()

model = ChatCohere(model='command-r')

chat_history = [
    SystemMessage(content='You are a personal AI assistant who is always ready to help is master')
]

while True:
    user_input = input('You: ')
    chat_history.append(HumanMessage(content=user_input))
    
    if user_input == 'exit':
        break
    
    result = model.invoke(chat_history)
    chat_history.append(AIMessage(content=result.content))
    print("BOT: ", result.content)
    print("--------------------------------\n")
    
print("---------------Chat Ended-----------------")
print("Final Chat History: ", chat_history)