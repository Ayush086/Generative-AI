from langchain_core.messages import SystemMessage, HumanMessage, AIMessage
from langchain_cohere import ChatCohere
from dotenv import load_dotenv

load_dotenv()

model = ChatCohere(model="command-r")

messages = [
    SystemMessage(content='You are a personal assistant'),
    HumanMessage(content='Tell me about NLP')
]

result = model.invoke(messages)

messages.append(AIMessage(content=result.content))
print(messages)