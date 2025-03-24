from langchain_cohere import ChatCohere
from langchain.prompts import PromptTemplate
from langchain.chains import LLMChain

# model initialization
model = ChatCohere(model='command-r', temperature=0.7)

# prompt template
prompt = PromptTemplate(
    input_variables=['topic'],
    template='Suggest an eye-catching blog title about {topic}.'
)

# take input
topic = input("Enter the topic: ")

# format prompt
formatted_prompt = prompt.format(topic=topic)

# invoke the model
title = model.invoke(formatted_prompt)

# print title
# print("Generated title: ", title.content)
# print(title)


## Using Chains ##
chain = LLMChain(model=model, prompt=prompt)
# topic = input("Enter the topic: ")
output = chain.run(topic)

print("Generated title: ", output)