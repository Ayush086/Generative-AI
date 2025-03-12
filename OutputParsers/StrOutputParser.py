from langchain_cohere import ChatCohere
from langchain_huggingface import ChatHuggingFace, HuggingFaceEndpoint
from langchain_core.prompts import PromptTemplate
from langchain_core.output_parsers import StrOutputParser
from dotenv import load_dotenv

load_dotenv()


# model initialization
llm = HuggingFaceEndpoint(
    repo_id='google/gemma-2-2b-it',
    task='text-generation'
)

model = ChatHuggingFace(llm=llm)

# model = ChatCohere(model='command-r')

## Work Flow ##
# detailed report on any topic
template1 = PromptTemplate(
    template='Generate a detailed report on {topic}',
    input_variables=['topic']
)


# # summary of that topic
template2 = PromptTemplate(
    template="Write a 10 Line summary on the following text: /n {text}",
    input_variables=['text']
)

# # get the detailed report abouttopic
# # prompt1 = template1.invoke({'topic': "black hole"})
# prompt1 = template1.format(topic="black hole")
# result = model.invoke(prompt1)
# # then generate the summary
# # prompt2 = template2.invoke({'text': result.content})
# prompt2 = template2.format(text=result.content)
# summary = model.invoke(prompt2)

# # finally, provide the summary
# print(summary.content)




## Performing same with string output parsers ##
parser = StrOutputParser()
chain = template1 | model | parser | template2 | model | parser

result = chain.invoke({'topic': 'black hole'})
print(result)