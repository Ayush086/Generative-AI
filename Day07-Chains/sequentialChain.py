from langchain_cohere import ChatCohere
from langchain_core.prompts import PromptTemplate
from langchain_core.output_parsers import StrOutputParser
from dotenv import load_dotenv

load_dotenv()


prompt1 = PromptTemplate(
    template="Generate 6 detailed report on {topic}",
    input_variables=['topic']
)

prompt2 = PromptTemplate(
    template = 'generate 5 points summary of following {text}',
    input_variables = ['text']
)

model = ChatCohere(model='command-r')

parser = StrOutputParser()

# generating a detailed report on given topic from LLM and summarizing it from LLM only 
chain = prompt1 | model | parser | prompt2 | model | parser

output = chain.invoke({'topic':'deep learning'})
print(output)

chain.get_graph().print_ascii()