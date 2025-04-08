from langchain_community.document_loaders import TextLoader
from langchain_cohere import ChatCohere
from langchain_core.output_parsers import StrOutputParser
from langchain_core.prompts import PromptTemplate
from dotenv import load_dotenv

load_dotenv()

model = ChatCohere(model='command-r')

prompt = PromptTemplate(
    template='Write a summary of the following text:\n\n{input} in 7-8 sentences.',
    input_variables=['input'],
)

parser = StrOutputParser()

# load text file
loader = TextLoader("data/program.txt", encoding='utf-8')

# load file in memory as langchain document
documents = loader.load()

# print(documents)

chain = prompt | model | parser
summary = chain.invoke({'input': documents[0].page_content})
# print the summary
print(summary)

## extras
# print(documents[0].page_content)
# print(documents[0].metadata)

