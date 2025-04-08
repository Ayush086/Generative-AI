from langchain_community.document_loaders import WebBaseLoader
from langchain_cohere import ChatCohere
from langchain_core.output_parsers import StrOutputParser
from langchain_core.prompts import PromptTemplate
from dotenv import load_dotenv

load_dotenv()

model = ChatCohere(model='command-r')

prompt = PromptTemplate(
    template='Answer the following question: {question}\n based on the content of the webpage:\n\n{input}',
    input_variables=['question', 'input']
)

parser = StrOutputParser()


url = "https://www.geeksforgeeks.org/passive-aggressive-classifiers/"
loader = WebBaseLoader(url)
docs = loader.load()

chain = prompt | model | parser
question = "What are the takeaways from the article?"
answer = chain.invoke({'question': question, 'input': docs[0].page_content})
print("Answer: ", answer)

# print(len(docs))
# print(docs[0].page_content)