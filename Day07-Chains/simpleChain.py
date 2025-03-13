from langchain_cohere import ChatCohere
from langchain_core.prompts import PromptTemplate
from langchain_core.output_parsers import StrOutputParser
from dotenv import load_dotenv

load_dotenv()


prompt = PromptTemplate(
    template="Generate 6 interesting facts about {topic}",
    input_variables=['topic']
)

model = ChatCohere(model='command-r')

parser = StrOutputParser()

chain = prompt | model | parser

output = chain.invoke({'topic':'cricket'})
print(output)


chain.get_graph().print_ascii()