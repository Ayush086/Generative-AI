from langchain_cohere import ChatCohere
from langchain.prompts import PromptTemplate
from langchain_core.output_parsers import StrOutputParser
from dotenv import load_dotenv
from langchain.schema.runnable import RunnableParallel

load_dotenv()

# prompt template
prompt1 = PromptTemplate(
    template = "Give an interesting short fact about {topic}",
    input_variables=["topic"]
)

prompt2 = PromptTemplate(
    template= "Give an horrifying short fact about {topic}",
    input_variables=["topic"]
)

model = ChatCohere()

parser = StrOutputParser()

parallel_chain = RunnableParallel({
    "fact1": (prompt1 | model | parser),
    "fact2": (prompt2 | model | parser)
})

result = parallel_chain.invoke({"topic":"Engineering in Computer Science"})
print('fact1: ',result["fact1"])
print()
print('fact2: ',result["fact2"])
