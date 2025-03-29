from langchain_cohere import ChatCohere
from langchain.prompts import PromptTemplate
from langchain_core.output_parsers import StrOutputParser
from dotenv import load_dotenv
from langchain.schema.runnable import RunnableSequence

load_dotenv()

# prompt template
prompt = PromptTemplate(
    template = "Give an interesting fact about {topic}",
    input_variables=["topic"]
)

prompt2 = PromptTemplate(
    template= "Summarize the provided text into 15-25 words: {text}",
    input_variables=["text"]
)

model = ChatCohere()

parser = StrOutputParser()

chain = RunnableSequence(prompt, model, parser, prompt2, model, parser)

print(chain.invoke({"topic":"Engineering"}))
