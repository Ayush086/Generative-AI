from langchain_cohere import ChatCohere
from langchain.prompts import PromptTemplate
from langchain_core.output_parsers import StrOutputParser
from dotenv import load_dotenv
from langchain.schema.runnable import RunnablePassthrough, RunnableBranch, RunnableSequence

load_dotenv()

def count_words(text):
    return len(text.split())

prompt1 = PromptTemplate(
    template = "Give an interesting short story about {topic}",
    input_variables=["topic"]
)

prompt2 = PromptTemplate(
    template = "Summarize the provided story: {text}",
    input_variables=["text"]
)

model = ChatCohere(model='command-r')

parser = StrOutputParser()

sequential_chain = RunnableSequence(prompt1, model, parser)

conditional_branch = RunnableBranch(
    (lambda x: count_words(x) > 150, RunnableSequence(prompt2, model, parser)),
    RunnablePassthrough()
)

final_chain = RunnableSequence(
    sequential_chain,
    conditional_branch
)

output = final_chain.invoke({'topic': 'black hole'})
print(output)

