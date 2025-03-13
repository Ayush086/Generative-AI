from langchain_cohere import ChatCohere
from langchain_core.prompts import PromptTemplate
from langchain_core.output_parsers import StrOutputParser, PydanticOutputParser
from langchain.schema.runnable import RunnableBranch, RunnableLambda
from pydantic import BaseModel, Field
from typing import Literal
from dotenv import load_dotenv

load_dotenv()

model = ChatCohere(model='command-r')

parser = StrOutputParser()

class Feedback(BaseModel):
    sentiment: Literal['positive', 'negative'] = Field("provide sentiment of feedback")

parser2 = PydanticOutputParser(pydantic_object=Feedback)

prompt1 = PromptTemplate(
    template='classify the sentiment of following text into positive or negative \n {feedback} \n {format_instruction}',
    input_variables=['feedback'],
    partial_variables={'format_instruction': parser2.get_format_instructions()}
)

classifier_chain = prompt1 | model | parser2

# conditional chain
positive_prompt = PromptTemplate(
    template="write an appropriate response for this positive sentiment feedback \n {feedback}",
    input_variables=['feedback']
)

negative_prompt = PromptTemplate(
    template="write an appropriate response for this negative sentiment feedback \n {feedback}",
    input_variables=['feedback']
)


branch_chain = RunnableBranch(
    (lambda x: x.sentiment == 'positive', positive_prompt | model | parser),
    (lambda x: x.sentiment == 'negative', negative_prompt | model | parser),
    RunnableLambda(lambda x: "couldn't find sentiment")
)

final_chain = classifier_chain | branch_chain

output = final_chain.invoke({
    'feedback': 'The product was really terrible.'
})
print(output)