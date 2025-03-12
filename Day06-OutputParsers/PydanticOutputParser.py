from langchain_cohere import ChatCohere
from langchain_core.prompts import PromptTemplate
from langchain.output_parsers import PydanticOutputParser
from pydantic import BaseModel, Field
from dotenv import load_dotenv

load_dotenv()


# model initialization
model = ChatCohere(model='command-r')

class Person(BaseModel):
    name: str = Field(..., title='Name of the person')
    age: int = Field(..., title='Age of the person', gt=18)
    city: str = Field(..., title='City of the person')
    
parser = PydanticOutputParser(pydantic_object=Person)

template = PromptTemplate(
    template="give me the name, age and city of a fictional {place} person \n {format_instruction}",
    input_variables=[],
    partial_variables={'format_instruction': parser.get_format_instructions()}
)

prompt = template.format(place='Italian')
print(f"Prompt:\n{prompt}")
result = model.invoke(prompt)
final_result = parser.parse(result.content)
print(final_result)