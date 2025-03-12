from langchain_cohere import ChatCohere
from langchain_core.prompts import PromptTemplate
from langchain_core.output_parsers import JsonOutputParser
from dotenv import load_dotenv

load_dotenv()


# model initialization
model = ChatCohere(model='command-r')

parser = JsonOutputParser()
template = PromptTemplate(
    template="give me the name, age and city of a fictional person \n {format_instruction}",
    input_variables=[],
    partial_variables={'format_instruction': parser.get_format_instructions()}
)

# prompt = template.format()
# print(f"Prompt:\n{prompt}")
# result = model.invoke(prompt)
# final_result = parser.parse(result.content)

chian = template | model | parser
final_result = chian.invoke({})
print(f"Result: \n {final_result}")

