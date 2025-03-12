from langchain_cohere import ChatCohere
from langchain_core.prompts import PromptTemplate
from langchain.output_parsers import StructuredOutputParser, ResponseSchema
from dotenv import load_dotenv

load_dotenv()


# model initialization
model = ChatCohere(model='command-r')

# define a response schema
schema = [
    ResponseSchema(
        name='fact_1',
        type='str',
        description='fact 1 about the topic'
    ),
    ResponseSchema(
        name='fact_2',
        type='str',
        description='fact 2 about the topic'
    ),
    ResponseSchema(
        name='fact_3',
        type='str',
        description='fact 3 about the topic'
    )
]

parser = StructuredOutputParser.from_response_schemas(schema)

template = PromptTemplate(
    template = 'give 3 fact about the topic: {topic}\n {format_instruction}',
    input_variables=['topic'],
    partial_variables={'format_instruction': parser.get_format_instructions()}
)

# prompt = template.invoke({'topic': 'black hole'})
# result = model.invoke(prompt)
# print(result.content)
# final_result = parser.parse(result.content)
# print(final_result)


chain = template | model | parser
res = chain.invoke({'topic': 'black hole'})
print(res)