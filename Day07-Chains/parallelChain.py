from langchain_cohere import ChatCohere
from langchain_core.prompts import PromptTemplate
from langchain_core.output_parsers import StrOutputParser
from langchain.schema.runnable import RunnableParallel
from dotenv import load_dotenv

load_dotenv()

model1 = ChatCohere(model='command-r')
model2 = ChatCohere(model='command-r')

prompt1 = PromptTemplate(
    template="Generate Text short and simple notes from the following text: \n {text}",
    input_variables=['text']
)

prompt2 = PromptTemplate(
template="Generate 5 MCQs from the following text: \n {text}",
input_variables=['text']
)

prompt3 = PromptTemplate(
    template='merge the provided notes and quiz into a single document \n Notes -> {notes} and Quiz -> {quiz}',
    input_variables=['notes', 'quiz']
)

parser = StrOutputParser()

parallel_chain = RunnableParallel({
    'notes_generation': prompt1 | model1 | parser,
    'quiz_generation': prompt2 | model2 | parser
})

merged_chain = prompt3 | model1 | parser

output = parallel_chain.invoke({
    'text': """Deep learning is a subset of machine learning in artificial intelligence (AI) that has networks capable of learning unsupervised from data that is unstructured or unlabeled. This technique is also known as deep neural learning or deep neural networks. The fundamental idea behind deep learning is to use neural networks with many layers (hence "deep") to model complex patterns and relationships within data.
    In traditional machine learning, features are often hand-engineered by humans to create a model. In contrast, deep learning enables a system to automatically learn features and representations from raw data, which can be in various forms, such as text, images, audio, and video. This capability makes deep learning particularly powerful for tasks involving large amounts of unstructured data.
    Deep learning models, known as neural networks, consist of layers of interconnected nodes (neurons). Each layer transforms the input data in a non-linear way, allowing the network to learn and model intricate patterns. The first layer, known as the input layer, receives the raw data. Subsequent layers, called hidden layers, process the data through various transformations. The final layer, known as the output layer, generates predictions or classifications.
    Training a deep learning model involves using a large dataset and an optimization algorithm called backpropagation. The model adjusts its internal parameters (weights) to minimize the difference between its predictions and the actual data. This iterative process continues until the model achieves a desired level of accuracy.
    Deep learning has achieved remarkable success in various fields, including computer vision, natural language processing, speech recognition, and autonomous systems. For example, deep learning models power image recognition systems, enabling applications such as facial recognition, object detection, and medical image analysis. In natural language processing, deep learning is used for tasks like language translation, sentiment analysis, and chatbots.
    The ability of deep learning models to automatically learn from raw data, combined with their capacity to handle vast and complex datasets, has revolutionized the field of artificial intelligence, opening new possibilities for innovation and discovery."""
})
# print(output)

result = merged_chain.invoke({
    'notes': output['notes_generation'],
    'quiz': output['quiz_generation']
})
print(result)
parallel_chain.get_graph().print_ascii()