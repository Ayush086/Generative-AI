from langchain_community.document_loaders import TextLoader
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain_cohere import CohereEmbeddings, ChatCohere
from langchain_community.vectorstores import FAISS
from langchain_core.messages import HumanMessage


# load doc
loader = TextLoader('doc.txt')
documents = loader.load()

# split text into chunks
text_splitter = RecursiveCharacterTextSplitter(chunk_size=500, chunk_overlap=50)
docs = text_splitter.split_documents(documents=documents)

# convert text into embeddings and store in FAISS
embeddings = CohereEmbeddings(model="embed-english-v3.0")
vectorStore = FAISS.from_documents(docs, embeddings)

# create retriever - fetches relevant docs
retriever = vectorStore.as_retriever()

# fatch relevant embeddings
question = "Key takeaways from document ?"
retrieved_docs = retriever.get_relevant_documents(question)

# combine the retrieved documents into a single document
retrieved_text = "\n".join([doc.page_content for doc in retrieved_docs])

# initialize LLM
llm = ChatCohere(model='command-r')

# pass retrieved documents to model
prompt = f"Based on the following text, answer the question: {question}\n\n{retrieved_text}"
answer = llm.invoke(prompt)

print(answer.content)