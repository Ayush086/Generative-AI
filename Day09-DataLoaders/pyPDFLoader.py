from langchain_community.document_loaders import PyPDFLoader

loader = PyPDFLoader("data/IJRESM_V7_I5_14.pdf")
# docs = loader.load()
docs = loader.lazy_load()

for pages in docs:
    print(pages.page_content)
    print(pages.metadata)
    print("==="*40)
    
    
    
"""
load() vs lazy_load()

- load() :
    - loads the entire document into memory as a list of Document objects.
- lazy_load() : 
    - returns a generator that yields Document objects one by one, allowing you to process large documents without loading them all into memory at once.
    - is useful for large documents or when you want to process documents in a streaming fashion.
"""