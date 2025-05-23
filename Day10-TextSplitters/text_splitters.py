from langchain.text_splitter import CharacterTextSplitter
from langchain_community.document_loaders import PyPDFLoader

loader = PyPDFLoader("../Day09-DataLoaders/data/IJRESM_V7_I5_14.pdf")

docs = loader.lazy_load()

text = """
For the prevention of security issues in publicly accessible  areas,  it  is  necessary  to  integrate  computer  vision  and  AI  into  an  automatic  video  identification  system.  In  detecting abnormal behaviour, traditional surveillance methods are insufficient and a system needs to be automated. The objective of the  project  is  to  use  deep  learning  techniques,  especially  CNN models, for analysing video footage posted on a web site in order to make surveillance more efficient. In addition, it involves segmenting the video into frames, extracting features using MobileNetV2  and  identifying  irregular  or  suspicious  activities. The system's functions include background and foreground extraction and anomaly detection that allows for a distinct distinction in the behaviour of normal and irregular activities on surveillance  video. The study seeks to bridge  the gap between surveillance technology by involving computer vision, image processing  and  artificial  intelligence  so  as  to  be  able  to  quickly identify  unusual  actions on  video.  In  addition,  when  detecting potential security threats, it ensures that timely alerts are sent via email.  This  research  demonstrates  the  importance  of addressing emerging  security  challenges  in  today's  cities,  contributing  to enhancing surveillance systems.
"""

splitter = CharacterTextSplitter(
    chunk_size=400,
    chunk_overlap=0, # used to create overlapping chunks
    separator=""
)

# chunks = splitter.split_text(text)
chunks = splitter.split_documents(docs)
print("Number of chunks:", len(chunks))
print("First chunk:", chunks[10])
# print("chunks:", chunks)