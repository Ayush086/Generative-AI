# lecture-5 (Structured Output)                                                                                                     Date: 04/03/2025
from typing import TypedDict, Annotated, Optional, Literal
from langchain_cohere import ChatCohere
from dotenv import load_dotenv
import os

load_dotenv()

model = ChatCohere(model="command-r")

 ## EXample of typed dictionary ##
# class Person(TypedDict):
#     # keys in dictionary
#     name: str
#     age: int
    
    
    
# newPerson = Person(name="John", age=30)
# newP: Person = {"name": "John", "age": 30}
# print(newP)


# --------------------------------------------------------

# schema
class Review(TypedDict):
    key_themes = Annotated[list[str], "Write down al key themes discussed in review in a list"]
    summary: Annotated[str, "brief summary of the review"]
    sentiment: Annotated[Literal['positive', 'negative'], 'return sentiment of the review either negative, positive or neutral']
    # optional fields
    pros: Annotated[Optional[list[str]], 'write all the pros about product in a list']
    cons: Annotated[Optional[list[str]], 'write all the cons about product in a list']
    
"""
    TypedDict -> define the schema of the dictionary
    Annotated -> enable to give description about key
    Optional -> makes key is as optional like if unable to find the answer for that key then it's okay
    Literal -> restrict the value of key to only the values given in the list
"""
    
structured_model = model.with_structured_output(Review)

result = structured_model.invoke("I recently purchased the XYZ wireless earbuds, and they have quickly become my favorite audio accessory. From the moment I unboxed them, I was impressed by their sleek, modern design and compact charging case. The earbuds fit snugly in my ears, providing a comfortable and secure fit, even during intense workouts. The sound quality is exceptional, with rich, deep bass and crisp, clear highs that make listening to music a truly immersive experience. The noise-canceling feature works wonders, effectively blocking out background noise and allowing me to focus on my tunes. The battery life is outstanding, lasting all day on a single charge, and the quick-charge feature is incredibly convenient for those times when I'm in a rush. The Bluetooth connectivity is rock-solid, with a stable connection and no audio dropouts, even when I'm moving around. I've used these earbuds for everything from calls to podcasts, and they've performed flawlessly in every scenario. The touch controls are intuitive and responsive, making it easy to adjust volume, skip tracks, and take calls without reaching for my phone. Overall, the XYZ wireless earbuds offer incredible value and performance at an affordable price. I highly recommend them to anyone in search of high-quality, reliable earbuds that deliver a top-notch audio experience.")
print(result)


""" Above code will not work because of compatibility issues with TypedDict. Langchain functions are compatible with well known models (openAI, anthropic) but not with much open-sourced models."""