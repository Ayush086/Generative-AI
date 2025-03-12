from pydantic import BaseModel, EmailStr, Field
from typing import Optional, Literal
from langchain_cohere import ChatCohere
from dotenv import load_dotenv
import os

load_dotenv()

model = ChatCohere(model="command-r")

# class Student(BaseModel):
#     name: str
#     age: Optional[int] = None  # set default value to none
#     email: EmailStr
#     cgpa: float = Field(gt=0, lt=10, default=0, description="cummulative score of student")  # cgpa should be greater than 0 and less than or equal to 10
    

"""
 has ability of type-coercing to indentify the typecasting errors
 contain built-in validation functions for commonly used patterns. eg., emails, etc.
 Field - has many attributes which helps to define the key properly
"""
# st = {'name': 'Ayush', "age":'22', 'email': 'abc@example.com', 'cgpa': 8.6}
# # st = {'name': 86} # throws validation error
# s = Student(**st)

# student_json = s.model_dump_json()

# print(s)
# print(type(s))
# print(student_json)



# =================================================================================================================================================================


class Review(BaseModel):
    key_times: list[str] = Field(description="Write down al key themes discussed in review in a list")
    summary: str = Field(description="brief summary of the review")
    sentiment: Literal['positive', 'negative'] = Field(description='return sentiment of the review either negative, positive or neutral')
    # optional fields
    pros: Optional[list[str]] = Field(description='write all the pros about product in a list')
    cons: Optional[list[str]] = Field(description='write all the cons about product in a list')
    name: Optional[str] = Field(description='name of the reviewer', default=None)
    

structured_model = model.with_structured_output(Review)

result = structured_model.invoke("I recently purchased the XYZ wireless earbuds, and they have quickly become my favorite audio accessory. From the moment I unboxed them, I was impressed by their sleek, modern design and compact charging case. The earbuds fit snugly in my ears, providing a comfortable and secure fit, even during intense workouts. The sound quality is exceptional, with rich, deep bass and crisp, clear highs that make listening to music a truly immersive experience. The noise-canceling feature works wonders, effectively blocking out background noise and allowing me to focus on my tunes. The battery life is outstanding, lasting all day on a single charge, and the quick-charge feature is incredibly convenient for those times when I'm in a rush. The Bluetooth connectivity is rock-solid, with a stable connection and no audio dropouts, even when I'm moving around. I've used these earbuds for everything from calls to podcasts, and they've performed flawlessly in every scenario. The touch controls are intuitive and responsive, making it easy to adjust volume, skip tracks, and take calls without reaching for my phone. Overall, the XYZ wireless earbuds offer incredible value and performance at an affordable price. I highly recommend them to anyone in search of high-quality, reliable earbuds that deliver a top-notch audio experience. Reviwed by Customer")


print(result)

json_form = result.model_dump_json()
print(json_form)