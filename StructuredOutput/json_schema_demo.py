from pydantic import BaseModel, EmailStr, Field
from typing import Optional, Literal
from langchain_cohere import ChatCohere
from dotenv import load_dotenv
import os

load_dotenv()

model = ChatCohere(model="command-r")


# json schema
json_schema = {
    'title': "Review", # schema name
    'type': 'object', # schema type
    # variables
    'properties': {
        'key_themes': {
            'type': 'array',
            'items': {
                'type': 'string'
            },
            'description': 'Write down all key themes discussed in review in a list'
        },
        'summary': {
            'type': 'string',
            'description': 'brief summary of the review'
        },
        'sentiment': {
            'type': 'string',
            'enum': ['positive', 'negative'],
            'description': 'return sentiment of the review either negative, positive or neutral'
        },
        'pros': {
            'type': ['array', 'null'],
            'items': {
                'type': 'string'
            },
            'description': 'write all the pros about product in a list'
        },
        'cons': {
            'type': ['array', 'null'],
            'items': {
                'type': 'string'
            },
            'description': 'write all the cons about product in a list'
        },
        'name': {
            'type': ['string', 'null'],
            'description': 'name of the reviewer'
        }
    },
    'required': ['key_themes', 'summary', 'sentiment']
}


    

structured_model = model.with_structured_output(json_schema)

result = structured_model.invoke("I recently purchased the XYZ wireless earbuds, and they have quickly become my favorite audio accessory. From the moment I unboxed them, I was impressed by their sleek, modern design and compact charging case. The earbuds fit snugly in my ears, providing a comfortable and secure fit, even during intense workouts. The sound quality is exceptional, with rich, deep bass and crisp, clear highs that make listening to music a truly immersive experience. The noise-canceling feature works wonders, effectively blocking out background noise and allowing me to focus on my tunes. The battery life is outstanding, lasting all day on a single charge, and the quick-charge feature is incredibly convenient for those times when I'm in a rush. The Bluetooth connectivity is rock-solid, with a stable connection and no audio dropouts, even when I'm moving around. I've used these earbuds for everything from calls to podcasts, and they've performed flawlessly in every scenario. The touch controls are intuitive and responsive, making it easy to adjust volume, skip tracks, and take calls without reaching for my phone. Overall, the XYZ wireless earbuds offer incredible value and performance at an affordable price. I highly recommend them to anyone in search of high-quality, reliable earbuds that deliver a top-notch audio experience. Reviwed by Customer")


print(result)

# json_form = result.model_dump_json()
# print(json_form)