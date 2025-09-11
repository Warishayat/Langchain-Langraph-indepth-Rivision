import os 
from dotenv import load_dotenv
from langchain_google_genai import ChatGoogleGenerativeAI
from typing import TypedDict,Annotated,Literal,Optional
load_dotenv()


GROQ_API_KEY = os.getenv("GROQ_API_KEY")
GOOGLE_API_KEY = os.getenv("GOOGLE_API_KEY")
HUGGINGFACEHUB_ACCESS_TOKEN = os.getenv("HUGGINGFACEHUB_ACCESS_TOKEN")

# in this whole lecture i will dicussed the Structured output
# i will discuss three main types 
# typeDict
# Pydantic 
# json Schema
model = ChatGoogleGenerativeAI(model="gemini-2.5-flash",api_key=GOOGLE_API_KEY)

# #class how data should look like that im expecting.
# class Product_Review(TypedDict):
#    sumamy:Annotated[str,"A brief summary of the review"]
#    sentiment:Annotated[str,"Sentiment of the review positive or negative"]

# s_model = model.with_structured_output(Product_Review)
# res=s_model.invoke("i dont want to buy this phone this app is not good and their feature is not ggod only the battery is good")
# print(res)
# print("---------------")
# print(res['sentiment'])
# print(res['sumamy'])


review = """
I recently purchased this phone and have been using it for a few weeks now. The first thing that impressed me is the sleek design and lightweight build, which makes it very comfortable to hold. The display is sharp and vibrant, offering a great experience for watching videos and browsing. Performance-wise, the phone handles multitasking smoothly and apps open quickly without much lag.

One of the strongest points is definitely the battery life. With moderate use, it easily lasts a full day and sometimes stretches into the second day, which is great for people always on the go. Fast charging is another plus, as the phone can reach 50% in less than half an hour, which is very convenient.

However, there are a few drawbacks as well. The battery tends to heat up slightly when playing heavy games or using the camera for extended periods. Also, while the standby time is good, constant use of mobile data drains the battery faster than expected. Another minor issue is that the charger provided is not as powerful as some competitors, which means full charging still takes a while.

Overall, the phone offers reliable performance and good value for money, but the battery, while strong, could be optimized better for heavy users.
"""

class Product_review(TypedDict):
    key_ideas : Annotated[list[str],"There will theme ideas that the review contain."]
    summary : Annotated[str,"A short summary of the review"]
    pros : Annotated[Optional[list[str]],"List of pros"]
    cons : Annotated[Optional[list[str]],"List of cons"]
    sentiment : Annotated[Literal["positive","negative","Neutral"],"Sentiment of the review"]

s_model = model.with_structured_output(Product_review)
result = s_model.invoke(review)
print()
print("typeDict  structured response")
print(result["sentiment"])
print(result.keys())


#there is somebig mistake typeDict is just for representation we will use pydantic for
# data validation.
from pydantic import BaseModel,Field
class StudentName(BaseModel):
    name:str="Waris hayat abbasi"
    age:Optional[int] = 12
    cgpa:float = Field(gt=1 ,lt=10,default=6.5,description="Its represnting cgpa of the student")

# student = StudentName(name="ali khan",cgpa=6.5) 
# print("the name of student")
# student_dict = dict(student)
# print(student_dict)

class Product_Review01(BaseModel):
    key_theme : list[str] = Field(description="this is the list of fields")
    summary : str = Field(description="A Short summary of the priduct")
    props : Optional[list[str]] = Field("List of pros if aviablable there")
    cons : Optional[list[str]] = Field("List of cons if aviablable there")
    sentiment : Literal["pos","neg"] = Field(description="Sentiment of the review")


s2_model = model.with_structured_output(Product_Review01)
res=s2_model.invoke(review)
print()
print("Pydantic BaseModel structured response")
res_dict = dict(res)
print(res_dict)



# Third type of input
#json_schema
json_schema = {
  "title": "Product_Review01",
  "type": "object",
  "properties": {
    "key_theme": {
      "type": "array",
      "items": {
        "type": "string"
      },
      "description": "this is the list of fields"
    },
    "summary": {
      "type": "string",
      "description": "A Short summary of the priduct"
    },
    "props": {
      "type": "array",
      "items": {
        "type": "string"
      },
      "description": "List of pros if aviablable there"
    },
    "cons": {
      "type": "array",
      "items": {
        "type": "string"
      },
      "description": "List of cons if aviablable there"
    },
    "sentiment": {
      "type": "string",
      "enum": ["pos", "neg"],
      "description": "Sentiment of the review"
    }
  },
  "required": [
    "key_theme",
    "summary",
    "sentiment"
  ]
}


s3_model = model.with_structured_output(json_schema)
result = s3_model.invoke(review)
print()
print("json schema structured response")
print(result)