from openai import OpenAI
import base64
from textwrap import dedent
from pydantic import BaseModel
from typing import List

client = OpenAI(
  api_key=""
)

developerPrompt = """You are a image reader specialized in extracting the information inside a medical blood report and convert it into the given structured data.
1. The user will provide a image or a sequence of image.
2. Recognize and extract the data of each exam result.
3. return it into the given structured data.


Now i define more in the detail how is structured the medical blood report

The medical blood report is made of 4 columns:
- a test name such as MCH, MCV and so on. be careful. 
-- we call this value "field_name"
-- Some names has a slash like "GOT/AST TRANSAMINASI" dont forget it
-- this should be treated as a str type
- a result value given by a numerical value
-- we call this value "field_value"
-- the numerical value should be treated as a str type.
-- The decimal value uses a comma. Substitute it with a point. for example "3,15" becomes "3.15"
- a unit of measurements such as ng/ML, 10^3/mm^3 or sometimes just a %
-- we call this value "field_unit_of_measure"
- a range between two numerical value values
-- the lowest value is called "reference_range_low"
-- the highest value is called "reference_range_high"
-- the numerical values should be rteated as a str type
-- if there isn't any range but you found something like " > x", that means the "reference_range_low" = x and "reference_range_low" = None
-- if there isn't any range but you found something like " < x", that means the "reference_range_low" = None
 and reference_range_high" = x
-- if there is a case not defined by me, that means the "reference_range_low" = None
 and "reference_range_high" = None

When i talk about integer, float, None value i'm refering to pydantic type

In the image or images given by the user other than the 4 columns that im interested of extracting, there are other stuff that you could recognize. Such as logo, other string. Ignore them.

Be careful to detect every blood exam. Sometimes, each row is closer with each other, other rows are more separated.

It's possible that the language of the text inside the image it's not english. But any medical blood report should always have four columns that you could recognize the structure as explained

the image could be not completely straight. Because it's a piece of paper the image of the paper could be sligtly rotated or a little bit folded. This could make each row, not straight but a little bended or rotated.
While reading each row, you should take the previous row as a reference to understand the next row

Before completing your task, keep extracting the same input image several times and compare the result.
if your current result is similar to the last 2 attempts, that means, your extraction is correct.
Otherwise your current result is wrong and needs to be discarded.
Remember, keep extracting and compare the results until you will have 3 extraction that is equal to each other.
"""

# Function to encode the image
def encode_image(image_path):
    with open(image_path, "rb") as image_file:
        return base64.b64encode(image_file.read()).decode("utf-8")
    
# Path to your image
image1 = "immagine1.png"
image2 = "immagine2.png"

# Getting the Base64 string
base64_image1 = encode_image(image1)
base64_image2 = encode_image(image2)



class singleGroup(BaseModel):
    field_name: str
    field_value: str
    field_unit_of_measure: str
    reference_range_low: str | None
    reference_range_high: str | None

class testResults(BaseModel):
    answer:List[singleGroup]

completion = client.beta.chat.completions.parse(
  model="gpt-4o-mini",
  temperature=0,
  messages=[
    {
        "role":"developer",
        "content":developerPrompt
    },
    {
        "role": "user", 
        "content": [
            {
                "type": "image_url",
                "image_url":{
                    "url": f"data:image/jpeg;base64,{base64_image1}",
                    "detail": "high",
                    },

            },
            {
                "type": "image_url",
                "image_url":{
                    "url": f"data:image/jpeg;base64,{base64_image2}",
                    "detail": "high",
                    },
     
            }
        ]
    }
  ],
  response_format = testResults
)

print(completion.choices[0].message.parsed)
