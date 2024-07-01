from ocrcheckup.benchmark.model import OCRBaseModel, OCRModelResponse, OCRModelInfo

from openai import OpenAI
from io import BytesIO
from PIL import Image
import base64

class OpenAI_GPT4o(OCRBaseModel):
  def info(self=None):
    return OCRModelInfo(
      name = "GPT-4o",
      version = "gpt-4o-2024-05-13",
      tags = ["cloud","lmm"]
    )

  def __init__(self,api_key):
    super().__init__()
    self.client = OpenAI(api_key=api_key)

  def evaluate(self,image):
    buffered = BytesIO()
    Image.fromarray(image).save(buffered, format="JPEG")
    img_str = base64.b64encode(buffered.getvalue()).decode("utf-8")

    response = self.client.chat.completions.create(
      model="gpt-4o",
      messages=[
          {
          "role": "user",
          "content": [
              {
              "type": "text",
              "text": "Read the text in the image. Return only the text as it is visible in the image."
              },
              {
              "type": "image_url",
              "image_url": {
                  "url": f"data:image/jpeg;base64,{img_str}"
              }
              }
          ]
          }
      ],
      temperature=1,
      max_tokens=256,
      top_p=1,
      frequency_penalty=0,
      presence_penalty=0
    )

    text = response.choices[0].message.content

    input_price = (response.usage.prompt_tokens/1000000) * 5
    output_price = (response.usage.completion_tokens/1000000) * 15
    cost = input_price+output_price

    return OCRModelResponse(
      prediction=text,
      cost=cost
    )