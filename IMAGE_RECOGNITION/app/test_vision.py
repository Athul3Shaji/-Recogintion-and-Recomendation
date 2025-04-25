
from dotenv import load_dotenv
import os
load_dotenv()
from google.cloud import vision


print("Using credentials from:", os.getenv("GOOGLE_APPLICATION_CREDENTIALS"))

client = vision.ImageAnnotatorClient()

with open("canstockphoto26807912.jpg", "rb") as f:
    content = f.read()

image = vision.Image(content=content)
response = client.label_detection(image=image)

for label in response.label_annotations:
    print(f"{label.description}: {label.score}")
