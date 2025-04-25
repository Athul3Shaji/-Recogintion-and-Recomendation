from fastapi import FastAPI, File, UploadFile
from google.cloud import vision
from PIL import Image
import io
import os
from dotenv import load_dotenv

load_dotenv() 

print(load_dotenv)
os.environ["GOOGLE_APPLICATION_CREDENTIALS"] = os.getenv("GOOGLE_APPLICATION_CREDENTIALS")


app = FastAPI()
client = vision.ImageAnnotatorClient()

@app.get("/")
def root():
    return {"message": "Image Recognition API is Live!"}




@app.post("/analyze/")
async def analyze_image(file: UploadFile = File(...)):
    image_bytes = await file.read()
    image = vision.Image(content=image_bytes)

    # Call Google Cloud Vision API
    response = client.annotate_image({
        'image': image,
        'features': [
            {'type': vision.Feature.Type.OBJECT_LOCALIZATION},
            {'type': vision.Feature.Type.FACE_DETECTION},
            {'type': vision.Feature.Type.TEXT_DETECTION}
        ],
    })

    results = {
        "objects": [obj.name for obj in response.localized_object_annotations],
        "faces": len(response.face_annotations),
        "texts": [text.description for text in response.text_annotations]
    }

    return results