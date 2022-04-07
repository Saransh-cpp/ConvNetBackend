import gc
import os
import cv2
import numpy as np
from keras.models import load_model
from keras.backend import clear_session
from fastapi import FastAPI, File, UploadFile


app = FastAPI()

labels = {
    0: "negative",
    1: "positive",
}


@app.get("/")
def home():
    return {"message": "Please refer to the README for more information."}


@app.post("/predict")
async def predict(image: UploadFile = File(...)):

    with open("image.jpg", "wb+") as f:
        f.write(image.file.read())

    img = cv2.imread("image.jpg")
    img = cv2.resize(img, (224, 224))

    img = img / 255.0

    model = load_model("./model/model.h5")
    y_pred = model.predict(np.array([img]))

    category = labels[np.argmax(y_pred.flatten())]

    os.remove("image.jpg")
    clear_session()
    gc.collect()
    del model
    gc.collect()

    return {"category": category}
