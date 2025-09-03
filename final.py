##################data poisining##########
from fastapi import FastAPI, File, UploadFile, Query
from fastapi.responses import JSONResponse
from data_poisoning_api.data_poisoning.detector import detect_data_poisoning
from fastapi import FastAPI, UploadFile, File, Form
from fastapi.responses import HTMLResponse
from fastapi.staticfiles import StaticFiles
from fastapi.templating import Jinja2Templates
from fastapi.responses import JSONResponse
import numpy as np
import tensorflow as tf
from tensorflow.keras.models import load_model, Sequential
from tensorflow.keras.layers import Dense, Conv2D, Flatten, MaxPooling2D
from tensorflow.keras.utils import to_categorical
from art.attacks.evasion import FastGradientMethod, ProjectedGradientDescent
from art.estimators.classification import TensorFlowV2Classifier, SklearnClassifier
import pickle
import os

from fastapi import FastAPI, File, UploadFile, Form
from fastapi.responses import JSONResponse
import pandas as pd
import joblib
import os
import tempfile
import uvicorn


from fastapi import FastAPI
from pydantic import BaseModel
from P_Inj.inference import classify_prompt
import torch

app = FastAPI(title="API")

@app.post("/detect/data-poisoning")
async def detect_poisoning(
    model: UploadFile = File(...),
    image_index: int = Query(0, description="Index of the test image to use")
    ):
    result = detect_data_poisoning(model, image_index)
    return JSONResponse(content=result)

if __name__ == "__main__":
    uvicorn.run("main:app", host="0.0.0.0", port=9000, reload=True)


###### e_attack_ ###########


def load_preprocess_data():
    (x_train, y_train), (x_test, y_test) = tf.keras.datasets.mnist.load_data()
    x_train = x_train.astype(np.float32) / 255.0
    x_test = x_test.astype(np.float32) / 255.0
    x_train = np.expand_dims(x_train, axis=-1)
    x_test = np.expand_dims(x_test, axis=-1)
    y_train_cat = to_categorical(y_train, 10)
    y_test_cat = to_categorical(y_test, 10)
    return (x_train, y_train_cat), (x_test, y_test_cat), y_test

def load_model_from_bytes(model_bytes, filename):
    ext = os.path.splitext(filename)[-1]
    if ext == ".h5":
        with open("temp_model.h5", "wb") as f:
            f.write(model_bytes)
        #return load_model("temp_model.h5",compile = False), "keras"
        return load_model("temp_model.h5", compile=False), "keras"
    
    elif ext == ".pkl":
        with open("temp_model.pkl", "wb") as f:
            f.write(model_bytes)
        with open("temp_model.pkl", "rb") as f:
            return pickle.load(f), "sklearn"
    else:
        raise ValueError("Unsupported file type")


def create_art_classifier(model, model_type):
    if model_type == "keras":
        loss_object = tf.keras.losses.CategoricalCrossentropy()
        classifier = TensorFlowV2Classifier(
            model=model,
            nb_classes=10,
            input_shape=(28,28,1),
            loss_object=loss_object,
            clip_values=(0,1),
            channels_first=False)
    elif model_type == "sklearn":
        classifier = SklearnClassifier(model=model)
    else:
        raise ValueError("Unsupported model type")
    return classifier

def generate_adversarial_example(classifier, x_sample, attack_type="fgsm"):
    if attack_type == "fgsm":
        attack = FastGradientMethod(estimator=classifier, eps=0.2)
    elif attack_type == "pgd":
        attack = ProjectedGradientDescent(estimator=classifier, eps=0.3, eps_step=0.01, max_iter=40)
    else:
        raise ValueError("Unsupported attack type")
    return attack.generate(x=x_sample)

def detect_adversarial(classifier, x_clean, x_adv, y_true):
    pred_clean = np.argmax(classifier.predict(x_clean), axis=1)
    pred_adv = np.argmax(classifier.predict(x_adv), axis=1)
    true_label = np.argmax(y_true, axis=1)
    detection = (pred_clean == true_label) and (pred_adv != true_label)
    return detection, int(pred_clean[0]), int(pred_adv[0]), int(true_label[0])

@app.post("/detect/e_attack/")
async def detect_attack(index: int = Form(...), attack_type: str = Form(...), model_file: UploadFile = File(...)):
    if attack_type not in ["fgsm", "pgd"]:
        return JSONResponse(content={"error": "Invalid attack type"}, status_code=400)
    try:
        model_bytes = await model_file.read()
        model, model_type = load_model_from_bytes(model_bytes, model_file.filename)
    except Exception as e:
        return JSONResponse(content={"error": str(e)}, status_code=500)
    try:
        (x_train, y_train_cat), (x_test, y_test_cat), y_test = load_preprocess_data()
        classifier = create_art_classifier(model, model_type)
        x_sample = x_test[index:index+1]
        y_sample = y_test_cat[index:index+1]
        x_adv = generate_adversarial_example(classifier, x_sample, attack_type)
        is_adv, pred_clean, pred_adv, true_lbl = detect_adversarial(classifier, x_sample, x_adv, y_sample)
        return {
            "true_label": int(true_lbl),
            "prediction_clean": int(pred_clean),
            "prediction_adversarial": int(pred_adv),
             "is_adversarial": bool(is_adv)
        }
      

    except Exception as e:
        return JSONResponse(content={"error": str(e)}, status_code=500)
    
    ###############Prompt Injection Detection##############

#from utils.feature_extraction import extract_features
vectorizer = joblib.load(r"C:\Users\priti\Downloads\prompt_injection\models\vectorizer.pkl")

# Load your trained classifier
model = joblib.load(r"C:\Users\priti\Downloads\prompt_injection\models\prompt_injection_detector.pkl")

@app.post("/predict/prompt-injection/")
async def predict_prompt_injection(prompt: str = Form(None), file: UploadFile = File(None)):
    try:
        if file:
            with tempfile.NamedTemporaryFile(delete=False) as tmp:
                tmp.write(await file.read())
                tmp_path = tmp.name

            # Read CSV file and assume it has a column 'prompt'
            df = pd.read_csv(tmp_path)
            prompts = df['prompt'].astype(str).tolist()
            os.remove(tmp_path)
        elif prompt:
            prompts = [prompt]
        else:
            return JSONResponse(content={"error": "No prompt or file provided"}, status_code=400)

        # Extract features
        #features = extract_features(prompts)
        features = vectorizer.transform(prompts)


        # Predict
        predictions = model.predict(features)
        results = [{"prompt": p, "label": "malicious" if pred else "safe"} for p, pred in zip(prompts, predictions)]
        return {"results": results}
    except Exception as e:
        return JSONResponse(content={"error": str(e)}, status_code=500)
    

###### P_Inj ___############
class PromptInput(BaseModel):
    prompt: str

@app.post("/predict")
async def predict_prompt(input_data: PromptInput):
    prediction = classify_prompt(input_data.prompt)
    return {"prompt": input_data.prompt, "prediction": prediction}

