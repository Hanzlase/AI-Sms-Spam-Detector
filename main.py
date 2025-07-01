from fastapi import FastAPI, Request, Form
from fastapi.responses import HTMLResponse
from fastapi.templating import Jinja2Templates
import joblib
import numpy as np
from mangum import Mangum
import os

app = FastAPI()

# Setup templates
templates = Jinja2Templates(directory="templates")

# Load the trained Naive Bayes classifier and TF-IDF vectorizer
model_dir = "model"
nb_classifier = joblib.load(os.path.join(model_dir, "nb_classifier.pkl"))
vectorizer = joblib.load(os.path.join(model_dir, "vectorizer.pkl"))

@app.get("/", response_class=HTMLResponse)
async def home(request: Request):
    return templates.TemplateResponse("index.html", {"request": request})

@app.post("/predict")
async def predict(message: str = Form(...)):
    message_vec = vectorizer.transform([message])
    prediction = nb_classifier.predict(message_vec)
    probability = np.max(nb_classifier.predict_proba(message_vec))

    result = {
        'prediction': 'Spam' if prediction[0] == 1 else 'Ham',
        'probability': f"{probability * 100:.2f}%"
    }
    return result

# Vercel serverless function handler
handler = Mangum(app)

if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8000)
