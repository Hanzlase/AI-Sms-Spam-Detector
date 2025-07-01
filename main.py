from fastapi import FastAPI, Request, Form
from fastapi.responses import HTMLResponse
from fastapi.templating import Jinja2Templates
from fastapi.staticfiles import StaticFiles
import joblib
import numpy as np
from mangum import Mangum

app = FastAPI()

# Setup templates
templates = Jinja2Templates(directory="templates")

# Load the trained Naive Bayes classifier and TF-IDF vectorizer
nb_classifier = joblib.load('model/nb_classifier.pkl')
vectorizer = joblib.load('model/vectorizer.pkl')

@app.get("/", response_class=HTMLResponse)
async def home(request: Request):
    return templates.TemplateResponse("index.html", {"request": request})

@app.post("/predict")
async def predict(message: str = Form(...)):
    # Convert the message into a TF-IDF vector
    message_vec = vectorizer.transform([message])

    # Use the trained classifier to make a prediction
    prediction = nb_classifier.predict(message_vec)

    # Get the probability/confidence of the prediction
    probability = np.max(nb_classifier.predict_proba(message_vec))

    # Prepare the result to send back to frontend
    result = {
        'prediction': 'Spam' if prediction[0] == 1 else 'Ham',
        'probability': f"{probability * 100:.2f}%"
    }

    return result

# This handler will make the FastAPI app work with Vercel serverless functions
handler = Mangum(app)

if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8000)
