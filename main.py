from fastapi import FastAPI, HTTPException
from fastapi.responses import FileResponse
from fastapi.staticfiles import StaticFiles
from pydantic import BaseModel
from typing import Dict
import os
import tensorflow as tf
from tensorflow.keras.preprocessing.sequence import pad_sequences
import json

app = FastAPI()

# Serve static files (images, CSS, etc.)
app.mount("/static", StaticFiles(directory="static"), name="static")

# Serve the main HTML file (index.html)
@app.get("/")
async def read_index():
    return FileResponse("templates/index.html")

# Data model for review input
class Review(BaseModel):
    review: str

# Load the trained model and tokenizer
model = tf.keras.models.load_model('app-model/sentiment_model_final.keras')

with open('app-model/tokenizer.json') as f:
    tokenizer_json = f.read()  # Read the file as a string
    tokenizer = tf.keras.preprocessing.text.tokenizer_from_json(tokenizer_json)

max_length = model.input_shape[1]  # Get max_length from model input shape

def analyze_sentiment(text: str) -> str:
    # Preprocess the input text
    sequence = tokenizer.texts_to_sequences([text])
    padded_sequence = pad_sequences(sequence, maxlen=max_length, padding='post', truncating='post')
    
    # Make a prediction
    prediction = model.predict(padded_sequence)[0][0]
    
    # Determine sentiment based on the prediction
    if prediction > 0.5:
        return "positive"
    else:
        return "negative"


@app.post("/analyse-sentiment")
async def analyse_sentiment(review: Review) -> Dict[str, str]:
    sentiment = analyze_sentiment(review.review)
    if sentiment:
        return {"sentiment": sentiment}
    else:
        raise HTTPException(status_code=400, detail="Error in sentiment analysis")
