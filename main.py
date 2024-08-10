from fastapi import FastAPI, HTTPException
from fastapi.responses import FileResponse
from fastapi.staticfiles import StaticFiles
from pydantic import BaseModel
from typing import Dict
import os

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

def analyze_sentiment(text: str) -> str:
    # TODO: Replace this logic with an actual ML model
    if "good" in text.lower():
        return "positive"
    elif "bad" in text.lower():
        return "negative"
    else:
        return "neutral"

@app.post("/analyse-sentiment")
async def analyse_sentiment(review: Review) -> Dict[str, str]:
    sentiment = analyze_sentiment(review.review)
    if sentiment:
        return {"sentiment": sentiment}
    else:
        raise HTTPException(status_code=400, detail="Error in sentiment analysis")
