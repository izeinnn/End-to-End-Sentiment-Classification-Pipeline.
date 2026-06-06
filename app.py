import os
from fastapi import FastAPI, HTTPException, Form
from fastapi.responses import HTMLResponse
from fastapi.staticfiles import StaticFiles
from fastapi.templating import Jinja2Templates
from pydantic import BaseModel
import torch
from transformers import AutoTokenizer, AutoModelForSequenceClassification

app = FastAPI(title="SentiFlow: End-to-End NLP Sentiment Classification Pipeline")

# Model configuration
MODEL_DIR = "./saved_model"
FALLBACK_MODEL = "distilbert-base-uncased-finetuned-sst-2-english"

# Determine hardware device
if torch.cuda.is_available():
    device = torch.device("cuda")
elif torch.backends.mps.is_available():
    device = torch.device("mps")
else:
    device = torch.device("cpu")

# Global variables to store loaded tokenizer & model
tokenizer = None
model = None
model_source = ""

def load_model():
    global tokenizer, model, model_source
    if os.path.exists(MODEL_DIR) and os.path.exists(os.path.join(MODEL_DIR, "config.json")):
        try:
            print(f"Loading custom fine-tuned model from '{MODEL_DIR}'...")
            tokenizer = AutoTokenizer.from_pretrained(MODEL_DIR)
            model = AutoModelForSequenceClassification.from_pretrained(MODEL_DIR)
            model.to(device)
            model.eval()
            model_source = "Custom Local Fine-Tuned Model (saved_model/)"
            return
        except Exception as e:
            print(f"Failed to load local model: {e}. Falling back to default...")
            
    print(f"Loading pretrained fallback model '{FALLBACK_MODEL}' from Hugging Face Hub...")
    tokenizer = AutoTokenizer.from_pretrained(FALLBACK_MODEL)
    model = AutoModelForSequenceClassification.from_pretrained(FALLBACK_MODEL)
    model.to(device)
    model.eval()
    model_source = f"Pre-trained Model ({FALLBACK_MODEL}) from Hugging Face Hub"

# Load model on startup
@app.on_event("startup")
def startup_event():
    load_model()

class SentimentRequest(BaseModel):
    text: str

class SentimentResponse(BaseModel):
    text: str
    sentiment: str
    confidence: float
    device: str
    model_used: str

@app.post("/api/predict", response_model=SentimentResponse)
async def predict_sentiment_api(request: SentimentRequest):
    global tokenizer, model, model_source
    if not request.text.strip():
        raise HTTPException(status_code=400, detail="Text cannot be empty")
        
    try:
        inputs = tokenizer(request.text, return_tensors="pt", truncation=True, padding=True)
        inputs = {k: v.to(device) for k, v in inputs.items()}
        
        with torch.no_grad():
            outputs = model(**inputs)
            
        probabilities = torch.nn.functional.softmax(outputs.logits, dim=-1)
        prediction = torch.argmax(probabilities, dim=-1).item()
        confidence = probabilities[0][prediction].item()
        
        # Check if output is binary classification mapping
        # Hugging Face SST-2 model might label 0=Negative, 1=Positive, or vice versa depending on config.
        # Generally SST-2 mapping: LABEL_0 is NEGATIVE, LABEL_1 is POSITIVE
        sentiment = "Positive" if prediction == 1 else "Negative"
        
        return SentimentResponse(
            text=request.text,
            sentiment=sentiment,
            confidence=confidence,
            device=str(device),
            model_used=model_source
        )
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

# Setup Jinja2 templates directory
templates = Jinja2Templates(directory="templates")

@app.get("/", response_class=HTMLResponse)
async def get_index():
    # Return HTML index page directly (we will write index.html to templates/)
    try:
        with open("templates/index.html", "r", encoding="utf-8") as f:
            html_content = f.read()
        return HTMLResponse(content=html_content)
    except FileNotFoundError:
        return HTMLResponse(content="<h1>Frontend file templates/index.html not found.</h1>", status_code=404)

@app.get("/api/status")
async def get_status():
    global model_source
    return {
        "status": "ready",
        "device": str(device),
        "model_used": model_source
    }
