import os
import torch
from transformers import AutoTokenizer, AutoModelForSequenceClassification
from src.config import Config

class SentimentPredictor:
    def __init__(self, model_dir=Config.OUTPUT_DIR, device=None):
        if device is None:
            self.device = Config.get_device()
        else:
            self.device = device
            
        print(f"Loading SentimentPredictor model from '{model_dir}' to {self.device}...")
        self.tokenizer = AutoTokenizer.from_pretrained(model_dir)
        self.model = AutoModelForSequenceClassification.from_pretrained(model_dir)
        self.model.to(self.device)
        self.model.eval()

    def predict(self, text: str):
        inputs = self.tokenizer(text, return_tensors="pt", truncation=True, padding=True)
        inputs = {k: v.to(self.device) for k, v in inputs.items()}
        
        with torch.no_grad():
            outputs = self.model(**inputs)
            
        probabilities = torch.nn.functional.softmax(outputs.logits, dim=-1)
        prediction = torch.argmax(probabilities, dim=-1).item()
        confidence = probabilities[0][prediction].item()
        
        sentiment = "Positive" if prediction == 1 else "Negative"
        return sentiment, confidence

if __name__ == "__main__":
    # Test script if weights exist
    if os.path.exists(Config.OUTPUT_DIR) and os.listdir(Config.OUTPUT_DIR):
        predictor = SentimentPredictor()
        text = "This movie was absolutely amazing! Highly recommended!"
        sentiment, conf = predictor.predict(text)
        print(f"Text: {text}\nPrediction: {sentiment} (Confidence: {conf*100:.2f}%)")
    else:
        print(f"No saved model found at '{Config.OUTPUT_DIR}'. Train the model first.")
