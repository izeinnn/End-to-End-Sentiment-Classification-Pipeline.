from transformers import AutoModelForSequenceClassification
from src.config import Config

def initialize_model(model_checkpoint=Config.MODEL_CHECKPOINT, num_labels=Config.NUM_LABELS, device=None):
    if device is None:
        device = Config.get_device()
        
    print(f"Initializing model '{model_checkpoint}' on {device}...")
    model = AutoModelForSequenceClassification.from_pretrained(model_checkpoint, num_labels=num_labels)
    model.to(device)
    return model
