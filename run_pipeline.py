import os
import torch
from src.config import Config
from src.data_loader import load_and_downsample_dataset
from src.preprocessor import get_tokenizer, preprocess_datasets
from src.model import initialize_model
from src.train import train_model
from src.evaluate import evaluate_model

def run():
    print("==================================================")
    print("      Starting NLP Deep Learning Pipeline        ")
    print("==================================================")
    
    # Set seeds for reproducibility
    torch.manual_seed(Config.SEED)
    import numpy as np
    np.random.seed(Config.SEED)

    device = Config.get_device()
    print(f"Hardware accelerator target: {device}")

    # Step 1: Load Data
    train_raw, test_raw = load_and_downsample_dataset()

    # Step 2: Preprocessing & Tokenization
    tokenizer = get_tokenizer()
    train_processed, test_processed = preprocess_datasets(train_raw, test_raw, tokenizer)

    # Step 3: Model Setup
    model = initialize_model(device=device)

    # Step 4: Training Loop
    model = train_model(model, train_processed, device)

    # Step 5: Evaluation
    accuracy = evaluate_model(model, test_processed, device)

    # Step 6: Save artifacts
    print(f"Saving fine-tuned model and tokenizer to '{Config.OUTPUT_DIR}'...")
    model.save_pretrained(Config.OUTPUT_DIR)
    tokenizer.save_pretrained(Config.OUTPUT_DIR)
    
    print("\nPipeline run complete! Model successfully trained, evaluated, and saved.")
    print("==================================================")

if __name__ == "__main__":
    run()
