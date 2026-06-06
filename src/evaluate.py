import torch
from torch.utils.data import DataLoader
import evaluate
from src.config import Config

def evaluate_model(model, test_dataset, device, batch_size=Config.BATCH_SIZE):
    print("Evaluating model...")
    test_dataloader = DataLoader(test_dataset, batch_size=batch_size)
    metric = evaluate.load("accuracy")
    
    model.eval()
    for batch in test_dataloader:
        batch = {k: v.to(device) for k, v in batch.items()}
        
        with torch.no_grad():
            outputs = model(**batch)
            
        logits = outputs.logits
        predictions = torch.argmax(logits, dim=-1)
        
        # Accumulate metrics
        metric.add_batch(predictions=predictions, references=batch["labels"])

    eval_results = metric.compute()
    accuracy = eval_results["accuracy"]
    print(f"\nFinal Test Accuracy: {accuracy * 100:.2f}%")
    return accuracy
