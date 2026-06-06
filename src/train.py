import torch
from torch.utils.data import DataLoader
from transformers import AdamW, get_scheduler
from tqdm.auto import tqdm
from src.config import Config

def train_model(model, train_dataset, device, epochs=Config.EPOCHS, batch_size=Config.BATCH_SIZE, lr=Config.LEARNING_RATE):
    print("Preparing DataLoader...")
    train_dataloader = DataLoader(train_dataset, shuffle=True, batch_size=batch_size)

    # Optimizer and Scheduler setup
    optimizer = AdamW(model.parameters(), lr=lr)
    num_training_steps = epochs * len(train_dataloader)
    lr_scheduler = get_scheduler(
        "linear",
        optimizer=optimizer,
        num_warmup_steps=0,
        num_training_steps=num_training_steps
    )

    print("Starting native PyTorch training loop...")
    progress_bar = tqdm(range(num_training_steps), desc="Training")
    model.train()

    for epoch in range(epochs):
        total_loss = 0
        for batch in train_dataloader:
            # Move inputs to device
            batch = {k: v.to(device) for k, v in batch.items()}
            
            # Forward pass
            outputs = model(**batch)
            loss = outputs.loss
            
            # Backward pass
            loss.backward()
            
            # Optimization
            optimizer.step()
            lr_scheduler.step()
            optimizer.zero_grad()
            
            total_loss += loss.item()
            progress_bar.update(1)
            
        avg_loss = total_loss / len(train_dataloader)
        print(f"Epoch {epoch + 1}/{epochs} completed. Average Loss: {avg_loss:.4f}")

    return model
