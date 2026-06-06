import os
import torch

class Config:
    # Hardware device setup
    @staticmethod
    def get_device():
        if torch.cuda.is_available():
            return torch.device("cuda")
        elif torch.backends.mps.is_available():
            return torch.device("mps")
        else:
            return torch.device("cpu")

    # Hyperparameters
    EPOCHS = 3
    LEARNING_RATE = 2e-5
    BATCH_SIZE = 16
    MAX_LENGTH = 256
    SEED = 42

    # Downsampling parameters (keeps execution quick for portfolios)
    TRAIN_SIZE = 2000
    TEST_SIZE = 500

    # Model parameters
    MODEL_CHECKPOINT = "distilbert-base-uncased"
    NUM_LABELS = 2

    # Directory Paths
    OUTPUT_DIR = "./saved_model"
    DATA_DIR = "./data"

    # Make output directory
    os.makedirs(OUTPUT_DIR, exist_ok=True)
    os.makedirs(DATA_DIR, exist_ok=True)
