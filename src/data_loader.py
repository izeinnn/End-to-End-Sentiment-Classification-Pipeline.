from datasets import load_dataset
from src.config import Config

def load_and_downsample_dataset(train_size=Config.TRAIN_SIZE, test_size=Config.TEST_SIZE, seed=Config.SEED):
    print("Loading IMDb dataset from Hugging Face...")
    raw_datasets = load_dataset("imdb")

    print(f"Downsampling to {train_size} train and {test_size} test samples...")
    train_dataset = raw_datasets["train"].shuffle(seed=seed).select(range(train_size))
    test_dataset = raw_datasets["test"].shuffle(seed=seed).select(range(test_size))

    print(f"Loaded datasets. Train size: {len(train_dataset)}, Test size: {len(test_dataset)}")
    
    # Print sample data point
    sample = train_dataset[0]
    print(f"Sample data point:\nText: {sample['text'][:150]}...\nLabel: {sample['label']} (0=Neg, 1=Pos)")

    return train_dataset, test_dataset

if __name__ == "__main__":
    load_and_downsample_dataset()
