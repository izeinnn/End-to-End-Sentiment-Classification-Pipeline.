from transformers import AutoTokenizer
from src.config import Config

def get_tokenizer(model_checkpoint=Config.MODEL_CHECKPOINT):
    return AutoTokenizer.from_pretrained(model_checkpoint)

def preprocess_datasets(train_dataset, test_dataset, tokenizer, max_length=Config.MAX_LENGTH):
    def tokenize_function(examples):
        return tokenizer(examples["text"], truncation=True, padding="max_length", max_length=max_length)

    print("Tokenizing datasets...")
    tokenized_train = train_dataset.map(tokenize_function, batched=True)
    tokenized_test = test_dataset.map(tokenize_function, batched=True)

    # Keep only tensor-compatible columns
    columns_to_keep = ["input_ids", "attention_mask", "label"]
    tokenized_train = tokenized_train.remove_columns([col for col in tokenized_train.column_names if col not in columns_to_keep])
    tokenized_test = tokenized_test.remove_columns([col for col in tokenized_test.column_names if col not in columns_to_keep])

    # Rename 'label' to 'labels' (required by Hugging Face models)
    tokenized_train = tokenized_train.rename_column("label", "labels")
    tokenized_test = tokenized_test.rename_column("label", "labels")

    # Format datasets to PyTorch tensors
    tokenized_train.set_format("torch")
    tokenized_test.set_format("torch")

    return tokenized_train, tokenized_test
