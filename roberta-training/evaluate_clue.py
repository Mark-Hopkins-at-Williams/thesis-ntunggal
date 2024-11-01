"""
Evaluate a trained model on CLUE benchmarks.
This file takes in three arguments: model path, tokenizer path, and the task to evaluate on
Available tasks are as follows: tnews, iflytek, cluewsc2020, afqmc, csl, ocnli
"""
import torch
from transformers import AutoTokenizer, AutoModelForSequenceClassification
from datasets import load_dataset
from torch.utils.data import DataLoader
from tqdm import tqdm
import sys


def tokenize_dataset(dataset, tokenizer, task_type: str):
    """
    Tokenize dataset for different task types.
    - task_type: 'single' for single sentence tasks
                 'pair' for sentence pair tasks
    """
    field_map = {
        'single': ['sentence'],
        'cluewsc': ['text'], # cluewsc2020 has field 'text' instead of 'sentence'
        'csl': ['abst'], # csl has field 'abst' instead of 'sentence'
        'pair': ['sentence1', 'sentence2']
    }
    
    if task_type not in field_map:
        raise ValueError(f"Unknown task_type '{task_type}'.")

    fields = field_map[task_type]
    
    # Define tokenization function
    def tokenize_function(example):
        return tokenizer(*[example[field] for field in fields], truncation=True, padding="max_length", max_length=512)

    return dataset.map(tokenize_function, batched=True)


def evaluate_on_task(model_path, tokenizer_path, task_name):
    """
    Evaluates the model on the given task.
    """
    tokenizer = AutoTokenizer.from_pretrained(tokenizer_path)

    # Load and tokenize dataset
    if task_name == "tnews":
        dataset = load_dataset("clue", "tnews")
        tokenized_dataset = tokenize_dataset(dataset, tokenizer, 'single')
        num_labels = 15
    elif task_name == "iflytek":
        dataset = load_dataset("clue", "iflytek")
        tokenized_dataset = tokenize_dataset(dataset, tokenizer, 'single')
        num_labels = 119
    elif task_name == "cluewsc2020":
        dataset = load_dataset("clue", "cluewsc2020")
        tokenized_dataset = tokenize_dataset(dataset, tokenizer, 'cluewsc')
        num_labels = 2
    elif task_name == "afqmc":
        dataset = load_dataset("clue", "afqmc")
        tokenized_dataset = tokenize_dataset(dataset, tokenizer, 'pair')
        num_labels = 2
    elif task_name == "csl":
        dataset = load_dataset("clue", "csl")
        tokenized_dataset = tokenize_dataset(dataset, tokenizer, 'csl')
        num_labels = 2
    elif task_name == "ocnli":
        dataset = load_dataset("clue", "ocnli")
        tokenized_dataset = tokenize_dataset(dataset, tokenizer, 'pair')
        num_labels = 3
    else:
        raise ValueError("Unknown task name.")
    
    model = AutoModelForSequenceClassification.from_pretrained(model_path, num_labels=num_labels)

    # Create Dataloaders
    batch_size = 16
    tokenized_dataset.set_format("torch", columns=["input_ids", "attention_mask", "label"])
    test_dataloader = DataLoader(tokenized_dataset['validation'], batch_size=batch_size)

    # Evaluation function
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model.to(device)

    def evaluate(model, dataloader):
        model.eval()
        total, correct = 0, 0
        with torch.no_grad():
            for batch in tqdm(dataloader):
                input_ids = batch['input_ids'].to(device)
                attention_mask = batch['attention_mask'].to(device)
                labels = batch['label'].to(device)

                outputs = model(input_ids, attention_mask=attention_mask)
                predictions = torch.argmax(outputs.logits, dim=-1)
                correct += (predictions == labels).sum().item()
                total += labels.size(0)

        return correct / total

    # Run evaluation
    accuracy = evaluate(model, test_dataloader)
    print(f"Accuracy on {task_name} validation set: {accuracy}")
    return accuracy


if __name__ == "__main__":
    model_path = sys.argv[1]
    tokenizer_path = sys.argv[2]
    task = sys.argv[3]

    evaluate_on_task(model_path, tokenizer_path, task)
    
