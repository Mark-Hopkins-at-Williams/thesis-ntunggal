"""
Evaluate a (finetuned) model on a single CLUE benchmark.

Arguments:
--model_dir: Directory of base model.
--tokenizer_dir: Directory of tokenizer (may need to use HuggingFace path).
--task: A CLUE benchmark to evaluate on (one of: tnews, iflytek, cluewsc2020, afqmc, csl, ocnli)
--note: Notes about the experiment (optional)
--log_file: name of file to write logs to (name includes .csv)
"""
import torch
from transformers import AutoTokenizer, AutoModelForSequenceClassification
from datasets import load_dataset
from torch.utils.data import DataLoader
from tqdm import tqdm
import argparse
import utils


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


def evaluate_on_task(model_path, tokenizer_path, task_name, note="", log_file=None):
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

    # Log results
    results = {
        'base_model_path': model_path,
        'tokenizer_path': tokenizer_path,
        'accuracy': accuracy,
        'task': task_name,
        'notes': note,
    }
    
    if log_file is not None:
        utils.write_results(log_file, results)

    return accuracy


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Evaluate a model on a single CLUE benchmark.")
    parser.add_argument("--model_dir", type=str, required=True, help="Directory where the base model is located.")
    parser.add_argument("--tokenizer_dir", type=str, required=True, help="Directory where tokenizer is located.")
    parser.add_argument("--task", type=str, required=True, help="Benchmark or task to evaluate on.")
    parser.add_argument("--note", type=str, help="Notes about experiment (optional)")
    parser.add_argument("--log_file", type=str, required=True, help="Logging file for experiment results.")
    args = parser.parse_args()

    evaluate_on_task(args.model_dir, args.tokenizer_dir, args.task, args.note, args.log_file)
    
