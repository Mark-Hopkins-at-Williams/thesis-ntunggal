"""
Finetune a pre-trained model on one of the CLUE benchmarks.
This file takes in three arguments: the model directory, output directory, and benchmark
Current benchmarks available are: tnews, iflytek, cluewsc2020, afqmc, csl, ocnli
"""
from transformers import AutoTokenizer, AutoModelForSequenceClassification
from datasets import load_dataset
import evaluate
from transformers import Trainer, TrainingArguments
import numpy as np
import argparse
import time
from evaluate_clue import tokenize_dataset, evaluate_on_task
from pathlib import Path
import os
import csv
import shutil

TASK_CONFIGS = {
    "tnews": {
        "dataset": ("clue", "tnews"),
        "task_type": "single",
        "num_labels": 15
    },
    "iflytek":{
        "dataset": ("clue", "iflytek"),
        "task_type": "single",
        "num_labels": 119
    },
    "cluewsc2020": {
        "dataset": ("clue", "cluewsc2020"), 
        "task_type": "cluewsc", 
        "num_labels": 2
    },
    "afqmc": {
        "dataset": ("clue", "afqmc"), 
        "task_type": "pair", 
        "num_labels": 2
    },
    "csl": {
        "dataset": ("clue", "csl"), 
        "task_type": "csl", 
        "num_labels": 2
    },
    "ocnli": {
        "dataset": ("clue", "ocnli"), 
        "task_type": "pair", 
        "num_labels": 3
    }
}

def finetune(tokenized_dataset, model_dir, output_dir, num_labels, tokenizer, task_name):
    """
    General function to run the finetuning.
    """
    accuracy_metric = evaluate.load("accuracy")
    
    def compute_metrics(eval_pred):
        logits, labels = eval_pred
        predictions = np.argmax(logits, axis=-1)
        accuracy = accuracy_metric.compute(predictions=predictions, references=labels)
        return accuracy
    
    model = AutoModelForSequenceClassification.from_pretrained(model_dir, num_labels=num_labels)
    training_args = TrainingArguments(
        output_dir=output_dir,
        overwrite_output_dir=True,
        eval_strategy="steps",
        eval_steps=500,
        logging_steps=500,
        learning_rate=2e-5,
        per_device_train_batch_size=32,
        per_device_eval_batch_size=32,
        num_train_epochs=1,
        load_best_model_at_end=True,
        metric_for_best_model="accuracy",
        greater_is_better=True,
    )

    trainer = Trainer(
        model=model,
        args=training_args,
        train_dataset=tokenized_dataset["train"],
        eval_dataset=tokenized_dataset["validation"],
        compute_metrics=compute_metrics
    )

    trainer.train()

    # Save model
    model_save_path = os.path.join(output_dir, task_name)

    if os.path.exists(model_save_path):
        print("Warning: model directory already exists! Deleting old directory")
        shutil.rmtree(model_save_path)
    else:
        os.mkdir(model_save_path)

    trainer.save_model(model_save_path)
    tokenizer.save_pretrained(model_save_path)

    # Remove checkpoint folders
    for filename in os.listdir(output_dir):
        file_path = os.path.join(output_dir, filename)
        if os.path.isdir(file_path) and filename.startswith("checkpoint-"):
            shutil.rmtree(file_path)


def finetune_on_task(model_dir, output_dir, task_name):
    """
    Given a specific task to finetune on, will tokenize the dataset
    and run finetuning on it.
    """
    tokenizer = AutoTokenizer.from_pretrained(model_dir)

    if task_name not in TASK_CONFIGS:
        raise ValueError(f"Unknown task '{task_name}'.")

    task_config = TASK_CONFIGS[task_name]
    dataset = dataset = load_dataset(*task_config["dataset"])
    tokenized_dataset = tokenize_dataset(dataset, tokenizer, task_config["task_type"])
    num_labels = task_config["num_labels"]

    finetune(tokenized_dataset, model_dir, output_dir, num_labels, tokenizer, task_name)


def finetune_on_tasks(base_model_dir, output_dir, task_names, note="", log_file=None):
    """
    Finetunes a given pretrained model on the specified tasks.

    base_model_dir: the directory where the pretrained model is located
    output_dir: the directory where the finetuned model will be located
    """
    results = dict()
    
    # Finetune on each task
    for task in task_names:
        start_time = time.time()

        finetune_on_task(base_model_dir, output_dir, task)
        finetuned_model_dir = os.path.join(output_dir, task)
        score = evaluate_on_task(finetuned_model_dir, finetuned_model_dir, task)
    
        # Write results
        results = {
            'base_model': base_model_dir,
            'finetuning_time': time.time() - start_time,
            'finetune-accuracy': score,
            'task': task,
            'notes': note,
        }
        
        if log_file is not None:
            write_results(log_file, results)
      

def acquire_lock(filename, check_interval=1):
    lock_file = filename + ".lock"
    print(f"waiting to acquire lock for {filename}...")
    while os.path.exists(lock_file):
        time.sleep(check_interval)
    print('...acquired!')
    file_path = Path(lock_file)
    file_path.touch()
    

def release_lock(filename):
    lock_file = filename + ".lock"
    if os.path.exists(lock_file):
        os.remove(lock_file)
    print(f'lock released on: {filename}')
    

def write_results(filename, results):    
    data = [results[key] for key in sorted(results.keys())]  
    headers = sorted(results.keys())
    
    try:
        acquire_lock(filename)
        # If log file doesn't exist, write the headers
        if not os.path.isfile(filename):
            with open(filename, mode='a', newline='') as file:
                writer = csv.writer(file)
                writer.writerow(headers)
        # Write results to csv
        with open(filename, mode='a', newline='') as file:
            writer = csv.writer(file)
            writer.writerow(data)
        release_lock(filename)

    except Exception:
        release_lock(filename)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Finetuning script for NLLB models.")
    parser.add_argument("--model_dir", type=str, required=True, help="Directory where the base model is located.")
    parser.add_argument("--output_dir", type=str, required=True, help="Directory where the finetuned model will be located.")
    parser.add_argument("--tasks", type=str, required=True, help="Tasks on which to finetune (comma separated)")
    parser.add_argument("--note", type=str, help="Notes about experiment (optional)")
    parser.add_argument("--log_file", type=str, required=True, help="Logging file for experiment results.")
    args = parser.parse_args()
    tasklist = args.tasks.split(',')    
    finetune_on_tasks(args.model_dir, args.output_dir, tasklist, args.note, args.log_file)