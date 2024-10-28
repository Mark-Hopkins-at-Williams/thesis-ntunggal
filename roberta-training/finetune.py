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
import torch
import time
from evaluate_clue import tokenize_dataset, evaluate_on_task
from pathlib import Path
import os
import csv

def finetune(tokenized_dataset, model_dir, output_dir, num_labels, tokenizer):
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
        evaluation_strategy="steps",
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
    trainer.save_model(output_dir)
    tokenizer.save_pretrained(output_dir)


def finetune_on_task(model_dir, output_dir, task_name):
    """
    Given a specific task to finetune on, will tokenize the dataset
    and run finetuning on it.
    """
    tokenizer = AutoTokenizer.from_pretrained(model_dir)

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

    finetune(tokenized_dataset, model_dir, output_dir, num_labels, tokenizer)


def finetune_on_tasks(model_dir, output_dir, task_names, note="", log_file=None):
    results = dict()
    start_time = time.time()
    for task in task_names:
        finetune_on_task(model_dir, output_dir, task)
        score = evaluate_on_task(output_dir, output_dir, task)
        results[f'finetune-{task}'] = score
    end_time = time.time()
    results['base_model'] = model_dir
    results['finetuning_time'] = end_time - start_time
    results['notes'] = note
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
    try:
        acquire_lock(filename)  # Acquire the lock before writing
        with open(filename, mode='a', newline='') as file:
            writer = csv.writer(file)
            writer.writerow(data)  # Write a new row
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