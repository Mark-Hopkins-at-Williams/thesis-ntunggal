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

import torch
import sys


def tokenize_dataset(dataset, tokenizer, task_type):
    """
    Tokenize dataset for different task types.
    - task_type: 'single' for single sentence tasks
                 'pair' for sentence pair tasks
    """
    if task_type == 'single':
        def tokenize_function(example):
            return tokenizer(example['sentence'], truncation=True, padding="max_length", max_length=512)
    elif task_type == 'cluewsc':
        # cluewsc2020 has field 'text' instead of 'sentence'
        def tokenize_function(example):
            return tokenizer(example['text'], truncation=True, padding="max_length", max_length=512)
    elif task_type == 'csl':
        # csl has field 'abst' instead of 'sentence'
        def tokenize_function(example):
            return tokenizer(example['abst'], truncation=True, padding="max_length", max_length=512)
    elif task_type == 'pair':
        def tokenize_function(example):
            return tokenizer(example['sentence1'], example['sentence2'], truncation=True, padding="max_length", max_length=512)
    
    return dataset.map(tokenize_function, batched=True)



def finetune(tokenized_dataset, model_dir, output_dir, num_labels):
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
        num_train_epochs=3,
    )

    trainer = Trainer(
        model=model,
        args=training_args,
        train_dataset=tokenized_dataset["train"],
        eval_dataset=tokenized_dataset["validation"],
        compute_metrics=compute_metrics
    )

    trainer.train()

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

    finetune(tokenized_dataset, model_dir, output_dir, num_labels)


if __name__ == "__main__":
    model_dir = sys.argv[1]
    output_dir = sys.argv[2]  
    task = sys.argv[3]
    
    finetune_on_task(model_dir, output_dir, task)