from transformers import RobertaTokenizer, RobertaForSequenceClassification
from transformers import AutoTokenizer, AutoModelForSequenceClassification
from datasets import load_dataset
import evaluate
from transformers import Trainer, TrainingArguments
import numpy as np

import torch

def tokenize_dataset(model_dir):
    # Load tokenizer, AFQMC dataset from CLUE
    dataset = load_dataset("clue", "afqmc")    
    tokenizer = AutoTokenizer.from_pretrained(model_dir)
    
    

    def tokenize_function(example):
        return tokenizer(example['sentence1'], example['sentence2'], truncation=True, padding="max_length", max_length=512)

    # Tokenize the dataset
    tokenized_dataset = dataset.map(tokenize_function, batched=True)
    return tokenized_dataset



def finetune(tokenized_dataset, model_dir, output_dir):
    accuracy_metric = evaluate.load("accuracy")
    
    def compute_metrics(eval_pred):
        logits, labels = eval_pred
        predictions = np.argmax(logits, axis=-1)
        accuracy = accuracy_metric.compute(predictions=predictions, references=labels)
        return accuracy
    
    model = AutoModelForSequenceClassification.from_pretrained(model_dir, num_labels=2)
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
    eval_results = trainer.evaluate()
    print(f"Validation Accuracy: {eval_results['eval_accuracy']}")

if __name__ == "__main__":
    import sys
    model_dir_arg = sys.argv[1]  
    output_dir_arg = sys.argv[2]
    tokenized_dataset = tokenize_dataset(model_dir_arg)
    finetune(tokenized_dataset, model_dir_arg, output_dir_arg)