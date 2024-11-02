from tokenizers import ByteLevelBPETokenizer
from transformers import (
    RobertaTokenizer,
    LineByLineTextDataset,
    DataCollatorForLanguageModeling,
    Trainer,
    TrainingArguments,
    RobertaForMaskedLM,
    RobertaConfig,
    EarlyStoppingCallback
)
import os
import json
import time
import random
import argparse
import shutil
from finetune import write_results


VOCAB_SIZE = 52_000
MAX_INPUT_SEQ_LENGTH = 512
TRAIN_SPLIT_RATIO = 0.9


def train_byte_level_tokenizer(text_file, model_dir):
    tokenizer = ByteLevelBPETokenizer()
    tokenizer.train(files=[text_file], vocab_size=VOCAB_SIZE, min_frequency=2, special_tokens=[
        "<s>", "<pad>", "</s>", "<unk>", "<mask>",
    ])
    
    
    if not os.path.exists(model_dir):
        os.mkdir(model_dir)
    else:
        print("Warning: model directory already exists!")    
    tokenizer.save_model(model_dir)
    
    # Manually create a tokenizer_config.json for Roberta
    tokenizer_config = {
        "max_len": 512,
        "do_lower_case": False,  # if your corpus is case-sensitive
    }
    
    with open(os.path.join(model_dir, "tokenizer_config.json"), "w") as f:
        json.dump(tokenizer_config, f)

    tokenizer = RobertaTokenizer.from_pretrained(model_dir)
    return tokenizer


def split_data(text_file, train_file, val_file, split_ratio=TRAIN_SPLIT_RATIO):
    """
    Splits the given data at data_dir into training and validation set, to get
    validation scores.
    """
    with open(text_file, 'r', encoding='utf-8') as f:
        lines = f.readlines()

    random.shuffle(lines)
    split_idx = int(len(lines) * split_ratio)
    
    with open(train_file, 'w', encoding='utf-8') as f:
        f.writelines(lines[:split_idx])
        
    with open(val_file, 'w', encoding='utf-8') as f:
        f.writelines(lines[split_idx:])


def train_roberta(tokenizer, data_dir, output_dir, note, log_file):
    """
    Trains a roberta model. Currently hard-coded to use a byte-level
    tokenizer.

    data_dir: directory where training data is located
    output_dir: directory where pretrained model is saved
    """
    start_time = time.time()

    # Prepare paths for split data
    train_file = os.path.join(output_dir, "train.txt")
    val_file = os.path.join(output_dir, "val.txt")
    
    # Split data into train and validation sets
    split_data(data_dir, train_file, val_file)
    
    max_pos_embeddings = MAX_INPUT_SEQ_LENGTH + 2 # adding 2 because of <s> and </s>
    num_attn_heads = 12
    num_hidden_layers = 6
    
    config = RobertaConfig(
        vocab_size=len(tokenizer),
        max_position_embeddings=max_pos_embeddings,
        num_attention_heads=num_attn_heads,
        num_hidden_layers=num_hidden_layers,
        type_vocab_size=1,
    )

    model = RobertaForMaskedLM(config=config)

    # Load training and validation datasets
    train_dataset = LineByLineTextDataset(
        tokenizer=tokenizer,
        file_path=train_file,
        block_size=128,
    )
    val_dataset = LineByLineTextDataset(
        tokenizer=tokenizer,
        file_path=val_file,
        block_size=128,
    )

    data_collator = DataCollatorForLanguageModeling(
        tokenizer=tokenizer, mlm=True, mlm_probability=0.15
    )

    training_args = TrainingArguments(
        output_dir=output_dir,
        overwrite_output_dir=True,
        num_train_epochs=4,
        per_device_train_batch_size=64,
        save_strategy="steps",
        eval_strategy="steps",
        eval_steps=50,
        save_steps=50,
        prediction_loss_only=True,
        load_best_model_at_end=True,
        metric_for_best_model="eval_loss",
        greater_is_better=False,
    )

    early_stopping_callback = EarlyStoppingCallback(
        early_stopping_patience=3,
        early_stopping_threshold=0.01
    )

    trainer = Trainer(
        model=model,
        args=training_args,
        data_collator=data_collator,
        train_dataset=train_dataset,
        eval_dataset=val_dataset,
        callbacks=[early_stopping_callback],
    )

    trainer.train()

    # Save best model
    trainer.save_model(output_dir)
    tokenizer.save_pretrained(output_dir)

    # Remove checkpoint folders
    for filename in os.listdir(output_dir):
        file_path = os.path.join(output_dir, filename)
        if os.path.isdir(file_path) and filename.startswith("checkpoint-"):
            shutil.rmtree(file_path)

    # Evaluate
    eval_results = trainer.evaluate()

    # Write results to csv
    results = {
        'model_type': 'Roberta',
        'tokenizer': tokenizer,
        'dataset': data_dir,
        'validation_loss': eval_results['eval_loss'],
        'max_pos_embeddings': max_pos_embeddings,
        'num_attention_heads': num_attn_heads,
        'num_hidden_layers': num_hidden_layers,
        'pretraining_time': time.time() - start_time,
        'notes': note,
    }

    if log_file is not None:
        write_results(log_file, results)
    
    
if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Pretraining script for NLLB models.")
    parser.add_argument("--data_dir", type=str, required=True, help="Directory where the pretraining data is located.")
    parser.add_argument("--output_dir", type=str, required=True, help="Directory where the pretrained model will be saved.")
    parser.add_argument("--note", type=str, help="Notes about experiment (optional)")
    parser.add_argument("--log_file", type=str, required=True, help="Logging file for experiment results.")
    args = parser.parse_args()
    
    tokenizer = train_byte_level_tokenizer(args.data_dir, args.output_dir)
    train_roberta(tokenizer, args.data_dir, args.output_dir, args.note, args.log_file)