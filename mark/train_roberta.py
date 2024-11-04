import argparse
import json
import os
import shutil
import sys
import time
from tokenizers import ByteLevelBPETokenizer
from torch.utils.data import IterableDataset
from transformers import (
    RobertaTokenizer,
    DataCollatorForLanguageModeling,
    Trainer,
    TrainingArguments,
    RobertaForMaskedLM,
    RobertaConfig,
    EarlyStoppingCallback
)

from finetune import write_results


VOCAB_SIZE = 52_000
MAX_INPUT_SEQ_LENGTH = 512


def logger(text):
    sys.stderr.write(text + "\n")
    sys.stderr.flush()


def train_byte_level_tokenizer(text_file, model_dir):
    logger('Training tokenizer...')
    
    if not os.path.exists(model_dir):
        os.mkdir(model_dir)
        tokenizer = ByteLevelBPETokenizer()
        tokenizer.train(files=[text_file], vocab_size=VOCAB_SIZE, min_frequency=2, special_tokens=[
            "<s>", "<pad>", "</s>", "<unk>", "<mask>",
        ])
        tokenizer.save_model(model_dir)
        tokenizer_config = {  # manually creates a tokenizer_config.json for Roberta
            "max_len": 512,
            "do_lower_case": False,  # if your corpus is case-sensitive
        }    
        with open(os.path.join(model_dir, "tokenizer_config.json"), "w") as f:
            json.dump(tokenizer_config, f)
    else:
        logger("...tokenizer directory already exists! Loading pretrained tokenizer.")
            
   
    tokenizer = RobertaTokenizer.from_pretrained(model_dir)
    return tokenizer



class TextIterableDataset(IterableDataset):
    def __init__(self, tokenizer, file_path, block_size):
        """
        Args:
            tokenizer: Tokenizer for encoding text data.
            file_path: Path to the text file.
            block_size: The maximum length of tokenized sequences.
        """
        self.tokenizer = tokenizer
        self.file_path = file_path
        self.block_size = block_size

    def __iter__(self):
        """
        Opens the file and yields each line as a tokenized tensor.
        """
        with open(self.file_path, "r", encoding="utf-8") as f:
            for line in f:
                line = line.strip()
                if len(line) > 0:  # Skip empty lines
                    tokenized = self.tokenizer(
                        line,
                        add_special_tokens=True,
                        truncation=True,
                        max_length=self.block_size,
                        return_tensors="pt"
                    )["input_ids"].squeeze()
                    yield tokenized
                    

                    
def train_roberta(tokenizer, data_dir, output_dir, note, log_file):
    """
    Trains a roberta model. Currently hard-coded to use a byte-level
    tokenizer.

    data_dir: directory where training data is located
    output_dir: directory where pretrained model is saved
    """
    start_time = time.time()

    logger('Loading data...')
    train_dataset = TextIterableDataset(
        tokenizer=tokenizer,
        file_path=os.path.join(data_dir, "train.txt"),
        block_size=128,
    )
    dev_dataset = TextIterableDataset(
        tokenizer=tokenizer,
        file_path=os.path.join(data_dir, "dev.txt"),
        block_size=128,
    )
    data_collator = DataCollatorForLanguageModeling(
        tokenizer=tokenizer, mlm=True, mlm_probability=0.15
    )
    
    logger('Training model...')
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
    training_args = TrainingArguments(
        output_dir=output_dir,
        overwrite_output_dir=True,
        max_steps=500000,
        per_device_train_batch_size=64,
        save_strategy="steps",
        eval_strategy="steps",
        eval_steps=1000,
        save_steps=1000,
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
        eval_dataset=dev_dataset,
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
    tokenizer = train_byte_level_tokenizer(os.path.join(args.data_dir, "train.txt"), args.output_dir)
    train_roberta(tokenizer, args.data_dir, args.output_dir, args.note, args.log_file)