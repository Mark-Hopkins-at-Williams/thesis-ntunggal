from baai import load_baai_data
import json
import os
from os.path import join
import sys
from tokenization import DefaultGPT2Tokenizer
from transformers import Trainer, TrainingArguments
from transformers import Trainer
from transformers import GPT2Config
from transformers import GPT2LMHeadModel
from transformers import TrainingArguments, Trainer


experiment_dir = sys.argv[1]
with open(join(experiment_dir, 'experiment.json')) as reader:
    experiment_config = json.load(reader)

model_config = experiment_config['model']
training_config = experiment_config['training']
checkpoints_dir = join(experiment_dir, "checkpoints")
logging_dir = join(experiment_dir, "logs")
os.makedirs(checkpoints_dir, exist_ok=True)
os.makedirs(logging_dir, exist_ok=True)


config = GPT2Config(
    vocab_size=50257,        # Vocabulary size (this is the default for GPT-2)
    n_positions=model_config['n_positions'],    # Maximum length of input sequences (in tokens)
    n_ctx=model_config['n_ctx'],                # Context size (same as n_positions)
    n_embd=model_config['n_embd'],              # Embedding dimension size
    n_layer=model_config['n_layer'],            # Number of transformer layers
    n_head=model_config['n_head'],              # Number of attention heads
    activation_function=model_config['activation_function'],  # Activation function
    pad_token_id=50256,      # Padding token (this should be updated)
    bos_token_id=50256,      # Beginning-of-sequence token
    eos_token_id=50256,      # End-of-sequence token
)

model = GPT2LMHeadModel(config)
train_dataset, validation_dataset, entropy = load_baai_data()
print("Data loaded, tokenizing now...", flush=True)

# Tokenize the datasets
tokenizer = DefaultGPT2Tokenizer(config.n_positions)
config.pad_token_id = tokenizer.pad_token_id  # Sync the pad_token_id
model.resize_token_embeddings(len(tokenizer))  # Adjust model embeddings
tokenized_train = train_dataset.map(tokenizer.tokenize, batched=True, remove_columns=["text"])
tokenized_validation = validation_dataset.map(tokenizer.tokenize, batched=True, remove_columns=["text"])

num_validation_tokens = 0
for line in tokenized_validation:
    #print(sum(line['attention_mask']))
    num_validation_tokens += sum(line['attention_mask'])
print(f'num tokens: {num_validation_tokens}')

entropy_ratios = []

class CustomTrainer(Trainer):
    
    def evaluate(self, eval_dataset=None, ignore_keys=None, metric_key_prefix="eval"):
        eval_results = super().evaluate(eval_dataset, ignore_keys, metric_key_prefix)
        loss_key = metric_key_prefix + "_loss"
        loss = eval_results[loss_key]
        entropy_ratio = (loss * num_validation_tokens) / entropy
        entropy_ratios.append(entropy_ratio)
        eval_results[metric_key_prefix + "_entropy_ratio"] = entropy_ratio      
        print(eval_results)  
        return eval_results


training_args = TrainingArguments(
    output_dir=checkpoints_dir,                                         # Directory for storing model checkpoints
    eval_strategy="steps",       
    logging_dir=logging_dir,              
    eval_steps=training_config['eval_steps'],
    max_steps=training_config['max_steps'],
    per_device_train_batch_size=training_config['batch_size'],     # Batch size for training
    per_device_eval_batch_size=training_config['batch_size'],      # Batch size for evaluation
    logging_steps=training_config['eval_steps'],                 
    save_steps=training_config['eval_steps'],                   
    learning_rate=training_config['learning_rate'],                
    weight_decay=training_config['weight_decay'],                 
    warmup_steps=training_config['warmup_steps'],                  # Warmup steps for learning rate scheduling
    max_grad_norm=training_config['max_grad_norm'],                # Gradient clipping
    report_to="tensorboard",           
)

trainer = CustomTrainer(
    model=model,                        
    args=training_args,                 
    train_dataset=tokenized_train,      
    eval_dataset=tokenized_validation,  
    tokenizer=tokenizer.tokenizer,             
)
trainer.train()

with open(join(experiment_dir, 'ratios.txt'), 'w') as writer:
    steps_so_far = training_config['eval_steps']
    for ratio in entropy_ratios:
        writer.write(f'{steps_so_far},{ratio}\n')
        steps_so_far += training_config['eval_steps']
