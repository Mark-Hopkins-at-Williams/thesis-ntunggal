from baai import load_baai_data
import json
import os
from os.path import join
import sys
from tokenization import (CharacterTokenizer,
                          SubwordBPETokenizer,
                          ByteTokenizer,
                          ByteBPETokenizer,
                          ChineseBPETokenizer,
                          RepackagedByteTokenizer)
from transformers import Trainer, TrainingArguments
from transformers import Trainer
from transformers import GPT2Config
from transformers import GPT2LMHeadModel
from transformers import TrainingArguments, Trainer
from collections import defaultdict

TOKENIZERS = {
    "CharacterTokenizer": CharacterTokenizer,
    "SubwordBPETokenizer": SubwordBPETokenizer,
    "ByteTokenizer": ByteTokenizer,
    "ByteBPETokenizer": ByteBPETokenizer,
    "ChineseBPETokenizer": ChineseBPETokenizer,
    "RepackagedByteTokenizer": RepackagedByteTokenizer,
}

# Load configs
experiment_dir = sys.argv[1]
with open(join(experiment_dir, 'experiment.json')) as reader:
    experiment_config = json.load(reader)

tokenizer_config = experiment_config['tokenizer']
model_config = experiment_config['model']
training_config = experiment_config['training']

Tokenizer = TOKENIZERS[tokenizer_config['name']]
save_directory = tokenizer_config['tokenizer_files_dir']
print(f"Tokenizer: {tokenizer_config['name']}", flush=True)
if tokenizer_config['name'] in ["ChineseBPETokenizer", "RepackagedByteTokenizer"]:
    print(f"Input method: {tokenizer_config['input_method']}", flush=True)

checkpoints_dir = join(experiment_dir, "checkpoints")
logging_dir = join(experiment_dir, "logs")
os.makedirs(checkpoints_dir, exist_ok=True)
os.makedirs(logging_dir, exist_ok=True)

# Load dataset
print("Loading dataset...", flush=True)
train_dataset, validation_dataset, entropy = load_baai_data()

# Create vocab if needed
if save_directory != "" and not os.path.exists(save_directory):
    print("Creating vocab...", flush=True)
    Tokenizer.create_vocab_from_config(train_dataset, tokenizer_config)

# Instantiate tokenizer
print("Creating tokenizer...", flush=True)
tokenizer = Tokenizer.from_config(tokenizer_config)

# Log tokenizer info
print(f"len(tokenizer): {len(tokenizer)}")
print(f"tokenizer.unk_token_id: {tokenizer.unk_token_id}")
print(f"tokenizer.bos_token_id: {tokenizer.bos_token_id}")
print(f"tokenizer.eos_token_id: {tokenizer.eos_token_id}")
print(f"tokenizer.pad_token_id: {tokenizer.pad_token_id}")

print(f"tokenizer.tokenize(你好,最近怎么样？): {tokenizer.tokenize('你好,最近怎么样？')}")
#print(f"tokenizer.remapped_utf8: {tokenizer.remapped_utf8}")

# TODO: currently testing wubi and cangjie
exit()

config = GPT2Config(
    vocab_size=len(tokenizer),        # Vocabulary size
    n_positions=model_config['n_positions'],    # Maximum length of input sequences (in tokens)
    n_ctx=model_config['n_ctx'],                # Context size (same as n_positions)
    n_embd=model_config['n_embd'],              # Embedding dimension size
    n_layer=model_config['n_layer'],            # Number of transformer layers
    n_head=model_config['n_head'],              # Number of attention heads
    activation_function=model_config['activation_function'],  # Activation function
    pad_token_id=tokenizer.pad_token_id,      # Padding token (this should be updated)
    bos_token_id=tokenizer.bos_token_id,      # Beginning-of-sequence token
    eos_token_id=tokenizer.eos_token_id,      # End-of-sequence token
)

# Tokenize datasets
print("Tokenizing datasets...", flush=True)
model = GPT2LMHeadModel(config)
config.pad_token_id = tokenizer.pad_token_id  # Sync the pad_token_id
model.resize_token_embeddings(len(tokenizer))  # Adjust model embeddings
tokenized_train = train_dataset.map(tokenizer.tokenize_batch, batched=True, remove_columns=["text"])
tokenized_validation = validation_dataset.map(tokenizer.tokenize_batch, batched=True, remove_columns=["text"])
print(f"total model parameters: {model.num_parameters()}")

num_validation_tokens = 0
for line in tokenized_validation:
    #print(sum(line['attention_mask']))
    num_validation_tokens += sum(line['attention_mask'])
print(f'num tokens: {num_validation_tokens}')

entropy_ratios = []

print("Training...", flush=True)
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
    output_dir=checkpoints_dir,                                    # Directory for storing model checkpoints
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
    tokenizer=tokenizer,             
)
trainer.train()

# Log parameter counts
param_counts = defaultdict(int)
for name, param in model.named_parameters():
    param_type = "Other"
    
    if "wte" in name:
        param_type = "Word Embeddings"
    elif "wpe" in name:
        param_type = "Position Embeddings"
    elif "ln" in name:
        param_type = "Layer Normalization"
    elif "mlp" in name or "c_fc" in name or "c_proj" in name:
        param_type = "Feed-Forward Network"
    elif "attn" in name:
        param_type = "Self-Attention"
    elif "lm_head" in name:
        param_type = "Final Output Layer"
    
    param_counts[param_type] += param.numel()

with open(join(experiment_dir, 'parameter_counts.txt'), 'w') as writer:
    for name, count in param_counts.items():
        writer.write(f'{name}: {count}\n')
    writer.write(f'Total parameter count: {model.num_parameters()}\n')

# Log ratios
with open(join(experiment_dir, 'ratios.txt'), 'w') as writer:
    steps_so_far = training_config['eval_steps']
    for ratio in entropy_ratios:
        writer.write(f'{steps_so_far},{ratio}\n')
        steps_so_far += training_config['eval_steps']
