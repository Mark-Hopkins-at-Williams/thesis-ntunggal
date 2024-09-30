from tokenizers import ByteLevelBPETokenizer
from transformers import RobertaTokenizerFast, BertTokenizerFast
from transformers import RobertaTokenizer, BertTokenizer
from transformers import AutoTokenizer
from transformers import LineByLineTextDataset
from transformers import DataCollatorForLanguageModeling
from transformers import Trainer, TrainingArguments
from transformers import RobertaForMaskedLM
from transformers import RobertaConfig
import os
import json


VOCAB_SIZE = 52_000
MAX_INPUT_SEQ_LENGTH = 512
#TEXT_FILE = 'data/eo.txt'
#MODEL_DIR = './esperberto'


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

def train_roberta(tokenizer, text_file, output_dir):
    config = RobertaConfig(
        vocab_size=len(tokenizer),
        max_position_embeddings=MAX_INPUT_SEQ_LENGTH + 2, # adding 2 because of <s> and </s>
        num_attention_heads=12,
        num_hidden_layers=6,
        type_vocab_size=1,
    )
    model = RobertaForMaskedLM(config=config)
    print(f'model parameters: {model.num_parameters()}')
    dataset = LineByLineTextDataset(
        tokenizer=tokenizer,
        file_path=text_file,
        block_size=128,
    )
    data_collator = DataCollatorForLanguageModeling(
        tokenizer=tokenizer, mlm=True, mlm_probability=0.15
    )
    training_args = TrainingArguments(
        output_dir=output_dir,
        overwrite_output_dir=True,
        num_train_epochs=1,
        per_device_train_batch_size=64,
        save_steps=10_000,
        save_total_limit=2,
        prediction_loss_only=True,
    )
    trainer = Trainer(
        model=model,
        args=training_args,
        data_collator=data_collator,
        train_dataset=dataset,
    )
    trainer.train()
    
    
if __name__ == "__main__":
    import sys
    text_file_arg = sys.argv[1]
    model_dir_arg = sys.argv[2]    
    my_tokenizer = train_byte_level_tokenizer(text_file_arg, model_dir_arg)
    train_roberta(my_tokenizer, text_file_arg, model_dir_arg)