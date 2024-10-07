from transformers import RobertaTokenizer, RobertaForSequenceClassification
from datasets import load_dataset
from transformers import Trainer, TrainingArguments

#model_dir = "./baidu-model"
#output_dir = "./baidu-afqmc-finetuned"

def tokenize_dataset(model_dir):
    # Load tokenizer, AFQMC dataset from CLUE
    tokenizer = RobertaTokenizer.from_pretrained(model_dir)
    dataset = load_dataset("clue", "afqmc")

    def tokenize_function(example):
        return tokenizer(example['sentence1'], example['sentence2'], truncation=True, padding="max_length", max_length=512)

    # Tokenize the dataset
    tokenized_dataset = dataset.map(tokenize_function, batched=True)
    return tokenized_dataset


def finetune(tokenized_dataset, model_dir, output_dir):
    model = RobertaForSequenceClassification.from_pretrained(model_dir, num_labels=2)
    training_args = TrainingArguments(
        output_dir=output_dir,
        overwrite_output_dir=True,
        evaluation_strategy="epoch",
        learning_rate=2e-5,
        per_device_train_batch_size=64,
        per_device_eval_batch_size=64,
        num_train_epochs=3,
    )

    trainer = Trainer(
        model=model,
        args=training_args,
        train_dataset=tokenized_dataset["train"],
        eval_dataset=tokenized_dataset["validation"],
    )

    trainer.train()


if __name__ == "__main__":
    import sys
    model_dir_arg = sys.argv[1]  
    output_dir_arg = sys.argv[2]
    tokenized_dataset = tokenize_dataset(model_dir_arg)
    finetune(tokenized_dataset, model_dir_arg, output_dir_arg)