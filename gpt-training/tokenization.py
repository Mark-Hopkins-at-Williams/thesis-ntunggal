from transformers import GPT2TokenizerFast
from transformers import PreTrainedTokenizer
import os
import json


class DefaultGPT2Tokenizer:

    def __init__(self, n_positions):
        self.tokenizer = GPT2TokenizerFast.from_pretrained("gpt2")
        self.tokenizer.add_special_tokens({'pad_token': '[PAD]'})
        self.pad_token_id = self.tokenizer.pad_token_id 
        self.n_positions = n_positions
        
    def __len__(self):
        return len(self.tokenizer)

    def tokenize(self, examples):
        tokenized = self.tokenizer(
            examples["text"], 
            padding="max_length", 
            truncation=True, 
            max_length=self.n_positions
        )
        tokenized["labels"] = tokenized["input_ids"].copy()
        return tokenized


class CharacterTokenizer(PreTrainedTokenizer):
    
    def __init__(self, n_positions, **kwargs):
        self.vocab: dict[str, int] = dict() # map token to id
        super().__init__(**kwargs)        
        self.ids_to_tokens: dict[int, str] | None = {id: token for token, id in self.vocab.items()} if self.vocab is not None else None
        self.special_tokens: dict[str: str | None] = {
            'bos_token': None,
            'eos_token': None,
            'unk_token': None,
            'sep_token': None,
            'pad_token': None,
            'cls_token': None,
            'mask_token': None,
        }
        self.pad_token_id1: int = None
        self.bos_token_id1: int = None
        self.eos_token_id1: int = None
        self.n_positions: int = n_positions

    def train(self, data, vocab_size: int, min_frequency: int, special_tokens: list[str], text_field="text"):
        """
        Initializes vocabulary.
        
        Args:
            data: Dataset object containing text to train
            vocab_size: max vocab size
            min_frequency: the number of times a pair should appear in order to be merged
            special_tokens: list of special tokens
            text_field: field name of Dataset containing the text to train
        """
        # Get characters
        char_set = set()
        for sample in data:
            assert text_field in sample
            text = sample.get(text_field, "")
            char_set.update(set(text))
            if len(char_set) > vocab_size:
                break
                
        # Ensure vocab does not exceed vocab_size
        char_set = list(char_set)[:vocab_size - len(special_tokens)]

        # Add special tokens to vocabulary
        for i, token in enumerate(special_tokens):
            if token == "<s>":
                self.special_tokens["bos_token"] = token
                self.special_tokens["cls_token"] = token
                self.bos_token_id1 = i
            elif token == "<pad>":
                self.special_tokens["pad_token"] = token
                self.pad_token_id1 = i
            elif token == "</s>":
                self.special_tokens["sep_token"] = token
                self.special_tokens["eos_token"] = token
                self.eos_token_id1 = i
            elif token == "<unk>":
                self.special_tokens["unk_token"] = token
            elif token == "<mask>":
                self.special_tokens["bos_token"] = token
            self.vocab[token] = i

        # Set rest of vocabulary
        for i, token in enumerate(char_set, start=len(special_tokens)):
            self.vocab[token] = i

        self.ids_to_tokens = {id: token for token, id in self.vocab.items()}

    def tokenize(self, texts: list[str]) -> dict:
        """Tokenize a batch of texts using characters in vocab."""
        
        if isinstance(texts, str):
            return [self.vocab.get(char, self.special_tokens["unk_token"]) for char in text]
        
        input_ids = []
        attention_mask = []

        for text in texts:
            ids = [self.vocab.get(char, self.special_tokens["unk_token"]) for char in text]
            input_ids.append(ids)
            attention_mask.append([1 * len(ids)])
            
        return {"input_ids": input_ids, "attention_mask": attention_mask}
    
    def convert_tokens_to_ids(self, tokens: list) -> list:
        """Convert characters (tokens) to IDs."""
        return [self.vocab.get(token, self.vocab["<unk>"]) for token in tokens]
    
    def convert_ids_to_tokens(self, ids):
        """Convert IDs back to characters."""
        if isinstance(ids, int):  # Handle single integer ID
            return self.ids_to_tokens.get(ids, "<unk>")
        return [self.ids_to_tokens.get(id, "<unk>") for id in ids]
    
    def save_pretrained(self, save_directory):
        """Save the vocabulary and configuration."""
        if not os.path.exists(save_directory):
            os.makedirs(save_directory)
        # Save the vocabulary
        with open(os.path.join(save_directory, "vocab.json"), "w") as f:
            json.dump(self.vocab, f)
        # Save the tokenizer configuration
        tokenizer_config = {"max_len": 512}
        with open(os.path.join(save_directory, "tokenizer_config.json"), "w") as f:
            json.dump(tokenizer_config, f)

    def save_model(self, save_directory):
        self.save_pretrained(save_directory)
    
    @classmethod
    def from_pretrained(cls, pretrained_model_name_or_path, *inputs, **kwargs):
        """Load vocabulary and configuration from a directory."""
        with open(os.path.join(pretrained_model_name_or_path, "vocab.json"), "r") as f:
            vocab = json.load(f)
        return cls(vocab, *inputs, **kwargs)
    
    def build_inputs_with_special_tokens(self, token_ids):
        """Add special tokens to a sequence."""
        return [self.vocab["<s>"]] + token_ids + [self.vocab["</s>"]]
    
    def get_vocab(self):
        """Return the vocabulary dictionary."""
        return self.vocab
    
    def __len__(self):
        return len(self.vocab) if self.vocab is not None else 0
    
    def __str__(self):
        info = {
            'tokenizer': 'CustomCharacterTokenizer',
            'vocab_size': len(self.vocab),
        }
        return str(info)
