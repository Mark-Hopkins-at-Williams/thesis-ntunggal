"""
Custom character-level tokenizer.
"""
import os
import json
from transformers import PreTrainedTokenizer
from typing import Dict, List

class CustomCharacterTokenizer(PreTrainedTokenizer):
    
    def __init__(self, vocab=None, **kwargs):
        self.vocab: Dict[str, int] = dict() # map token to id
        super().__init__(**kwargs)        
        self.ids_to_tokens: Dict[int, str] | None = {id: token for token, id in self.vocab.items()} if self.vocab is not None else None
        self.special_tokens: Dict[str: str | None] = {
            'bos_token': None,
            'eos_token': None,
            'unk_token': None,
            'sep_token': None,
            'pad_token': None,
            'cls_token': None,
            'mask_token': None,
        }
        self.pad_token_id: int = None
    
    def train(self, files: List[str], vocab_size: int, min_frequency: int, special_tokens: List[str]):
        """
        Initializes vocabulary.
        files: a list of .txt files
        min_frequency: the number of times a pair should appear in order to be merged
        """
        # Get characters
        char_set = set()
        for file in files:
            with open(file) as f:
                for line in f:
                    char_set.update(set(self.tokenize(line)))
                    if len(char_set) > vocab_size:
                        break
                else:
                    continue
                break
        
        # Add special tokens to vocabulary
        # Manually coded for now
        for i, token in enumerate(special_tokens):
            if token == "<s>":
                self.special_tokens["bos_token"] = token
                self.special_tokens["cls_token"] = token
            elif token == "<pad>":
                self.special_tokens["pad_token"] = token
                self.pad_token_id = i
            elif token == "</s>":
                self.special_tokens["sep_token"] = token
                self.special_tokens["eos_token"] = token
            elif token == "<unk>":
                self.special_tokens["unk_token"] = token
            elif token == "<mask>":
                self.special_tokens["bos_token"] = token
            self.vocab[token] = i

        # Set vocabulary
        char_set_list = list(char_set)
        for i in range(len(special_tokens), min(len(char_set), vocab_size)):
            self.vocab[char_set_list[i]] = i

        self.ids_to_tokens = {id: token for token, id in self.vocab.items()}
    
    def tokenize(self, text: str) -> List:
        """Tokenize using individual characters."""
        return list(text)
    
    def convert_tokens_to_ids(self, tokens: List) -> List:
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
        
    
if __name__ == "__main__":
    VOCAB_SIZE = 100
    
    tokenizer = CustomCharacterTokenizer()
    corpus = "/mnt/storage/ntunggal/test_text.txt"
    
    tokenizer.train(files=[corpus], vocab_size=VOCAB_SIZE, min_frequency=2, special_tokens=[
        "<s>", "<pad>", "</s>", "<unk>", "<mask>",
    ])

    print(f"tokenizer: {tokenizer}")

    vocab = tokenizer.get_vocab()
    print(vocab)