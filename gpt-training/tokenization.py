import os
import json
from collections import Counter
from typing import Optional, Tuple
from transformers import GPT2TokenizerFast
from transformers import AddedToken, PreTrainedTokenizer, logging

logger = logging.get_logger(__name__)

VOCAB_FILES_NAMES = {
    "vocab_file": "vocab.json",
    "merges_file": "merges.txt",
}

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
    """
    Construct a character-level tokenizer.

    Args:
        n_positions (`int`):
            Maximum length of input sequences (in tokens).
        max_vocab_size (`int`):
            Upper limit of vocabulary size.
        vocab_file (`str`):
            Path to the vocabulary file.
        train_file (`str`):
            Path to training data file.
        errors (`str`, *optional*, defaults to `"replace"`):
            Paradigm to follow when decoding bytes to UTF-8. See
            [bytes.decode](https://docs.python.org/3/library/stdtypes.html#bytes.decode) for more information.
        unk_token (`str`, *optional*, defaults to `"<|endoftext|>"`):
            The unknown token. A token that is not in the vocabulary cannot be converted to an ID and is set to be this
            token instead.
        bos_token (`str`, *optional*, defaults to `"<|endoftext|>"`):
            The beginning of sequence token.
        eos_token (`str`, *optional*, defaults to `"<|endoftext|>"`):
            The end of sequence token.
        pad_token (`str`, *optional*):
            The token used for padding, for example when batching sequences of different lengths.
        add_bos_token (`bool`, *optional*, defaults to `False`):
            Whether or not to add an initial beginning of sentence token to the input. This allows to treat the leading
            word just as any other word.
    """
    
    def __init__(
        self,
        n_positions,
        max_vocab_size=None,
        vocab_file=None,
        train_file=None,
        errors="replace",
        unk_token="<|endoftext|>",
        bos_token="<|endoftext|>",
        eos_token="<|endoftext|>",
        pad_token=None,
        add_bos_token=False,
        **kwargs,
    ):
        bos_token = AddedToken(bos_token, lstrip=False, rstrip=False) if isinstance(bos_token, str) else bos_token
        eos_token = AddedToken(eos_token, lstrip=False, rstrip=False) if isinstance(eos_token, str) else eos_token
        unk_token = AddedToken(unk_token, lstrip=False, rstrip=False) if isinstance(unk_token, str) else unk_token
        pad_token = AddedToken(pad_token, lstrip=False, rstrip=False) if isinstance(pad_token, str) else pad_token

        self.add_bos_token = add_bos_token
        
        self.n_positions = n_positions
        self.max_vocab_size = max_vocab_size

        # Load vocab if it exists, otherwise train tokenizer
        if vocab_file is not None:
            print("Loading vocab...", flush=True)
            with open(vocab_file, encoding="utf-8") as vocab_handle:
                self.encoder = json.load(vocab_handle)
        elif train_file is not None:
            print("Training vocab...", flush=True)
            self.encoder = self._train(train_file, max_vocab_size)
        else:
            raise ValueError("CharacterTokenizer needs a vocab file or train file.")
        
        self.decoder = {v: k for k, v in self.encoder.items()}
        self.errors = errors  # how to handle errors in decoding
        self.cache = {}

        super().__init__(
            errors=errors,
            unk_token=unk_token,
            bos_token=bos_token,
            eos_token=eos_token,
            pad_token=pad_token,
            **kwargs,
        )
        print("CharacterTokenizer init finished.", flush=True)

    @property
    def vocab_size(self):
        return len(self.encoder)

    def get_vocab(self):
        return dict(self.encoder, **self.added_tokens_encoder)
    
    def _train(self, dataset, max_vocab_size=None, text_field="text"):
        print("_train method called...", flush=True)
        char_counter = Counter()
        for example in dataset:
            text = example[text_field]
            char_counter.update(text)

        # Assign IDs to special tokens
        current_id = 0
        for token in self.special_tokens_map:
            self.encoder[token] = current_id
            self.decoder[current_id] = token
            current_id += 1

        # Assign IDs to characters
        for char, _ in char_counter.most_common(max_vocab_size):
            if char not in self.encoder:
                self.encoder[char] = current_id
                self.decoder[current_id] = char
                current_id += 1
        print(f"_train method finished. vocab size: {len(self.encoder)}", flush=True)

    def _tokenize(self, text):
        """Tokenize a string."""
        return [char if char in self.encoder else self.unk_token for char in text]

    def _convert_token_to_id(self, token):
        """Converts a token (str) in an id using the vocab."""
        return self.encoder.get(token, self.encoder.get(self.unk_token))

    def _convert_id_to_token(self, index):
        """Converts an index (integer) in a token (str) using the vocab."""
        return self.decoder.get(index)
    
    def build_inputs_with_special_tokens(self, token_ids_0, token_ids_1=None):
        if self.add_bos_token:
            bos_token_ids = [self.bos_token_id]
        else:
            bos_token_ids = []

        output = bos_token_ids + token_ids_0

        if token_ids_1 is None:
            return output

        return output + bos_token_ids + token_ids_1

    def save_vocabulary(self, save_directory: str, filename_prefix: Optional[str] = None) -> Tuple[str]:
        if not os.path.isdir(save_directory):
            logger.error(f"Vocabulary path ({save_directory}) should be a directory")
            return
        
        vocab_file = os.path.join(
            save_directory, (filename_prefix + "-" if filename_prefix else "") + VOCAB_FILES_NAMES["vocab_file"]
        )

        with open(vocab_file, "w", encoding="utf-8") as f:
            f.write(json.dumps(self.encoder, indent=2, sort_keys=True, ensure_ascii=False) + "\n")

        return (vocab_file,)
    
    # def __call__(self, text, **kwargs):
    #     tokens = self.tokenize(text)
    #     token_ids = self.convert_tokens_to_ids(tokens)
    #     attention_mask = [1] * len(token_ids)
    #     return {"input_ids": token_ids, "attention_mask": attention_mask}