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
        vocab_file (`str`):
            Path to the vocabulary file.
        n_positions (`int`):
            Maximum length of input sequences (in tokens).
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
        vocab_file,
        n_positions,
        errors="replace",
        unk_token="<|UNK|>",
        bos_token="<|BOS|>",
        eos_token="<|EOS|>",
        pad_token="<|PAD|>",
        add_bos_token=False,
        **kwargs,
    ):
        bos_token = AddedToken(bos_token, lstrip=False, rstrip=False)
        eos_token = AddedToken(eos_token, lstrip=False, rstrip=False)
        unk_token = AddedToken(unk_token, lstrip=False, rstrip=False)
        pad_token = AddedToken(pad_token, lstrip=False, rstrip=False)
        
        self.add_bos_token = add_bos_token
        self.n_positions = n_positions

        # Load vocab
        with open(vocab_file, encoding="utf-8") as vocab_handle:
            self.encoder = json.load(vocab_handle)
        self.decoder = {v: k for k, v in self.encoder.items()}      
        self.errors = errors  # how to handle errors in decoding

        super().__init__(
            errors=errors,
            unk_token=unk_token,
            bos_token=bos_token,
            eos_token=eos_token,
            pad_token=pad_token,
            **kwargs,
        )

    @classmethod
    def create_vocab(cls, train_dataset, file_path, special_tokens=[], text_field="text"):
        """
        Creates a vocab.json file from a given training dataset.

        Args:
            train_dataset: Dataset from which to create vocab from.
            file_path: path of vocab.json file to create.
            special_tokens: list of special tokens.
            text_field: the column header of the dataset containing text to train on. 
        """
        char_counter = Counter()
        for i, example in enumerate(train_dataset, start=1):
            text = example[text_field]
            char_counter.update(text)
            # Stop at 1 million examples
            if i >= 1000000:
                break

        # Add special tokens to vocab
        vocab = {}
        current_id = 0
        for token in special_tokens:
            if token not in vocab:
                vocab[token] = current_id
                current_id += 1

        # Add regular tokens to vocab
        for char, _ in char_counter.most_common():
            if char not in vocab:
                vocab[char] = current_id
                current_id += 1
        
        os.makedirs(os.path.dirname(file_path), exist_ok=True)
        with open(file_path, "w", encoding="utf-8") as f:
            json.dump(vocab, f, indent=None, ensure_ascii=False, separators=(",", ":"))
    
    @property
    def vocab_size(self):
        return len(self.encoder)

    def get_vocab(self):
        return dict(self.encoder, **self.added_tokens_encoder)

    def _tokenize(self, text):
        """Tokenize a string."""
        return [char if char in self.encoder else self.unk_token for char in text]
    
    def tokenize_batch(self, examples):
        if isinstance(examples, str):
            texts = [examples]
        else:
            texts = examples["text"]
        tokenized = self(
            texts, 
            padding="max_length", 
            truncation=True, 
            max_length=self.n_positions, 
            return_tensors=None
        )
        tokenized["labels"] = tokenized["input_ids"].copy()
        return tokenized

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


class SubwordBPETokenizer(PreTrainedTokenizer):
    """
    Construct a subword BPE tokenizer.
    """
    
    def __init__(
        self,
        vocab_file,
        merges_file,
        n_positions,
        errors="replace",
        unk_token="<|UNK|>",
        bos_token="<|BOS|>",
        eos_token="<|EOS|>",
        pad_token="<|PAD|>",
        add_bos_token=False,
        **kwargs,
    ):
        bos_token = AddedToken(bos_token, lstrip=False, rstrip=False)
        eos_token = AddedToken(eos_token, lstrip=False, rstrip=False)
        unk_token = AddedToken(unk_token, lstrip=False, rstrip=False)
        pad_token = AddedToken(pad_token, lstrip=False, rstrip=False)
        
        self.add_bos_token = add_bos_token
        self.n_positions = n_positions

        # Load vocab and merges
        with open(vocab_file, encoding="utf-8") as vocab_handle:
            self.encoder = json.load(vocab_handle)
        self.decoder = {v: k for k, v in self.encoder.items()}
        with open(merges_file, encoding="utf-8") as merges_handle:
            bpe_merges = merges_handle.read().split("\n")
        bpe_merges = [tuple(merge.split()) for merge in bpe_merges]
        self.bpe_ranks = dict(zip(bpe_merges, range(len(bpe_merges))))        
        self.errors = errors
        self.cache = {}

        super().__init__(
            errors=errors,
            unk_token=unk_token,
            bos_token=bos_token,
            eos_token=eos_token,
            pad_token=pad_token,
            **kwargs,
        )

    @classmethod
    def create_vocab(cls, train_dataset, vocab_file, merges_file, special_tokens=[], text_field="text"):
        """
        Creates a vocab.json and merges.txt from a given training dataset.

        Args:
            train_dataset: Dataset from which to create vocab from.
            vocab_file: path of vocab.json file to create.
            merges_file: path of merges.txt file to create.
            special_tokens: list of special tokens.
            text_field: the column header of the dataset containing text to train on. 
        """
        char_counter = Counter()
        tokenized_corpus = []
        for i, example in enumerate(train_dataset, start=1):
            text = example[text_field]
            char_counter.update(text)
            tokenized_corpus.append(list(text))
            if i >= 10: # Stop at 1 million examples
                break

        # Add special tokens to vocab
        vocab = {}
        current_id = 0
        for token in special_tokens:
            if token not in vocab:
                vocab[token] = current_id
                current_id += 1

        # Add regular tokens to vocab
        for char, _ in char_counter.most_common():
            if char not in vocab:
                vocab[char] = current_id
                current_id += 1

        # Iteratively merge
        merges = []
        max_vocab_size = 30000

        #while len(vocab) < max_vocab_size:
        for _ in range(10):
            # Count the pairs
            pairs = Counter()
            for text in tokenized_corpus:
                for i in range(len(text) - 1):
                    pairs[(text[i], text[i+1])] += 1

            # Merge most frequent pair
            best_pair = max(pairs, key=pairs.get)
            merges.append(best_pair)
            merged_token = ''.join(best_pair)
            vocab[merged_token] = current_id
            current_id += 1

            # Update tokenized_corpus to include new merge
            for idx, text in enumerate(tokenized_corpus):
                new_text = []
                i = 0
                while i < len(text):
                    # Update if current and next token is current best pair
                    if i < len(text) - 1 and (text[i], text[i+1]) == best_pair:
                        new_text.append(merged_token)
                        i += 2
                    # Otherwise move to next token
                    else:
                        new_text.append(text[i])
                        i += 1
                tokenized_corpus[idx] = new_text

        # Save vocab.json and merges.txt
        os.makedirs(os.path.dirname(vocab_file), exist_ok=True)
        with open(vocab_file, "w", encoding="utf-8") as f:
            json.dump(vocab, f, indent=None, ensure_ascii=False, separators=(",", ":"))      
        os.makedirs(os.path.dirname(merges_file), exist_ok=True)
        with open(merges_file, "w", encoding="utf-8") as f:
            f.write("\n".join(f"{a} {b}" for a, b in merges))
    
    @property
    def vocab_size(self):
        return len(self.encoder)

    def get_vocab(self):
        return dict(self.encoder, **self.added_tokens_encoder)

    def _tokenize(self, text):
        """Tokenize a string."""
        return [char if char in self.encoder else self.unk_token for char in text]
    
    def tokenize_batch(self, examples):
        if isinstance(examples, str):
            texts = [examples]
        else:
            texts = examples["text"]
        tokenized = self(
            texts, 
            padding="max_length", 
            truncation=True, 
            max_length=self.n_positions, 
            return_tensors=None
        )
        tokenized["labels"] = tokenized["input_ids"].copy()
        return tokenized

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