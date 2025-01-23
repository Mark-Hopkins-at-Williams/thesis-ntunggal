import os
import json
from collections import Counter
from typing import Optional, Tuple
from transformers import GPT2TokenizerFast
from transformers import AddedToken, PreTrainedTokenizer, logging
import sentencepiece as spm

logger = logging.get_logger(__name__)

VOCAB_FILES_NAMES = {
    "vocab_file": "vocab.json",
    "merges_file": "merges.txt",
    "spm_model": "spm.model",
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
    """
    
    def __init__(
        self,
        vocab_file,
        n_positions,
        unk_token="<unk>",
        bos_token="<bos>",
        eos_token="<eos>",
        pad_token="<pad>",
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

        super().__init__(
            unk_token=unk_token,
            bos_token=bos_token,
            eos_token=eos_token,
            pad_token=pad_token,
            **kwargs,
        )

    @classmethod
    def create_vocab(
        cls, 
        train_dataset, 
        save_directory, 
        special_tokens=[], 
        text_field="text", 
        max_vocab_size=50257, 
        max_examples=None,
        model_prefix="char_tokenizer",
    ):
        """
        Creates and saves a vocab.json file.

        Args:
            train_dataset: Dataset to create vocab from.
            save_directory: Directory to save spm.model and vocab.json.
            special_tokens: List of special tokens to include.
            text_field: Field containing text data in the dataset.
            max_vocab_size: Max vocabulary size.
            max_examples: Number of examples to use for training.
            model_prefix: Prefix for the SentencePiece model file.
        """
        char_counter = Counter()
        for i, example in enumerate(train_dataset, start=1):
            text = example[text_field]
            text = text.replace("\u2028", "").replace("\u2029", "")
            char_counter.update(text)
            if (max_examples is not None and i >= max_examples) or len(char_counter) >= max_vocab_size:
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
            if current_id >= max_vocab_size:
                break
        assert len(vocab) <= max_vocab_size
        # Save vocab.json
        vocab_file = os.path.join(save_directory, "vocab.json")
        os.makedirs(os.path.dirname(vocab_file), exist_ok=True)
        with open(vocab_file, "w", encoding="utf-8") as f:
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
    Construct a Subword BPE tokenizer.
    """
    
    def __init__(
        self,
        vocab_file,
        n_positions,
        unk_token="<unk>",
        bos_token="<s>",
        eos_token="</s>",
        pad_token="<pad>",
        add_bos_token=False,
        **kwargs,
    ):
        # Hardcoded since sentencepiece uses these special tokens
        bos_token = AddedToken("<s>", lstrip=False, rstrip=False)
        eos_token = AddedToken("</s>", lstrip=False, rstrip=False)
        unk_token = AddedToken("<unk>", lstrip=False, rstrip=False)
        pad_token = AddedToken(pad_token, lstrip=False, rstrip=False)
        
        self.add_bos_token = add_bos_token
        self.n_positions = n_positions

        # Load spm vocab and merges
        self.sp = spm.SentencePieceProcessor()
        self.sp.load(vocab_file)
        self.encoder = {self.sp.id_to_piece(i): i for i in range(self.sp.get_piece_size())}
        self.decoder = {v: k for k, v in self.encoder.items()}

        super().__init__(
            unk_token=unk_token,
            bos_token=bos_token,
            eos_token=eos_token,
            pad_token=pad_token,
            **kwargs,
        )

    @classmethod
    def create_vocab(
        cls, 
        train_dataset, 
        save_directory, 
        special_tokens=[], 
        text_field="text", 
        max_vocab_size=50257, 
        max_examples=None,
        model_prefix="bpe_tokenizer",
    ):
        """
        Creates a SentencePiece model and saves vocab.json and spm.model.

        Args:
            train_dataset: Dataset to create vocab from.
            save_directory: Directory to save spm.model and vocab.json.
            special_tokens: List of special tokens to include.
            text_field: Field containing text data in the dataset.
            max_vocab_size: Max vocabulary size.
            max_examples: Number of examples to use for training.
            model_prefix: Prefix for the SentencePiece model file.
        """
        os.makedirs(save_directory, exist_ok=True)
        temp_file = os.path.join(save_directory, "temp_text.txt")
        
        with open(temp_file, "w", encoding="utf-8") as f:
            for i, example in enumerate(train_dataset, start=1):
                text = example[text_field].strip()
                if text:
                    f.write(text + "\n")
                if max_examples is not None and i >= max_examples:
                    break
        
        # Define SentencePiece training parameters
        print(f"special tokens: {special_tokens}")
        spm.SentencePieceTrainer.train(
            input=temp_file,
            model_prefix=os.path.join(save_directory, model_prefix),
            vocab_size=max_vocab_size,
            user_defined_symbols=special_tokens,
            character_coverage=0.9995,
            model_type="bpe",
        )

        # Create and save vocab.json
        spm_model_file = os.path.join(save_directory, f"{model_prefix}.model")
        sp = spm.SentencePieceProcessor(model_file=spm_model_file)
        vocab = {sp.id_to_piece(i): i for i in range(sp.get_piece_size())}
        vocab_file = os.path.join(save_directory, "vocab.json")
        with open(vocab_file, "w", encoding="utf-8") as f:
            json.dump(vocab, f, indent=None, ensure_ascii=False, separators=(",", ":"))

        # Cleanup temporary file
        os.remove(temp_file)
    
    @property
    def vocab_size(self):
        return len(self.encoder)

    def get_vocab(self):
        return dict(self.encoder, **self.added_tokens_encoder)

    def _tokenize(self, text):
        """Tokenize a string. Returns a list of BPE tokens."""
        return self.sp.encode(text, out_type=str)
    
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
        return self.decoder.get(index, self.unk_token)
    
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
    

class ByteTokenizer(PreTrainedTokenizer):
    """
    Construct a UTF-8 byte-level tokenizer.
    """
    
    def __init__(
        self,
        vocab_file,
        n_positions,
        unk_token="<unk>",
        bos_token="<s>",
        eos_token="</s>",
        pad_token="<pad>",
        **kwargs,
    ):
        bos_token = AddedToken(bos_token, lstrip=False, rstrip=False)
        eos_token = AddedToken(eos_token, lstrip=False, rstrip=False)
        unk_token = AddedToken(unk_token, lstrip=False, rstrip=False)
        pad_token = AddedToken(pad_token, lstrip=False, rstrip=False)
        
        self.n_positions = n_positions
        self._added_tokens_decoder = {0: pad_token, 1: bos_token, 2: eos_token, 3: unk_token}
        self.offset = len(self._added_tokens_decoder)
        self._utf_vocab_size = 2**8

        super().__init__(
            unk_token=unk_token,
            bos_token=bos_token,
            eos_token=eos_token,
            pad_token=pad_token,
            **kwargs,
        )

    @classmethod
    def create_vocab(
        cls, 
        train_dataset, 
        save_directory, 
        special_tokens=[], 
        text_field="text", 
        max_vocab_size=50257, 
        max_examples=None,
        model_prefix="byte_tokenizer",
    ):
        """
        ByteTokenizer does not need to be trained.

        Args (not used):
            train_dataset: Dataset to create vocab from.
            save_directory: Directory to save spm.model and vocab.json.
            special_tokens: List of special tokens to include.
            text_field: Field containing text data in the dataset.
            max_vocab_size: Max vocabulary size.
            max_examples: Number of examples to use for training.
            model_prefix: Prefix for the SentencePiece model file.
        """
        return
    
    @property
    def vocab_size(self):
        # Special tokens handled separately
        return self._utf_vocab_size

    def get_vocab(self):
        vocab = {self.convert_ids_to_tokens(i): i for i in range(self.vocab_size + self.offset)}
        vocab.update(self.added_tokens_encoder)
        return vocab

    def _tokenize(self, text):
        """Tokenize a string. Returns a list of strings (tokens)."""
        tokens = [chr(i) for i in text.encode("utf-8")]
        return tokens
    
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
        if len(token) != 1:
            token_id = None
        else:
            token_id = ord(token) + self.offset
        return token_id

    def _convert_id_to_token(self, index):
        """Converts an index (integer) in a token (str) using the vocab."""
        token = chr(index - self.offset)
        return token

    # ByteTokenizer has no vocab file
    def save_vocabulary(self, save_directory: str, filename_prefix: Optional[str] = None) -> Tuple[str]:
        return ()
    

class ByteBPETokenizer(PreTrainedTokenizer):
    """
    Construct a Subword BPE tokenizer.
    """
    
    def __init__(
        self,
        vocab_file,
        n_positions,
        unk_token="<unk>",
        bos_token="<s>",
        eos_token="</s>",
        pad_token="<pad>",
        add_bos_token=False,
        **kwargs,
    ):
        # Hardcoded since sentencepiece uses these special tokens
        bos_token = AddedToken("<s>", lstrip=False, rstrip=False)
        eos_token = AddedToken("</s>", lstrip=False, rstrip=False)
        unk_token = AddedToken("<unk>", lstrip=False, rstrip=False)
        pad_token = AddedToken(pad_token, lstrip=False, rstrip=False)
        
        self.add_bos_token = add_bos_token
        self.n_positions = n_positions

        # Load spm vocab and merges
        self.sp = spm.SentencePieceProcessor()
        self.sp.load(vocab_file)
        self.encoder = {self.sp.id_to_piece(i): i for i in range(self.sp.get_piece_size())}
        self.decoder = {v: k for k, v in self.encoder.items()}

        super().__init__(
            unk_token=unk_token,
            bos_token=bos_token,
            eos_token=eos_token,
            pad_token=pad_token,
            **kwargs,
        )

    @classmethod
    def create_vocab(
        cls, 
        train_dataset, 
        save_directory, 
        special_tokens=[], 
        text_field="text", 
        max_vocab_size=50257, 
        max_examples=None,
        model_prefix="byte_bpe",
    ):
        """
        Creates a SentencePiece model and saves vocab.json and spm.model.

        Args:
            train_dataset: Dataset to create vocab from.
            save_directory: Directory to save spm.model and vocab.json.
            special_tokens: List of special tokens to include.
            text_field: Field containing text data in the dataset.
            max_vocab_size: Max vocabulary size.
            max_examples: Number of examples to use for training.
            model_prefix: Prefix for the SentencePiece model file.
        """
        def text_to_byte_seq(text):
            return "".join([chr(c) for c in text.encode("utf-8")])
        
        os.makedirs(save_directory, exist_ok=True)
        temp_file = os.path.join(save_directory, "temp_text.txt")
        
        with open(temp_file, "w", encoding="utf-8") as f:
            for i, example in enumerate(train_dataset, start=1):
                text = example[text_field].strip()
                if text:
                    f.write(text_to_byte_seq(text) + "\n")
                if max_examples is not None and i >= max_examples:
                    break
        
        # Define SentencePiece training parameters
        print(f"special tokens: {special_tokens}")
        spm.SentencePieceTrainer.train(
            input=temp_file,
            model_prefix=os.path.join(save_directory, model_prefix),
            vocab_size=max_vocab_size,
            user_defined_symbols=special_tokens,
            character_coverage=1.0,
            model_type="bpe",
            max_sentence_length=16384
        )

        # Create and save vocab.json
        spm_model_file = os.path.join(save_directory, f"{model_prefix}.model")
        sp = spm.SentencePieceProcessor(model_file=spm_model_file)
        vocab = {sp.id_to_piece(i): i for i in range(sp.get_piece_size())}
        vocab_file = os.path.join(save_directory, "vocab.json")
        with open(vocab_file, "w", encoding="utf-8") as f:
            json.dump(vocab, f, indent=None, ensure_ascii=False, separators=(",", ":"))

        # Cleanup temporary file
        os.remove(temp_file)
    
    @property
    def vocab_size(self):
        return len(self.encoder)

    def get_vocab(self):
        return dict(self.encoder, **self.added_tokens_encoder)

    def _tokenize(self, text):
        """Tokenize a string. Returns a list of BPE tokens."""
        return self.sp.encode(text, out_type=str)
    
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
        return self.decoder.get(index, self.unk_token)

    def save_vocabulary(self, save_directory: str, filename_prefix: Optional[str] = None) -> Tuple[str]:
        return ()