import os
import json
from abc import ABC, abstractmethod
from collections import Counter
from typing import Optional, Tuple
from transformers import GPT2TokenizerFast
from transformers import AddedToken, PreTrainedTokenizer, logging
import sentencepiece as spm
from pypinyin import pinyin, Style
import pickle

logger = logging.get_logger(__name__)

VOCAB_FILES_NAMES = {
    "vocab_file": "vocab.json",
    "merges_file": "merges.txt",
    "spm_model": "spm.model",
}

class DefaultGPT2Tokenizer:

    def __init__(self, n_positions):
        self.tokenizer = GPT2TokenizerFast.from_pretrained("gpt2")
        self.tokenizer.add_special_tokens({'pad_token': '<pad>'})
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

class CustomTokenizerBase(PreTrainedTokenizer, ABC):
    """
    Common methods for a custom tokenizer.
    """

    @abstractmethod
    def __init__(
        self,
        vocab_file=None,
        n_positions=None,
        unk_token="<unk>",
        bos_token="<bos>",
        eos_token="<eos>",
        pad_token="<pad>",
        add_bos_token=False,
        **kwargs,
    ):
        pass

    # Many tokenizers will include a create_vocab method here.

    @property
    def vocab_size(self):
        return len(self.encoder)

    def get_vocab(self):
        return dict(self.encoder, **self.added_tokens_encoder)

    @abstractmethod
    def _tokenize(self, text):
        pass
    
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
    
    # Required to overwrite method in PreTrainedTokenizer
    def save_vocabulary(self, save_directory: str, filename_prefix: Optional[str] = None) -> Tuple[str]:
        return ()

    @classmethod
    def from_config(cls, config, n_positions):
        kwargs = {
            'vocab_file': os.path.join(config['tokenizer_files_dir'], config['vocab_file_name']),
            'n_positions': n_positions
        }
        for token in ['unk_token', 'bos_token', 'eos_token', 'pad_token']:
            if token in config['special_tokens']:
                kwargs[token] = config['special_tokens'][token]
        return cls(**kwargs)

    @classmethod
    def create_vocab_from_config(cls, train_data, config):
        return cls.create_vocab(
            train_data,
            save_directory=config['tokenizer_files_dir'],
            special_tokens=list(config['special_tokens'].values()),
            max_vocab_size=config.get('max_vocab_size', ''),
            max_examples=config.get('max_examples', ''),
        )

class ChineseTokenizerBase(CustomTokenizerBase):
    
    @staticmethod
    def get_stroke_dict(file_path):
        """Returns a dict mapping character to wubi/cangjie code from yaml file."""
        stroke_dict = {}
        parsing = False

        with open(file_path, "r", encoding="utf-8") as f:
            for line in f:
                line = line.strip()
                if line == "...":
                    parsing = True
                    continue
                if not parsing or line.startswith("#") or not line:
                    continue
                
                parts = line.split()
                if len(parts) >= 2:
                    char = parts[0]
                    code = parts[1]
                    if char not in stroke_dict:
                        stroke_dict[char] = code
        return stroke_dict
    
    @staticmethod
    def _pretokenize(text, input_method, stroke_dict=None):
        """Converts a text in Chinese into its input representation."""
        
        number_to_circle = {'1': '①', '2': '②', '3': '③', '4': '④'}
        def replace_tone(syllable):
            if syllable[-1] in number_to_circle:
                return syllable[:-1] + number_to_circle[syllable[-1]]
            return syllable
        
        if input_method == "pinyin_tone_above":
            # nǐ#hǎo#，#wǒ#shì#xiǎo#bái
            pretokenized = pinyin(text, style=Style.TONE, heteronym=False)
            return "#".join(["".join(word) for word in pretokenized])
        elif input_method == "pinyin_tone_after":
            # ni③#hao③#，#wo③#shi④#xiao③#bai②#
            pretokenized = pinyin(text, style=Style.TONE3, heteronym=False)
            return "#".join(["".join(replace_tone(word) for word in group) for group in pretokenized])
        elif input_method == "zhuyin":
            # ㄋㄧˇ#ㄏㄠˇ#，#ㄨㄛˇ#ㄕˋ#ㄒㄧㄠˇ#ㄅㄞˊ#
            pretokenized = pinyin(text, style=Style.BOPOMOFO, heteronym=False)
            return "#".join(["".join(word) for word in pretokenized])
        elif input_method == "wubi":
            assert stroke_dict
            return "#".join([stroke_dict.get(char, char) for char in text])
        elif input_method == "cangjie":
            assert stroke_dict
            return "#".join([stroke_dict.get(char, char) for char in text])
        else:
            raise ValueError(f"Received invalid input_method: {input_method}.")
        
    @classmethod
    def from_config(cls, config, n_positions):
        kwargs = {
            'vocab_file': os.path.join(config['tokenizer_files_dir'], config['vocab_file_name']),
            'n_positions': n_positions
        }
        for token in ['unk_token', 'bos_token', 'eos_token', 'pad_token']:
            if token in config['special_tokens']:
                kwargs[token] = config['special_tokens'][token]
        input_method = config['input_method']
        if input_method in ['pinyin_tone_above', 'pinyin_tone_after', 'zhuyin', 'wubi', 'cangjie']:
            kwargs['input_method'] = input_method
        else:
            raise ValueError(f"Received invalid input_method: {input_method}.")
        return cls(**kwargs)
    
    @classmethod
    def create_vocab_from_config(cls, train_data, config):
        return cls.create_vocab(
            train_data,
            save_directory=config['tokenizer_files_dir'],
            special_tokens=list(config['special_tokens'].values()),
            max_vocab_size=config.get('max_vocab_size', ''),
            max_examples=config.get('max_examples', ''),
            input_method=config.get('input_method', '')
        )
    

class CharacterTokenizer(CustomTokenizerBase):
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

        PreTrainedTokenizer.__init__(
            self,
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
        max_vocab_size=50257, 
        max_examples=None,
    ):
        """
        Creates and saves a vocab.json file.

        Args:
            train_dataset: Dataset to create vocab from.
            save_directory: Directory to save spm.model and vocab.json.
            special_tokens: List of special tokens to include.
            max_vocab_size: Max vocabulary size.
            max_examples: Number of examples to use for training.
        """
        char_counter = Counter()
        for i, example in enumerate(train_dataset, start=1):
            text = example["text"]
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

    def _tokenize(self, text):
        """Tokenize a string."""
        return [char if char in self.encoder else self.unk_token for char in text]

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


class SubwordBPETokenizer(CustomTokenizerBase):
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

        PreTrainedTokenizer.__init__(
            self,
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
            max_vocab_size: Max vocabulary size.
            max_examples: Number of examples to use for training.
            model_prefix: Prefix for the SentencePiece model file.
        """
        os.makedirs(save_directory, exist_ok=True)
        temp_file = os.path.join(save_directory, "temp_text.txt")
        
        with open(temp_file, "w", encoding="utf-8") as f:
            for i, example in enumerate(train_dataset, start=1):
                text = example["text"].strip()
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
            max_sentence_length=32768,
            split_by_unicode_script=False,
            normalization_rule_name='identity',
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

    def _tokenize(self, text):
        """Tokenize a string. Returns a list of BPE tokens."""
        return self.sp.encode(text, out_type=str)

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
    

class ByteTokenizer(CustomTokenizerBase):
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

        PreTrainedTokenizer.__init__(
            self,
            unk_token=unk_token,
            bos_token=bos_token,
            eos_token=eos_token,
            pad_token=pad_token,
            **kwargs,
        )
    
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

    @classmethod
    def create_vocab_from_config(cls, train_data, config):
        # ByteTokenizer does not need to be trained
        pass
    

class ByteBPETokenizer(CustomTokenizerBase):
    """
    Construct a Byte BPE tokenizer.
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

        PreTrainedTokenizer.__init__(
            self,
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
                text = example["text"].strip()
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
            max_sentence_length=32768,
            split_by_unicode_script=False,
            normalization_rule_name='identity',
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

    def _tokenize(self, text):
        """Tokenize a string. Returns a list of BPE tokens."""
        def text_to_byte_seq(text):
            return "".join([chr(c) for c in text.encode("utf-8")])
        return self.sp.encode(text_to_byte_seq(text), out_type=str)
    

class ChineseBPETokenizer(ChineseTokenizerBase):
    """
    Construct a Chinese BPE tokenizer.
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
        input_method=None,
        **kwargs,
    ):
        # Hardcoded since sentencepiece uses these special tokens
        bos_token = AddedToken("<s>", lstrip=False, rstrip=False)
        eos_token = AddedToken("</s>", lstrip=False, rstrip=False)
        unk_token = AddedToken("<unk>", lstrip=False, rstrip=False)
        pad_token = AddedToken(pad_token, lstrip=False, rstrip=False)
        
        assert input_method in ['pinyin_tone_above', 'pinyin_tone_after', 'zhuyin', 'wubi', 'cangjie']
        self.add_bos_token = add_bos_token
        self.n_positions = n_positions
        self.input_method = input_method

        # Load dictionaries for wubi and cangjie
        # TODO: fix path mess
        self.stroke_dict = None
        if self.input_method == "wubi":
            #self.stroke_dict = __class__.get_stroke_dict("/mnt/storage/ntunggal/wubi86.dict.yaml")
            self.stroke_dict = __class__.get_stroke_dict("wubi86-components.dict.yaml")
        elif self.input_method == "cangjie":
            #self.stroke_dict = __class__.get_stroke_dict("/mnt/storage/ntunggal/cangjie5.dict.yaml")
            self.stroke_dict = __class__.get_stroke_dict("cangjie5-components.dict.yaml")

        # Load spm vocab and merges
        self.sp = spm.SentencePieceProcessor()
        self.sp.load(vocab_file)
        self.encoder = {self.sp.id_to_piece(i): i for i in range(self.sp.get_piece_size())}
        self.decoder = {v: k for k, v in self.encoder.items()}

        PreTrainedTokenizer.__init__(
            self,
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
        max_vocab_size=50257, 
        max_examples=None,
        model_prefix="chinese_bpe",
        input_method=None,
    ):
        """
        Creates a SentencePiece model and saves vocab.json and spm.model.

        Args:
            train_dataset: Dataset to create vocab from.
            save_directory: Directory to save spm.model and vocab.json.
            special_tokens: List of special tokens to include.
            max_vocab_size: Max vocabulary size.
            max_examples: Number of examples to use for training.
            model_prefix: Prefix for the SentencePiece model file.
        """        
        
        os.makedirs(save_directory, exist_ok=True)
        temp_file = os.path.join(save_directory, "temp_text.txt")
        
        assert input_method in ['pinyin_tone_above', 'pinyin_tone_after', 'zhuyin', 'wubi', 'cangjie']
        stroke_dict = None
        # TODO: fix path mess
        if input_method == "wubi":
            #stroke_dict = cls.get_stroke_dict("/mnt/storage/ntunggal/wubi86.dict.yaml")
            stroke_dict = cls.get_stroke_dict("wubi86-components.dict.yaml")
        elif input_method == "cangjie":
            #stroke_dict = cls.get_stroke_dict("/mnt/storage/ntunggal/cangjie5.dict.yaml")
            stroke_dict = cls.get_stroke_dict("cangjie5-components.dict.yaml")
        
        with open(temp_file, "w", encoding="utf-8") as f:
            for i, example in enumerate(train_dataset, start=1):
                text = example["text"].strip()
                f.write(cls._pretokenize(text, input_method, stroke_dict) + "\n")
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
            max_sentence_length=32768,
            split_by_unicode_script=False,
            normalization_rule_name='identity',
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

    def _tokenize(self, text):
        """Tokenize a string. Returns a list of BPE tokens."""
        text = self._pretokenize(text, self.input_method, self.stroke_dict)
        return self.sp.encode(text, out_type=str)
    

class RepackagedByteTokenizer(ChineseTokenizerBase):
    """
    Construct a repackaged UTF-8 byte tokenizer. Maps single tokens to UTF-8 byte representations.
    """
    
    def __init__(
        self,
        vocab_file,
        n_positions,
        unk_token="<unk>",
        bos_token="<s>",
        eos_token="</s>",
        pad_token="<pad>",
        input_method="pinyin_tone_above",
        sp_path="",
        **kwargs,
    ):
        bos_token = AddedToken(bos_token, lstrip=False, rstrip=False)
        eos_token = AddedToken(eos_token, lstrip=False, rstrip=False)
        unk_token = AddedToken("<unk>", lstrip=False, rstrip=False)
        pad_token = AddedToken(pad_token, lstrip=False, rstrip=False)
        
        self.n_positions = n_positions
        self._added_tokens_decoder = {0: pad_token, 1: bos_token, 2: eos_token, 3: unk_token}
        self.offset = len(self._added_tokens_decoder)
        self._utf_vocab_size = 2**8
        self.input_method = input_method
        self.sp = spm.SentencePieceProcessor()
        self.sp.load(sp_path)

        # Load dictionaries for wubi and cangjie
        # TODO: fix path mess
        self.stroke_dict = None
        if self.input_method == "wubi":
            #self.stroke_dict = __class__.get_stroke_dict("/mnt/storage/ntunggal/wubi86.dict.yaml")
            self.stroke_dict = __class__.get_stroke_dict("wubi86-components.dict.yaml")
        elif self.input_method == "cangjie":
            #self.stroke_dict = __class__.get_stroke_dict("/mnt/storage/ntunggal/cangjie5.dict.yaml")
            self.stroke_dict = __class__.get_stroke_dict("cangjie5-components.dict.yaml")
        
        with open(vocab_file, "rb") as f:
            self.remapped_utf8 = pickle.load(f)
            
        PreTrainedTokenizer.__init__(
            self,
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
        max_examples=None,
        input_method="",
        sp_path=""
    ):
        """
        Repurposed to create and save a dictionary mapping tokens to new
        UTF-8 representations.

        Useful args:
            train_dataset: Dataset to create vocab from.
            save_directory: Directory to save remapped UTF-8 file.
            max_examples: Number of examples to use for training.
            input_method: Input method to pretokenize by.
            sp_path: Path to SentencePiece model file for getting pretokenized tokens. 
        """
        sp = spm.SentencePieceProcessor()
        sp.load(sp_path)

        stroke_dict = None
        # TODO: fix path mess
        if input_method == "wubi":
            #stroke_dict = cls.get_stroke_dict("/mnt/storage/ntunggal/wubi86.dict.yaml")
            stroke_dict = cls.get_stroke_dict("wubi86-components.dict.yaml")
        elif input_method == "cangjie":
            #stroke_dict = cls.get_stroke_dict("/mnt/storage/ntunggal/cangjie5.dict.yaml")
            stroke_dict = cls.get_stroke_dict("cangjie5-components.dict.yaml")

        # Count pretokens in pretokenized text
        tk_counts = Counter()
        for i, example in enumerate(train_dataset, start=1):
            text = example["text"].strip()
            if text:
                if input_method != "": # "" is subword
                    text = cls._pretokenize(text, input_method, stroke_dict)
                pretokens = sp.encode(text, out_type=str)
                tk_counts.update(pretokens)
            if max_examples is not None and i >= max_examples:
                break
            if (i % 10000) == 0:
                print(f"Processed example {i}", flush=True)
        
        # Assign tokens to UTF-8 representations in freq order
        tk_list = [tok for tok, count in tk_counts.most_common()]
        tk_list.append("<unk>")
        tk_iter = iter(tk_list)
        remapped_dict = {}

        # Helper function to assign UTF-8. r = new UTF-8 representation
        def assign_mapping(r):
            try:
                tok = next(tk_iter)
                remapped_dict[tok] = r
                return True
            except StopIteration:
                return False # Ran out of tokens to map

        # In a function to make breaks easier 
        def do_remapping():
            # 1-byte
            for b1 in range(0x20, 0x80):  
                if not assign_mapping(chr(b1)): return

            # 2-byte
            for b1 in range(0xC2, 0xE0):  
                for b2 in range(0x80, 0xC0):
                    if not assign_mapping(chr(b1) + chr(b2)): return

            # 3-byte
            for b1 in range(0xE0, 0xF0):
                for b2 in range(0x80, 0xC0):
                    if b1 == 0xE0 and b2 < 0xA0: continue
                    if b1 == 0xED and b2 >= 0xA0: continue
                    for b3 in range(0x80, 0xC0):
                        if not assign_mapping(chr(b1) + chr(b2) + chr(b3)): return

            # 4-byte
            for b1 in range(0xF0, 0xF5):
                for b2 in range(0x80, 0xC0):
                    if b1 == 0xF0 and b2 < 0x90: continue
                    if b1 == 0xF4 and b2 >= 0x90: continue
                    for b3 in range(0x80, 0xC0):
                        for b4 in range(0x80, 0xC0):
                            if not assign_mapping(chr(b1) + chr(b2) + chr(b3) + chr(b4)): return
        
        do_remapping()
    
        # Save remapped_dict to file using pickle
        os.makedirs(save_directory, exist_ok=True)
        remapping_file = os.path.join(save_directory, "remapped_utf8_byte.pkl")
        with open(remapping_file, "wb") as f:
            pickle.dump(remapped_dict, f)
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
        if self.input_method != "": # "" is subword
            text = self._pretokenize(text, self.input_method, self.stroke_dict)
        pretokens = self.sp.encode(text, out_type=str)
        utf8_strings = [self.remapped_utf8.get(tok, self.remapped_utf8.get(self.unk_token)) for tok in pretokens]
        return [char for s in utf8_strings for char in s]

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

    @classmethod
    def from_config(cls, config, n_positions):
        kwargs = {
            'vocab_file': os.path.join(config['tokenizer_files_dir'], config['vocab_file_name']),
            'n_positions': n_positions
        }
        for token in ['unk_token', 'bos_token', 'eos_token', 'pad_token']:
            if token in config['special_tokens']:
                kwargs[token] = config['special_tokens'][token]
        input_method = config['input_method']
        if input_method in ['pinyin_tone_above', 'pinyin_tone_after', 'zhuyin', 'wubi', 'cangjie', '']:
            kwargs['input_method'] = input_method
        else:
            raise ValueError(f"Received invalid input_method: {input_method}.")
        assert config['sp_path']
        kwargs['sp_path'] = config['sp_path']
        return cls(**kwargs)
    
    @classmethod
    def create_vocab_from_config(cls, train_data, config):
        sp_path = config.get('sp_path', '')
        assert sp_path != ''
        return cls.create_vocab(
            train_data,
            save_directory=config['tokenizer_files_dir'],
            max_examples=config.get('max_examples', ''),
            input_method=config.get('input_method', ''),
            sp_path=sp_path,
        )