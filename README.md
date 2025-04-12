# thesis-ntunggal


## Directories and files

- `experiments`: contains subdirectories with experiments for each tokenizer.
    - `experiment.json`: contains configs for experiment.
    - `checkpoints` and `logs`: checkpoints and logs for experiment.
    - `parameter_counts.txt`: model parameter counts.
    - `ratios.txt`: experiment results.
- `baai.py`: code for loading the BAAI dataset
- `entropy.py`: code for entropy calculations
- `tokenization.py`: contains custom tokenizer classes
- `train_gpt.py`: trains and evaluates a GPT model using a tokenizer specified by configs


## Set up a new experiment:

1. Add a new subdirectory under `experiments` with the name of the experiment.
2. In the new directory, create `experiment.json`. This contains the configs for the experiment. `tokenizer` configs are different for each tokenizer (see here for tokenizer-specific guide), `model` configs set model architecture, and `training` configs are training settings. The latter two config types are mostly kept consistent across experiments. Below is a description of all setting options.

- tokenizer
    - **name**: (str) class name of the tokenizer to use.
    - **max_vocab_size**: (num | "") vocabulary size cap. For None, use "".
    - **max_examples**: (num | "") max number of examples of the training dataset to use. To use all, use "".
    - **special_tokens**: (dict[str, str]) a dictionary of special tokens. key is either `'unk_token'`, `'bos_token'`, `'eos_token'`, `'pad_token'` and value is the token to use (as a string).
        - For BPE tokenizers, only use `'pad_token'`
    - **tokenizer_files_dir**: (str) directory to store tokenizer-related files, such as `vocab.json` or `.model` files. For None, use "".
        - ByteTokenizer does not need this, use ""
    - **vocab_file_name**: (str) the name of the tokenizer file (examples: `vocab.json` or `subword_bpe.model`). For None, use "".
        - ByteTokenizer does not need this, use ""
    - **input_method**: (str) for Chinese tokenizers, the input method to tokenize by.
    - **sp_file**: (str) file path to SentencePiece model file, for repackaged tokenizers.
- model
    - **n_positions**: (num) max token sequence length
    - **n_ctx**: (num) max length of context window (same as n_positions)
    - **n_embd**: (num) depth of embedding dimension
    - **n_layer**: (num) number of hidden layers
    - **n_head**: (num) number of attention heads
    - **activation_function**: (str) activation function to use
- training
    - **eval_steps**: (num) evaluation will run every 
    - **max_steps**: (num) maximum number of steps to train for
    - **batch_size**: (num) batch size
    - **learning_rate**: (num) learning rate
    - **weight_decay**: (num) weight decay
    - **warmup_steps**: (num) warmup steps
    - **max_grad_norm**: (num) max_grad_norm

3. In `run.sh`, put the following command, making sure the argument points to your experiment directory:
```bash
python train_gpt.py experiments/my_experiment
```

## Specific tokenizer settings

Each tokenizer uses slightly different configs. For example, some tokenizers need to save a vocab or model file, Chinese tokenizers need an input method specified while others don't, etc. Below are lists of valid settings for each tokenizer. If a field is in parenthesis, the user can choose any value fitting the description.

Note: all BPE tokenizers should only have {"pad_token": "<pad>"} for special tokens, since the others are defined by SentencePiece.

### CharacterTokenizer

- tokenizer
    - name: "CharacterTokenizer"
    - max_vocab_size: (any number)
    - max_examples: (any number) or "" for all
    - special_tokens: (dictionary with any/all keys 'unk_token', 'bos_token', 'eos_token', 'pad_token')
    - tokenizer_files_dir: (path to directory to store `vocab.json`)
    - vocab_file_name: (actual name of vocab file, ex. `vocab.json`)
    - input_method: ""
    - sp_file: ""

### SubwordTokenizer

- tokenizer
    - name: "SubwordTokenizer"
    - max_vocab_size: (any number)
    - max_examples: (any number) or "" for all
    - special_tokens: {"pad_token": "<pad>"}
    - tokenizer_files_dir: (path to directory to store `vocab.json`, `subword_bpe.model`, `subword_bpe.vocab`)
    - vocab_file_name: (actual name of vocab file, ex. `subword_bpe.model`)
    - input_method: ""
    - sp_file: ""

### ByteTokenizer

- tokenizer
    - name: "ByteTokenizer"
    - max_vocab_size: ""
    - max_examples: (any number) or "" for all
    - special_tokens: {"pad_token": "<pad>"}
    - tokenizer_files_dir: ""
    - vocab_file_name: ""
    - input_method: ""
    - sp_file: ""
- model
    - n_embd: (adjust to parameter match with other models)

### ByteBPETokenizer

- tokenizer
    - name: "ByteBPETokenizer"
    - max_vocab_size: (any number)
    - max_examples: (any number) or "" for all
    - special_tokens: {"pad_token": "<pad>"}
    - tokenizer_files_dir: (path to directory to store `vocab.json`, `byte_bpe.model`, `byte_bpe.vocab`)
    - vocab_file_name: (actual name of vocab file, ex. `byte_bpe.model`)
    - input_method: ""
    - sp_file: ""

### ChineseBPETokenizer

This class is multiple tokenizers packed into one class. There are set options for the setting `input_method`.

- tokenizer
    - name: "ChineseBPETokenizer"
    - max_vocab_size: (any number)
    - max_examples: (any number) or "" for all
    - special_tokens: {"pad_token": "<pad>"}
    - tokenizer_files_dir: (path to directory to store `vocab.json`, `chinese_bpe.model`, `chinese_bpe.vocab`)
    - vocab_file_name: (actual name of vocab file, ex. `chinese_bpe.model`)
    - input_method: "pinyin_tone_above" or "pinyin_tone_after" or "zhuyin" or "wubi" or "cangjie"
    - sp_file: ""

### RepackagedBPETokenizer

- tokenizer
    - name: "ChineseBPETokenizer"
    - max_vocab_size: (any number)
    - max_examples: (any number) or "" for all
    - special_tokens: {"pad_token": "<pad>"}
    - tokenizer_files_dir: (path to directory to store `vocab.json`, `chinese_bpe.model`, `chinese_bpe.vocab`)
    - vocab_file_name: (actual name of vocab file, ex. `chinese_bpe.model`)
    - input_method: "pinyin_tone_above" or "pinyin_tone_after" or "zhuyin" or "wubi" or "cangjie"
    - sp_path: (path to a SentencePiece `.model` file corresponding to the input method selected)