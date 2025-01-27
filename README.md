# thesis-ntunggal


## Directories and files

- `experiments`: contains subdirectories with experiments for each tokenizer.
    - `experiment.json`: contains configs for experiment.
    - `checkpoints` and `logs`: checkpoints and logs for experiment.
    - `ratios.txt`: experiment results.
- `baai.py`: code for loading the BAAI dataset
- `entropy.py`: code for entropy calculations
- `tokenization.py`: contains custom tokenizer classes
- `train_gpt.py`: trains and evaluates a GPT model using a tokenizer specified by configs


## Steps to recreate

To set up a new experiment:
1. Add a new subdirectory under `experiments` with the name of your experiment.
2. In the new directory, create `experiment.json`. This contains the configs for the experiment. Below is a guide to all the config settings:

- tokenizer
    - **name**: (str) class name of the tokenizer to use, from tokenizer.py
    - **max_vocab_size**: (num | "") vocabulary size will be capped at this value. if not setting a value, use ""
        - CharacterTokenizer, SubwordBPETokenizer, ByteBPETokenizer need this set
        - ByteTokenizer does not need this, use ""
    - **max_examples**: (num | "") max number of examples of the training dataset to use. to use all examples, put ""
    - **special_tokens**: (dict[str, str]) a dictionary of special tokens. key is the token name (one of `'unk_token'`, `'bos_token'`, `'eos_token'`, `'pad_token'`) and value is the token to use (as a string).
        - For BPE tokenizers, only use `'pad_token'`
    - **tokenizer_files_dir**: (str) directory to store files related to the tokenizer, such as `vocab.json` or `.model` files. if not setting a value, use ""
        - ByteTokenizer does not need this, use ""
    - **vocab_file_name**: (str) the name of the tokenizer file. for example, `vocab.json` or `subword_bpe.model`. if not setting a value, use ""
        - ByteTokenizer does not need this, use ""
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
