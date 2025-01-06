#!/bin/sh
#SBATCH -c 1                
#SBATCH -t 0-24:00          
#SBATCH -p dl               
#SBATCH --mem=10G           
#SBATCH -o log_%j.out  
#SBATCH -e log_%j.err
#SBATCH --gres=gpu:1
python thesis-ntunggal/roberta-training/evaluate_clue.py --model_dir /mnt/storage/ntunggal/finetuned-baidu-82500/afqmc --tokenizer_dir /mnt/storage/ntunggal/baidu-model-2/checkpoint-82500 --task afqmc --log_file clue_evaluations.csv
python thesis-ntunggal/roberta-training/evaluate_clue.py --model_dir /mnt/storage/ntunggal/finetuned-baidu-82500/cluewsc2020 --tokenizer_dir /mnt/storage/ntunggal/baidu-model-2/checkpoint-82500 --task cluewsc2020 --log_file clue_evaluations.csv
python thesis-ntunggal/roberta-training/evaluate_clue.py --model_dir /mnt/storage/ntunggal/finetuned-baidu-82500/csl --tokenizer_dir /mnt/storage/ntunggal/baidu-model-2/checkpoint-82500 --task csl --log_file clue_evaluations.csv
python thesis-ntunggal/roberta-training/evaluate_clue.py --model_dir /mnt/storage/ntunggal/finetuned-baidu-82500/iflytek --tokenizer_dir /mnt/storage/ntunggal/baidu-model-2/checkpoint-82500 --task iflytek --log_file clue_evaluations.csv
python thesis-ntunggal/roberta-training/evaluate_clue.py --model_dir /mnt/storage/ntunggal/finetuned-baidu-82500/ocnli --tokenizer_dir /mnt/storage/ntunggal/baidu-model-2/checkpoint-82500 --task ocnli --log_file clue_evaluations.csv
python thesis-ntunggal/roberta-training/evaluate_clue.py --model_dir /mnt/storage/ntunggal/finetuned-baidu-82500/tnews --tokenizer_dir /mnt/storage/ntunggal/baidu-model-2/checkpoint-82500 --task tnews --log_file clue_evaluations.csv