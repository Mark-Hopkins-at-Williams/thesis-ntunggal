#!/bin/sh
#SBATCH -c 1                
#SBATCH -t 0-24:00          
#SBATCH -p dl               
#SBATCH --mem=10G           
#SBATCH -o log_%j.out  
#SBATCH -e log_%j.err
#SBATCH --gres=gpu:1
python thesis-ntunggal/roberta-training/train_roberta.py --data_dir /mnt/storage/ntunggal/baidubaike_small.txt --output_dir test-custom-tokenizer --note "testing custom char tokenizer" --log_file test-char-tokenizer.csv
