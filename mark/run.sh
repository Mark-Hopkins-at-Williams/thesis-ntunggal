#!/bin/sh
#SBATCH -c 1                
#SBATCH -t 0-24:00          
#SBATCH -p dl               
#SBATCH --mem=10G           
#SBATCH -o log_%j.out  
#SBATCH -e log_%j.err
#SBATCH --gres=gpu:1
python train_roberta.py --data_dir /mnt/storage/hopkins/data/chinese-monolingual/baidubaike/ --output_dir roberta-baidu --note "test pretraining on baidu" --log_file roberta-baidu.csv
