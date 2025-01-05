#!/bin/sh
#SBATCH -c 1                
#SBATCH -t 0-24:00          
#SBATCH -p dl               
#SBATCH --mem=10G           
#SBATCH -o log_%j.out  
#SBATCH -e log_%j.err
#SBATCH --gres=gpu:1
python thesis-ntunggal/roberta-training/finetune.py --model_dir /mnt/storage/ntunggal/baidu-model/checkpoint-56263 --output_dir /mnt/storage/ntunggal/finetuned-baidu-redo --tasks tnews,iflytek,cluewsc2020,afqmc,csl,ocnli --log_file refinetune.csv

