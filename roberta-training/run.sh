#!/bin/sh
#SBATCH -c 1                
#SBATCH -t 0-12:00          
#SBATCH -p dl               
#SBATCH --mem=10G           
#SBATCH -o log_%j.out  
#SBATCH -e log_%j.err
#SBATCH --gres=gpu:1
python thesis-ntunggal/roberta-training/finetune.py /mnt/storage/ntunggal/baidu-model-2/checkpoint-56263 /mnt/storage/ntunggal/baidu-model-2-finetuned
