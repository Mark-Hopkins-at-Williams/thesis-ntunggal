#!/bin/sh
#SBATCH -c 1                
#SBATCH -t 0-24:00          
#SBATCH -p dl               
#SBATCH --mem=10G           
#SBATCH -o log_%j.out  
#SBATCH -e log_%j.err
#SBATCH --gres=gpu:1
#python finetune.py --model_dir /mnt/storage/ntunggal/baidu-model/checkpoint-56263 --output_dir baidu-cluewsc2020 --tasks cluewsc2020 --note "testing synchronization" --log_file foo.log
python thesis-ntunggal/roberta-training/train_bert.py --data_dir /mnt/storage/ntunggal/baidubaike_small.txt --output_dir /mnt/storage/ntunggal/baidu-model-small --note "testing pretrain" --log_file foo.log