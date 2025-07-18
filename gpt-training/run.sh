#!/bin/sh
#SBATCH -c 1                
#SBATCH -t 7-24:00          
#SBATCH -p dl               
#SBATCH --mem=10G           
#SBATCH -o log_%j.out  
#SBATCH -e log_%j.err
#SBATCH --gres=gpu:1
CUDA_VISIBLE_DEVICES=0 python train_gpt.py experiments/testing