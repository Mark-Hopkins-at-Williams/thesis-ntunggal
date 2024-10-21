#!/bin/sh
#SBATCH -c 1                
#SBATCH -t 0-24:00          
#SBATCH -p dl               
#SBATCH --mem=10G           
#SBATCH -o log_%j.out  
#SBATCH -e log_%j.err
#SBATCH --gres=gpu:1
python thesis-ntunggal/roberta-training/finetune.py /mnt/storage/ntunggal/baidu-model/checkpoint-56263 /mnt/storage/ntunggal/finetuned-baidu/baidu-ocnli ocnli

#python thesis-ntunggal/roberta-training/finetune.py KoichiYasuoka/roberta-base-chinese /mnt/storage/ntunggal/finetuned-roberta-base-chinese/roberta-base-iflytek iflytek
#python thesis-ntunggal/roberta-training/finetune.py KoichiYasuoka/roberta-base-chinese /mnt/storage/ntunggal/finetuned-roberta-base-chinese/roberta-base-cluewsc2020 cluewsc2020
#python thesis-ntunggal/roberta-training/finetune.py KoichiYasuoka/roberta-base-chinese /mnt/storage/ntunggal/finetuned-roberta-base-chinese/roberta-base-afqmc afqmc
#python thesis-ntunggal/roberta-training/finetune.py KoichiYasuoka/roberta-base-chinese /mnt/storage/ntunggal/finetuned-roberta-base-chinese/roberta-base-csl csl
#python thesis-ntunggal/roberta-training/finetune.py KoichiYasuoka/roberta-base-chinese /mnt/storage/ntunggal/finetuned-roberta-base-chinese/roberta-base-ocnli ocnli

#python thesis-ntunggal/roberta-training/finetune.py hfl/chinese-roberta-wwm-ext /mnt/storage/ntunggal/finetuned-chinese-roberta-wwm/wwm-iflytek iflytek
#python thesis-ntunggal/roberta-training/finetune.py hfl/chinese-roberta-wwm-ext /mnt/storage/ntunggal/finetuned-chinese-roberta-wwm/wwm-cluewsc2020 cluewsc2020
#python thesis-ntunggal/roberta-training/finetune.py hfl/chinese-roberta-wwm-ext /mnt/storage/ntunggal/finetuned-chinese-roberta-wwm/wwm-afqmc afqmc
#python thesis-ntunggal/roberta-training/finetune.py hfl/chinese-roberta-wwm-ext /mnt/storage/ntunggal/finetuned-chinese-roberta-wwm/wwm-csl csl
#python thesis-ntunggal/roberta-training/finetune.py hfl/chinese-roberta-wwm-ext /mnt/storage/ntunggal/finetuned-chinese-roberta-wwm/wwm-ocnli ocnli

