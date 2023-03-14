#! /bin/bash
#SBATCH -J test
#SBATCH -p shlab_medical_pretrain
#SBATCH --gres=gpu:1
#SBATCH -o /mnt/petrelfs/guoyiqiu/coding/slurm_log/%x-%j.out
#SBATCH -e /mnt/petrelfs/guoyiqiu/coding/slurm_log/%x-%j.out
python main.py