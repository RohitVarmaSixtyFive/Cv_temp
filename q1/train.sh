#!/bin/bash
#SBATCH -A research
#SBATCH -n 9
#SBATCH --gres=gpu:1
#SBATCH -w gnode092
#SBATCH --partition=long
#SBATCH --time=3-00:00:00

source ~/.bashrc
source activate base

python3 compare_configs.py 

# python3 train.py 