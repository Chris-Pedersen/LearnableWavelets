#!/bin/bash
#SBATCH --job-name=e2_err_1k        # Job name
#SBATCH --mail-type=END,FAIL             # Mail events (NONE, BEGIN, END, FAIL, ALL)
#SBATCH --mail-user=c.pedersen@nyu.edu   # Where to send mail	
#SBATCH --ntasks=1                       # Run on a single CPU
#SBATCH --output=e2_err_optuna_1k_%j.log        # Standard output and error log
#SBATCH --partition gpu
#SBATCH --mem=200G
#SBATCH --cpus-per-task=10
#SBATCH --gpus-per-task=1
#SBATCH --time=120:00:00

pwd; hostname; date

nvidia-smi
source /mnt/home/cpedersen/miniconda3/etc/profile.d/conda.sh

python3 sn_optuna.py

