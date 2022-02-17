#!/bin/bash
#SBATCH --job-name=sn_optuna    # Job name
#SBATCH --mail-type=END,FAIL          # Mail events (NONE, BEGIN, END, FAIL, ALL)
#SBATCH --mail-user=c.pedersen@nyu.edu     # Where to send mail	
#SBATCH --ntasks=16                    # Run on a single CPU
#SBATCH --output=sn_optuna_%j.log   # Standard output and error log
#SBATCH --partition gpu
#SBATCH --time=10:00:00
#SBATCH --gpus 1
#SBATCH --constraint=v100-32gb


pwd; hostname; date


nvidia-smi

python3 sn_optuna.py

