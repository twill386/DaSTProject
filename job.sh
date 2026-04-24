#!/bin/bash
#SBATCH --job-name=DaSTProject1
#SBATCH --time=00:30:00
#SBATCH --partition=gpu-preempt
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --gpus-per-node=1
#SBATCH --cpus-per-task=4        
#SBATCH --mem=12G
#SBATCH --error=logs/error-%j.txt
#SBATCH --out=logs/out-%j.txt

source /work/pi_csc592_uri_edu/Thomas/.venv/bin/activate
cd /work/pi_csc592_uri_edu/Thomas/DaSTProject
python dast.py --dataset=mnist --niter=10 --batchSize=500 --alpha=0.2 --beta=0.1 --G_type=1 --save_folder=saved_model