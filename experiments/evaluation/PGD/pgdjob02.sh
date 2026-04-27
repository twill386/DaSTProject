#!/bin/bash
#SBATCH --job-name=DaSTEvalPGD
#SBATCH --time=01:00:00
#SBATCH --partition=gpu-preempt
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --gpus-per-node=1
#SBATCH --cpus-per-task=4
#SBATCH --mem=12G
#SBATCH --error=logs/error-%j.txt
#SBATCH --out=logs/out-%j.txt

source /work/pi_csc592_uri_edu/Thomas/DaSTProject/.venv/bin/activate
cd /work/pi_csc592_uri_edu/Thomas/DaSTProject/DaST
python test_dast.py --mode=dast --adv=PGD --cuda