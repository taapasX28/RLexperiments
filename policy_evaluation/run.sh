#!/bin/bash
#SBATCH --array=0-6
#SBATCH --job-name=repn_learning   
#SBATCH --time=41:00:00
#SBATCH --mem-per-cpu=6000M
#SBATCH --account=def-szepesva
#SBATCH --output=repn_learning%A%a.out
#SBATCH --error=repn_learning%A%a.err

python policy_evaluation.py --cfg ./cfg_temp/$SLURM_ARRAY_TASK_ID.json