#!/bin/bash -l
 
# Slurm parameters
#SBATCH --job-name=job_name
#SBATCH --output=job_name-%j.%N.out
#SBATCH --ntasks=1
#SBATCH --time=1-00:00:00
#SBATCH --gpus=1
 
# Activate everything you need
module load cuda/10.1
# python main.py --train
# python main.py --saved_model /misc/home/RUS_CIP/st170269/git/dl-lab-2020-team12/experiments/human_activity_recognition_2021-02-15T17-47-01-241096/ckpts/best_model/
python main.py --saved_model /misc/home/RUS_CIP/st170269/git/dl-lab-2020-team12/experiments/human_activity_recognition_2021-02-13T15-39-05-136196/ckpts/best_model/