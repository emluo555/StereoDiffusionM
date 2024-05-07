#!/bin/bash
#SBATCH --job-name=stereoDPT025       # create a short name for your job
#SBATCH --nodes=1                # node count
#SBATCH --ntasks=1               # total number of tasks across all nodes
#SBATCH --cpus-per-task=1        # cpu-cores per task (>1 if multi-threaded tasks)
#SBATCH --mem-per-cpu=40G         # memory per cpu-core (4G is default)
#SBATCH --gres=gpu:1             # number of gpus per node
#SBATCH --time=01:00:00          # total run time limit (HH:MM:SS)
#SBATCH --mail-type=begin        # send email when job begins
#SBATCH --mail-type=end          # send email when job ends
#SBATCH --mail-user=el5267@princeton.edu

module purge
conda deactivate
module load anaconda3/2020.11
conda activate /scratch/gpfs/el5267/stereodiffusiontest/StereoDiffusion/env38

python run_DPTimg2stereo.py --img_path="./images" --depthmodel_path="midas_models/dpt_hybrid-midas-501f0c75.pt"
