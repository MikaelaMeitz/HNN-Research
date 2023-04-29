#!/bin/bash
#SBATCH -q regular
#SBATCH -C haswell
#SBATCH -N 1
#SBATCH --time=12:00:00
#SBATCH --error=%x-%j.err
#SBATCH --output=%x-%j.out

module load python
conda activate myPyTorch
export HDF5_USE_FILE_LOCKING=FALSE
srun python MainExp-Copy1.py 

