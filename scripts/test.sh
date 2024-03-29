#!/bin/bash
#SBATCH --job-name=train_dllm
#SBATCH --partition=medai
#SBATCH --nodelist=SH-IDC1-10-140-1-154
#SBATCH --quotatype=auto
#SBATCH --output=slurm_log/%x-%j.out
#SBATCH --error=slurm_log/%x-%j.out

CUDA_VISIBLE_DEVICES="0,1,2,3,4,5,6,7" python -c "import gpustat; gpustat.print_gpustat();"