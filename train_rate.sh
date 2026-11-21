#!/bin/bash
#SBATCH --job-name=snn_rate_5seeds
#SBATCH --partition=a100
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=4
#SBATCH --gres=gpu:1
#SBATCH --mem=32G
#SBATCH --time=2-23:00:00
#SBATCH --output=/mnt/fast/nobackup/users/ss06262/snn-deepfake/logs/rate_%j.out
#SBATCH --error=/mnt/fast/nobackup/users/ss06262/snn-deepfake/logs/rate_%j.err
set -euo pipefail
PROJECT=/mnt/fast/nobackup/users/ss06262/snn-deepfake
export PYTHONPATH="$PROJECT/python_deps:${PYTHONPATH:-}"
export PYTHONUNBUFFERED=1
export OMP_NUM_THREADS=4
apptainer exec --nv \
  oras://container-registry.surrey.ac.uk/shared-containers/pytorch:latest \
  python -u "$PROJECT/Deepfake_Audio_Detection_SNN/RateEncoding/train_rate.py"
