#!/bin/sh
#SBATCH --ntasks=1
#SBATCH --nodes=1
#SBATCH --job-name=simple
#SBATCH --partition=all
#SBATCH --gres=gpu:4
#SBATCH --mem=240G
#SBATCH --cpus-per-task=40
#SBATCH --time=1-8

hostname
echo -n memory=; ulimit -m
echo -n nproc=; nproc
nvidia-smi