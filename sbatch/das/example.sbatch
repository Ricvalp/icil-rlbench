#!/bin/sh
#SBATCH --ntasks=1
#SBATCH --nodes=1
#SBATCH --job-name=snt-r-exp05-perc-low20
#SBATCH --partition=defq
#SBATCH --gres=gpu:1
#SBATCH --mem=64G
#SBATCH --cpus-per-task=8
#SBATCH --time=2-00:00
#SBATCH --output=/home/valperga/icil-rlbench/sbatch/das/logs/example-%j.out
#SBATCH --error=/home/valperga/icil-rlbench/sbatch/das/logs/example-%j.err

...