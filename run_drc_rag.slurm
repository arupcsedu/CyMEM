#!/bin/bash
#SBATCH --job-name=drc_rag
#SBATCH --nodes=2            # number of nodes
#SBATCH --ntasks-per-node=4     # 4 ranks per node → total 8
#SBATCH -A bii_dsc_community
#SBATCH --cpus-per-task=1
#SBATCH --time=00:30:00
#SBATCH --partition=parallel     # change to your cluster's partition
##SBATCH --gres=gpu:a100:2
#SBATCH --output=drc_rag_%j.out
#SBATCH --error=drc_rag_%j.err


# Optional: if you need GPUs; comment out if CPU-only
##SBATCH --gres=gpu:1

# Load modules / activate env (customize for your site)

# module load anaconda  # or module load python, etc.
# source activate gcylon_env  # or conda activate gcylon_env

module load miniforge/24.3.0-py3.11 

source activate /scratch/djy8hg/env/deep_rc_rag_env

cd /sfs/gpfs/tardis/project/bi_dsc_community/drc_rag
export IDX=/sfs/gpfs/tardis/project/bi_dsc_community/drc_rag/shared_index_wikitext
export RUNS=/sfs/gpfs/tardis/project/bi_dsc_community/drc_rag/runs/global_ranks_$SLURM_NTASKS
mkdir -p $RUNS


# Use srun to launch one Python process per task
#python -m main  --data_dir data/wikitext_eval   --use_faiss   --llm gpt2
#srun --ntasks=$SLURM_NTASKS python -m main  --data_dir data/wikitext_eval   --use_faiss   --llm gpt2

#For running global master(rank 0) and slaves(1-n) parallelism
srun --ntasks=$SLURM_NTASKS python -m main_global \
  --data_dir data/wikitext_eval \
  --global_index_dir $IDX \
  --use_faiss \
  --llm gpt2 \
  --eval_dir data/wikitext_eval \
  --lat_csv $RUNS/latencies.global.csv \
  --lat_json $RUNS/latencies.global.json \
  --llm_stats_json $RUNS/llm_generation_stats.global.json


 



